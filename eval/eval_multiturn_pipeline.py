"""Pipelined multi-trait multi-turn steering evaluation with batched generation.

Generation (GPU) and judging (API) run concurrently in a pipeline.
Within each config, all conversations for a given turn are batched together
using model.generate() with left-padding.

Each config entry carries a ``method`` field selecting the steerer:
  - "residual" (default): residual-stream steering. Required fields:
    ``layers`` (e.g. "20:2.0"), ``vector_path``. Optional: ``target_module``,
    ``zero_indexed_vectors``, ``persona_instruction_type``. Omit ``layers``
    for an unsteered baseline.
  - "gcad": gated cropped attention-delta steering. Required fields:
    ``layer_indices`` (list[int]) or ``layer_idx`` (int), ``base_coef``,
    ``scale``. Optional: ``inverse``, ``debug``. Vectors are loaded from
    ``vectors/prompt_attn_{delta,k_pos}/<model_name>/<trait>_*.pt``.

Usage:
    python -m eval.eval_multiturn_pipeline \
        --model /path/to/model \
        --configs configs.json \
        --n_turns 5 --n_groups 5 --n_samples 10
"""

import os
import sys
import asyncio
import json
import random
import threading
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from judge import OpenAiJudge
from activation_steer import make_steerer
from eval.model_utils import load_model
from eval.prompts import Prompts
from config import setup_credentials
import logging
import fire

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

config = setup_credentials()


def sample_question_groups(questions, n_groups, n_turns, seed=42):
    if n_turns == 1 and n_groups == len(questions):
        return [[q] for q in questions]
    rng = random.Random(seed)
    groups = []
    for _ in range(n_groups):
        group = rng.sample(questions, min(n_turns, len(questions)))
        rng.shuffle(group)
        groups.append(group)
    return groups


def generate_batched_multiturn(
    model, tokenizer, question_groups, steerer_ctx, n_samples,
    max_tokens_per_turn=500, temperature=1.0, system_prompt=None, batch_size=16,
):
    """Generate multi-turn conversations with batched model.generate() per turn.

    ``steerer_ctx`` is a context manager (e.g. an ActivationSteerer or a gated
    steerer) that is entered once per turn and wraps the batched generation.
    Pass ``torch.no_grad()`` for an unsteered baseline.

    For each turn, all conversations (n_groups × n_samples) are batched together.
    After each turn, answers are appended to message histories for the next turn.

    Returns list of dicts with group_idx, sample_idx, turn_idx, question, answer, n_tokens.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_groups = len(question_groups)
    n_turns = len(question_groups[0])
    total_convs = n_groups * n_samples

    # Initialize message histories for all conversations
    histories = []
    conv_meta = []
    for gi, questions in enumerate(question_groups):
        for si in range(n_samples):
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            histories.append(msgs)
            conv_meta.append({"gi": gi, "si": si, "questions": questions})

    all_rows = []

    for turn_idx in range(n_turns):
        # Build prompts for this turn
        for ci in range(total_convs):
            q = conv_meta[ci]["questions"][turn_idx]
            histories[ci].append({"role": "user", "content": q})

        prompts = [
            tokenizer.apply_chat_template(h, tokenize=False, add_generation_prompt=True)
            for h in histories
        ]

        # Batched generation
        all_outputs = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        with steerer_ctx:
            with torch.no_grad():
                for bi, i in enumerate(range(0, len(prompts), batch_size)):
                    batch = prompts[i:i + batch_size]
                    tokenized = tokenizer(batch, return_tensors="pt", padding=True)
                    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
                    output = model.generate(
                        **tokenized,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else None,
                        max_new_tokens=max_tokens_per_turn,
                        use_cache=True,
                    )
                    prompt_len = tokenized["input_ids"].shape[1]
                    decoded = [
                        tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
                        for o in output
                    ]
                    all_outputs.extend(decoded)

        # Append answers to histories and collect rows
        for ci in range(total_convs):
            answer = all_outputs[ci]
            histories[ci].append({"role": "assistant", "content": answer})
            q = conv_meta[ci]["questions"][turn_idx]
            all_rows.append({
                "group_idx": conv_meta[ci]["gi"],
                "sample_idx": conv_meta[ci]["si"],
                "turn_idx": turn_idx,
                "question": q,
                "answer": answer,
                "n_tokens": len(tokenizer.encode(answer, add_special_tokens=False)),
            })

        print(f"    Turn {turn_idx}: generated {total_convs} answers ({n_batches} batches)")

    return all_rows


def build_instructions(model, layers_str, vector_path, target_module, zero_indexed_vectors):
    layer_configs = []
    for l_c in str(layers_str).split(","):
        parts = l_c.split(":")
        layer_configs.append((int(parts[0]), float(parts[1])))
    layer_offset = 0 if zero_indexed_vectors else -1
    instructions = []
    for layer_idx, coef in layer_configs:
        vector = torch.load(vector_path, weights_only=False)[layer_idx]
        instructions.append({
            "steering_vector": vector,
            "coeff": coef,
            "layer_idx": layer_idx + layer_offset,
            "positions": "all",
            "target_module": target_module,
        })
    return instructions


def _build_gcad_ctx(model, cfg):
    """Build a GCAD steerer context for cfg. Lazy-imports gated_steerer."""
    from gated_steerer import GatedPromptAttnDeltaSteerer, MultiLayerGatedSteerer

    model_name = Path(cfg.get("model_name", "Qwen2.5-7B-Instruct")).name
    trait = cfg["trait"]

    layer_indices = cfg.get("layer_indices")
    if layer_indices is None:
        layer_indices = [int(cfg["layer_idx"])]

    v_path = PROJECT_ROOT / "vectors" / "prompt_attn_delta" / model_name / \
        f"{trait}_response_avg_diff_prompt_attn_delta.pt"
    k_path = PROJECT_ROOT / "vectors" / "prompt_attn_k_pos" / model_name / \
        f"{trait}_prompt_key_pos.pt"
    d_path = PROJECT_ROOT / "vectors" / "prompt_attn_k_pos" / model_name / \
        f"{trait}_avg_demand.pt"

    v_full = torch.load(v_path, map_location="cpu", weights_only=False)
    k_full = torch.load(k_path, map_location="cpu", weights_only=False)
    d_full = torch.load(d_path, map_location="cpu", weights_only=False)

    base_coef = float(cfg["base_coef"])
    scale = float(cfg["scale"])
    inverse = bool(cfg.get("inverse", False))
    debug = bool(cfg.get("debug", False))

    if len(layer_indices) == 1:
        l = layer_indices[0]
        return GatedPromptAttnDeltaSteerer(
            model, v_vector=v_full[l].float(), k_pos=k_full[l].float(),
            layer_idx=l,
            base_coef=base_coef, scale=scale,
            avg_demand=float(d_full[l].item()),
            inverse=inverse, debug=debug,
        )

    v_dict = {l: v_full[l].float() for l in layer_indices}
    k_dict = {l: k_full[l].float() for l in layer_indices}
    d_dict = {l: float(d_full[l].item()) for l in layer_indices}

    return MultiLayerGatedSteerer(
        model, v_vectors=v_dict, k_pos_vectors=k_dict, avg_demands=d_dict,
        layer_indices=layer_indices,
        base_coef=base_coef, scale=scale,
        inverse=inverse, debug=debug,
    )


def build_steerer_ctx(model, cfg):
    """Dispatch on cfg["method"] to build a steerer context manager.

    "residual" (default) — reads cfg["layers"] and cfg["vector_path"].
        If "layers" is missing, returns a no-op torch.no_grad() (baseline).
    "gcad" — reads cfg["layer_indices"] (or "layer_idx"), "base_coef",
        "scale". Loads vectors from PROJECT_ROOT/vectors/prompt_attn_*.
    """
    method = cfg.get("method", "residual")
    if method == "gcad":
        return _build_gcad_ctx(model, cfg)
    if method == "residual":
        layers_str = cfg.get("layers")
        if layers_str is None:
            return torch.no_grad()
        instructions = build_instructions(
            model, layers_str, cfg["vector_path"],
            cfg.get("target_module", "layer"),
            cfg.get("zero_indexed_vectors", False),
        )
        return make_steerer(model, {"instructions": instructions})
    raise ValueError(f"unknown method: {method!r}")


async def judge_one(question, answer, trait_eval_prompt, judge_model, sem):
    async with sem:
        tj = OpenAiJudge(judge_model, trait_eval_prompt, eval_type="0_100")
        cj = OpenAiJudge(judge_model, Prompts["coherence_0_100"], eval_type="0_100")
        ts, cs = None, None
        for attempt in range(5):
            try:
                if ts is None:
                    ts = await tj.judge(question=question, answer=answer)
                if cs is None:
                    cs = await cj.judge(question=question, answer=answer)
                break
            except:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)
        return ts, cs


async def judge_trait_answers(rows, trait_eval_prompt, judge_model, sem, trait_name):
    total = len(rows)
    done = [0]
    async def judge_and_track(row):
        result = await judge_one(row["question"], row["answer"], trait_eval_prompt, judge_model, sem)
        done[0] += 1
        if done[0] % 50 == 0 or done[0] == total:
            print(f"  Judging {trait_name}: {done[0]}/{total}")
        return result
    scores = await asyncio.gather(*[judge_and_track(r) for r in rows])
    for row, (ts, cs) in zip(rows, scores):
        row["trait_score"] = ts
        row["coherence"] = cs
    return rows


def save_and_summarize(rows, output_path, trait, n_turns):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Saved {trait} → {output_path}")
    for turn in range(n_turns):
        t_df = df[df["turn_idx"] == turn]
        if len(t_df) == 0:
            continue
        print(f"    Turn {turn}: trait={t_df['trait_score'].mean():.2f}±{t_df['trait_score'].std():.2f}, "
              f"coh={t_df['coherence'].mean():.2f}±{t_df['coherence'].std():.2f}, n={len(t_df)}")


def main(
    model=None,
    configs=None,
    n_turns=5,
    n_groups=5,
    n_samples=10,
    max_tokens_per_turn=500,
    batch_size=64,
    judge_model="gpt-4.1-mini-2025-04-14",
    overwrite=False,
    temperature=1.0,
    seed=42,
):
    if model is None:
        model = os.environ.get("MODEL_PATH_OR_ID", "Qwen/Qwen2.5-7B-Instruct")
    if configs is None:
        raise ValueError("--configs is required")

    with open(configs) as f:
        cfg_list = json.load(f)

    model_name = Path(model).name
    for c in cfg_list:
        c.setdefault("overwrite", overwrite)
        c.setdefault("model_name", model_name)

    print(f"Pipeline: {len(cfg_list)} configs, {n_groups}g × {n_turns}t × {n_samples}s")
    print(f"Loading model...")
    llm, tokenizer = load_model(model)

    # Start async event loop in background thread for judging
    loop = asyncio.new_event_loop()
    sem = asyncio.Semaphore(20)

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    pending_futures = []
    completed = [0]
    total_configs = 0

    for ci, cfg in enumerate(cfg_list):
        trait_name = cfg["trait"]
        out_path = cfg["output_path"]

        if os.path.exists(out_path) and not cfg.get("overwrite", False):
            print(f"  SKIP {trait_name}: {out_path} exists")
            continue

        print(f"\n[{ci+1}/{len(cfg_list)}] Generating: {trait_name} → {out_path}")

        trait_data = json.load(open(PROJECT_ROOT / f"data/trait_data_eval/{trait_name}.json"))
        trait_eval_prompt = trait_data["eval_prompt"]
        all_questions = trait_data["questions"]

        system_prompt = None
        pit = cfg.get("persona_instruction_type")
        if pit is not None:
            instruction = trait_data["instruction"][0][pit]
            a_or_an = "an" if trait_name[0].lower() in "aeiou" else "a"
            system_prompt = f"You are {a_or_an} {trait_name} assistant. {instruction}"

        question_groups = sample_question_groups(all_questions, n_groups, n_turns, seed)

        steerer_ctx = build_steerer_ctx(llm, cfg)

        rows = generate_batched_multiturn(
            llm, tokenizer, question_groups, steerer_ctx, n_samples,
            max_tokens_per_turn=max_tokens_per_turn,
            temperature=temperature,
            system_prompt=system_prompt,
            batch_size=batch_size,
        )

        print(f"  Generated {len(rows)} answers for {trait_name}. Submitting to judge...")
        total_configs += 1

        async def judge_and_save(rows, trait_eval_prompt, out_path, trait_name):
            await judge_trait_answers(rows, trait_eval_prompt, judge_model, sem, trait_name)
            save_and_summarize(rows, out_path, trait_name, n_turns)
            completed[0] += 1
            print(f"  [{completed[0]}/{total_configs}] Saved {trait_name}")

        future = asyncio.run_coroutine_threadsafe(
            judge_and_save(rows, trait_eval_prompt, out_path, trait_name),
            loop,
        )
        pending_futures.append(future)

    print(f"\nGeneration done. Waiting for {len(pending_futures)} judging jobs...")
    for future in pending_futures:
        future.result()

    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join()
    print(f"\nAll done. {len(pending_futures)} configs evaluated.")


if __name__ == "__main__":
    fire.Fire(main)
