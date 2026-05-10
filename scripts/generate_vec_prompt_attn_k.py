#!/usr/bin/env python3
"""Generate post-RoPE K_pos and avg_demand calibration for gated prompt-attn steering.

For each trait we save:
- K_pos: post-RoPE K averaged over the system-prompt token positions of pos prompts.
  Shape: (n_layers, n_kv_heads, head_dim).
- avg_demand: average per-token demand d = mean_h(Q_post_rope[h] . K_pos[h//ng]) / sqrt(d_head)
  over response tokens in pos prompts. Used to center the sigmoid gate at 0.
  Shape: (n_layers,).

No normalization is applied to K_pos (we keep the natural attention-logit scale).
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from generate_vec import get_persona_effective


DEFAULT_MODEL = os.environ.get("MODEL_PATH_OR_ID", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_EXTRACT_DIR = PROJECT_ROOT / "eval_persona_extract" / "Qwen2.5-7B-Instruct"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "vectors" / "prompt_attn_k_pos" / "Qwen2.5-7B-Instruct"
DEFAULT_TRAITS = [
    "apathetic",
    "creative",
    "curious",
    "empathetic",
    "evil",
    "factual",
    "hallucinating",
    "honest",
    "humorous",
    "impolite",
    "optimistic",
    "pessimistic",
    "polite",
    "righteous",
    "sycophantic",
]
SYS_END_MARKER = "<|im_end|>\n"


def compute_k_pos_and_avg_demand(model, tokenizer, prompts, responses, layer_list=None):
    """Single pass: accumulate post-RoPE K over sys positions and post-RoPE Q over response positions.

    Returns:
        k_pos: (n_layers, n_kv_heads, head_dim) — mean post-RoPE K over sys tokens, averaged over samples.
        avg_demand: (n_layers,) — mean per-token demand over response tokens (pos prompts).
    """
    n_layers = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(n_layers))

    nh = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = model.config.hidden_size // nh
    ng = nh // nkv  # query heads per kv head

    texts = [p + a for p, a in zip(prompts, responses)]

    captured_q = {l: None for l in layer_list}
    captured_k = {l: None for l in layer_list}

    def make_q_hook(l):
        def hook(m, inp, out):
            captured_q[l] = out.detach()
        return hook

    def make_k_hook(l):
        def hook(m, inp, out):
            captured_k[l] = out.detach()
        return hook

    handles = []
    for l in layer_list:
        attn = model.model.layers[l].self_attn
        handles.append(attn.q_proj.register_forward_hook(make_q_hook(l)))
        handles.append(attn.k_proj.register_forward_hook(make_k_hook(l)))

    k_sums = {l: None for l in layer_list}
    k_counts = {l: 0 for l in layer_list}
    q_sums = {l: None for l in layer_list}
    q_counts = {l: 0 for l in layer_list}

    try:
        for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
            if SYS_END_MARKER not in prompt:
                raise ValueError("Could not find system prompt end marker in prompt text")
            sys_end_pos = prompt.index(SYS_END_MARKER) + len(SYS_END_MARKER)
            sys_text = prompt[:sys_end_pos]
            sys_token_len = len(tokenizer.encode(sys_text, add_special_tokens=False))
            prompt_token_len = len(tokenizer.encode(prompt, add_special_tokens=False))

            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            T = inputs["input_ids"].shape[1]
            position_ids = torch.arange(T, device=model.device).unsqueeze(0)

            for l in layer_list:
                captured_q[l] = None
                captured_k[l] = None

            with torch.no_grad():
                model(**inputs, output_hidden_states=False, output_attentions=False, use_cache=False)

            for l in layer_list:
                q_raw = captured_q[l]  # (1, T, nh*hd) on layer device
                k_raw = captured_k[l]  # (1, T, nkv*hd)
                dev = q_raw.device
                # (1, H, T, D)
                q = q_raw.view(1, T, nh, hd).transpose(1, 2)
                k = k_raw.view(1, T, nkv, hd).transpose(1, 2)
                cos, sin = model.model.rotary_emb(k, position_ids.to(dev))
                cos = cos.to(dev)
                sin = sin.to(dev)
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                q_rope = q_rope.float().cpu()
                k_rope = k_rope.float().cpu()

                # K_pos: sum over sys positions
                sys_k_sum = k_rope[0, :, :sys_token_len, :].sum(dim=1)  # (nkv, hd)
                if k_sums[l] is None:
                    k_sums[l] = sys_k_sum
                else:
                    k_sums[l] += sys_k_sum
                k_counts[l] += sys_token_len

                # Q_resp: sum over response positions (prompt_token_len:)
                if prompt_token_len < T:
                    q_resp_sum = q_rope[0, :, prompt_token_len:, :].sum(dim=1)  # (nh, hd)
                    n_resp = T - prompt_token_len
                    if q_sums[l] is None:
                        q_sums[l] = q_resp_sum
                    else:
                        q_sums[l] += q_resp_sum
                    q_counts[l] += n_resp
    finally:
        for h in handles:
            h.remove()

    k_pos = torch.stack(
        [k_sums[l] / max(k_counts[l], 1) for l in range(n_layers)], dim=0
    )  # (n_layers, nkv, hd)

    # avg_demand per layer: (1/nh) sum_h (Q_mean[h] . K_pos[h//ng]) / sqrt(hd)
    avg_demand_list = []
    for l in range(n_layers):
        if q_sums[l] is None or q_counts[l] == 0:
            avg_demand_list.append(torch.tensor(0.0))
            continue
        q_mean = q_sums[l] / q_counts[l]  # (nh, hd)
        k_per_q = k_pos[l].repeat_interleave(ng, dim=0)  # (nh, hd)
        d = (q_mean * k_per_q).sum(dim=-1).mean() / (hd ** 0.5)
        avg_demand_list.append(d)
    avg_demand = torch.stack(avg_demand_list, dim=0)  # (n_layers,)

    return k_pos, avg_demand


def save_trait_prompt_k_pos(model, tokenizer, pos_path, neg_path, trait, save_dir, threshold=50):
    # We still read neg_path to reuse the filtering logic, but only use pos side.
    _, _, pos_prompts, _, pos_responses, _ = get_persona_effective(
        pos_path,
        neg_path,
        trait,
        threshold,
    )

    if len(pos_prompts) == 0:
        raise ValueError(f"No effective pos prompt/response pairs found for trait={trait} at threshold={threshold}")

    k_pos, avg_demand = compute_k_pos_and_avg_demand(model, tokenizer, pos_prompts, pos_responses)

    os.makedirs(save_dir, exist_ok=True)
    k_path = Path(save_dir) / f"{trait}_prompt_key_pos.pt"
    d_path = Path(save_dir) / f"{trait}_avg_demand.pt"
    torch.save(k_pos, k_path)
    torch.save(avg_demand, d_path)

    print(f"Saved {trait} -> {k_path}  shape={tuple(k_pos.shape)}")
    print(f"Saved {trait} -> {d_path}  shape={tuple(avg_demand.shape)}")
    print(f"  avg_demand layer19 = {avg_demand[19].item():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate post-RoPE K_pos + avg_demand for gated prompt-attn steering."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--extract_dir", default=str(DEFAULT_EXTRACT_DIR))
    parser.add_argument("--save_dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for trait in args.traits:
        pos_csv = extract_dir / f"{trait}_pos_instruct.csv"
        neg_csv = extract_dir / f"{trait}_neg_instruct.csv"
        k_out = save_dir / f"{trait}_prompt_key_pos.pt"
        d_out = save_dir / f"{trait}_avg_demand.pt"

        if not pos_csv.exists():
            print(f"SKIP {trait}: missing {pos_csv}")
            continue
        if not neg_csv.exists():
            print(f"SKIP {trait}: missing {neg_csv}")
            continue
        if k_out.exists() and d_out.exists() and not args.overwrite:
            print(f"SKIP {trait}: vectors already exist at {save_dir}")
            continue

        print(f"=== Generating post-RoPE K_pos + avg_demand for: {trait} ===")
        save_trait_prompt_k_pos(
            model=model,
            tokenizer=tokenizer,
            pos_path=str(pos_csv),
            neg_path=str(neg_csv),
            trait=trait,
            save_dir=str(save_dir),
            threshold=args.threshold,
        )
        print()

    print(f"Done. Saved vectors to {save_dir}/")


if __name__ == "__main__":
    main()
