from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    

def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu())
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg

def get_attn_delta_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(0, max_layer)) # Extract from all layers
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    
    captured_attn = {layer: [] for layer in layer_list}
    handles = []
    
    def get_hook(layer_idx):
        def hook(module, inputs, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            captured_attn[layer_idx].append(attn_out.detach().cpu())
        return hook
    
    for layer in layer_list:
        # Qwen2 uses model.model.layers
        module = model.model.layers[layer].self_attn
        handles.append(module.register_forward_hook(get_hook(layer)))
            
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        
        for layer in layer_list:
            captured_attn[layer].clear()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)

        for layer in layer_list:
            attn_out = captured_attn[layer][0]
            # Only extract response
            response_avg[layer].append(attn_out[:, prompt_len:, :].mean(dim=1))
                
    for handle in handles:
        handle.remove()
        
    for layer in layer_list:
        if len(response_avg[layer]) > 0:
            response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg


def get_prompt_attn_delta_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    """Extract the attention output contributed ONLY by system prompt tokens.

    For each response token i at layer l, computes:
        sys_contribution_i = o_proj( concat_heads( sum_{j in sys_tokens} alpha_ij * V_j ) )
    where alpha_ij are post-softmax attention weights (already normalized over
    the full sequence) and V_j are value states.

    This isolates the direct influence of the system prompt on attention output,
    excluding the effect of divergent generated content.

    Requires output_attentions=True (falls back to eager attention, slower but exact).
    Uses two hooks per layer:
      1. v_proj hook — captures raw value states
      2. self_attn hook — grabs attn_weights from output, combines with captured V,
         computes sys_contribution, applies o_proj, discards intermediates
    """
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(0, max_layer))

    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = model.config.hidden_size // num_heads

    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]
    texts = [p + a for p, a in zip(prompts, responses)]

    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        # Find system prompt token boundary:
        # prompt = "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n..."
        sys_end_marker = "<|im_end|>\n"
        sys_end_pos = prompt.index(sys_end_marker) + len(sys_end_marker)
        sys_text = prompt[:sys_end_pos]
        sys_token_len = len(tokenizer.encode(sys_text, add_special_tokens=False))

        # Per-sample hook state (captured_v is consumed by attn_hook via pop)
        captured_v = {}
        sys_contributions = {}
        handles = []

        def make_v_hook(layer_idx):
            def hook(module, input, output):
                captured_v[layer_idx] = output.detach()
            return hook

        def make_attn_hook(layer_idx, sys_len):
            def hook(module, input, output):
                attn_weights = output[1]  # (batch, num_heads, seq_len, seq_len)
                v_raw = captured_v.pop(layer_idx)  # (batch, seq_len, num_kv_heads * head_dim)

                bsz, seq_len_v = v_raw.shape[0], v_raw.shape[1]
                # Reshape to (batch, num_kv_heads, seq_len, head_dim) and GQA-repeat
                v = v_raw.view(bsz, seq_len_v, num_kv_heads, head_dim).transpose(1, 2)
                v = repeat_kv(v, num_kv_groups)  # (batch, num_heads, seq_len, head_dim)

                # Attention output from system prompt tokens only
                sys_w = attn_weights[:, :, :, :sys_len]   # (batch, num_heads, seq_len, sys_len)
                sys_v = v[:, :, :sys_len, :]               # (batch, num_heads, sys_len, head_dim)
                sys_out = sys_w @ sys_v                     # (batch, num_heads, seq_len, head_dim)

                # Reshape back and apply o_proj (linear, so decomposition is exact)
                sys_out = sys_out.transpose(1, 2).reshape(bsz, -1, num_heads * head_dim)
                sys_out = module.o_proj(sys_out)

                sys_contributions[layer_idx] = sys_out.detach().cpu()
            return hook

        for layer in layer_list:
            handles.append(
                model.model.layers[layer].self_attn.v_proj.register_forward_hook(make_v_hook(layer)))
            handles.append(
                model.model.layers[layer].self_attn.register_forward_hook(make_attn_hook(layer, sys_token_len)))

        with torch.no_grad():
            model(**inputs, output_attentions=True, output_hidden_states=False)

        for handle in handles:
            handle.remove()

        for layer in layer_list:
            sys_out = sys_contributions[layer]
            # Average over response token positions
            response_avg[layer].append(sys_out[:, prompt_len:, :].mean(dim=1))

    for layer in layer_list:
        if len(response_avg[layer]) > 0:
            response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg


def get_mlp_delta_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(0, max_layer))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]

    captured_mlp = {layer: [] for layer in layer_list}
    handles = []

    def get_hook(layer_idx):
        def hook(module, inputs, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            captured_mlp[layer_idx].append(mlp_out.detach().cpu())
        return hook

    for layer in layer_list:
        module = model.model.layers[layer].mlp
        handles.append(module.register_forward_hook(get_hook(layer)))

    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        for layer in layer_list:
            captured_mlp[layer].clear()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)

        for layer in layer_list:
            mlp_out = captured_mlp[layer][0]
            response_avg[layer].append(mlp_out[:, prompt_len:, :].mean(dim=1))

    for handle in handles:
        handle.remove()

    for layer in layer_list:
        if len(response_avg[layer]) > 0:
            response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg


import pandas as pd
import os

def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    # Normalize column names (some CSVs have trailing spaces)
    persona_pos.columns = persona_pos.columns.str.strip()
    persona_neg.columns = persona_neg.columns.str.strip()
    mask = (persona_pos[trait] >=threshold) & (persona_neg[trait] < 100-threshold) & (persona_pos["coherence"] >= 50) & (persona_neg["coherence"] >= 50)

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()    
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses


def save_persona_vector(model_name, pos_path, neg_path, trait, save_dir, threshold=50, extraction_method="hidden_state"):
    # prompt_attn_delta reads attn_weights from self_attn output, which is None
    # under SDPA / FlashAttn even with output_attentions=True. Force eager.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses = get_persona_effective(pos_path, neg_path, trait, threshold)

    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    if extraction_method == "attn_delta":
        persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_attn_delta_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
        persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_attn_delta_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)
    elif extraction_method == "prompt_attn_delta":
        persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_prompt_attn_delta_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
        persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_prompt_attn_delta_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)
    elif extraction_method == "mlp_delta":
        persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_mlp_delta_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
        persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_mlp_delta_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)
    else:
        persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
        persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)
    


    persona_effective_prompt_avg_diff = None
    persona_effective_prompt_last_diff = None
    
    if extraction_method in ("attn_delta", "mlp_delta", "prompt_attn_delta"):
        n_layers = model.config.num_hidden_layers
        persona_effective_response_avg_diff = torch.stack([persona_effective_response_avg["pos"][l].mean(0).float() - persona_effective_response_avg["neg"][l].mean(0).float() for l in range(n_layers)], dim=0)
    else:
        persona_effective_prompt_avg_diff = torch.stack([persona_effective_prompt_avg["pos"][l].mean(0).float() - persona_effective_prompt_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_avg["pos"]))], dim=0)
        persona_effective_response_avg_diff = torch.stack([persona_effective_response_avg["pos"][l].mean(0).float() - persona_effective_response_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_response_avg["pos"]))], dim=0)
        persona_effective_prompt_last_diff = torch.stack([persona_effective_prompt_last["pos"][l].mean(0).float() - persona_effective_prompt_last["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_last["pos"]))], dim=0)

    os.makedirs(save_dir, exist_ok=True)

    suffix_map = {"attn_delta": "_attn_delta", "mlp_delta": "_mlp_delta", "prompt_attn_delta": "_prompt_attn_delta"}
    suffix = suffix_map.get(extraction_method, "")

    if persona_effective_prompt_avg_diff is not None:
        torch.save(persona_effective_prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff{suffix}.pt")
    torch.save(persona_effective_response_avg_diff, f"{save_dir}/{trait}_response_avg_diff{suffix}.pt")
    if persona_effective_prompt_last_diff is not None:
        torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff{suffix}.pt")

    print(f"Persona vectors saved to {save_dir}")    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--extraction_method", type=str, default="hidden_state", choices=["hidden_state", "attn_delta", "mlp_delta", "prompt_attn_delta"])
    args = parser.parse_args()
    save_persona_vector(args.model_name, args.pos_path, args.neg_path, args.trait, args.save_dir, args.threshold, args.extraction_method)