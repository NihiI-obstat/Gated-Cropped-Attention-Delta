"""Gated prompt-attn-delta steerer (post-RoPE, K_pos, centered gate).

Single-layer: GatedPromptAttnDeltaSteerer (backward compat).
Multi-layer:  MultiLayerGatedSteerer — hooks multiple layers, each with its own
              V, K_pos, avg_demand; shared base_coef and scale.

Per-token demand (natural attention logit) at layer l:
    d_i[l] = (1/nh) * sum_h ( Q_post_rope[i,h] . K_pos[l, h//ng] ) / sqrt(d_head)

Per-token coefficient:
    c_i[l] = 2 * base_coef * sigmoid(sign * scale * (d_i[l] - avg_demand[l]))

Q is captured from q_proj output (pre-RoPE); RoPE applied at hook time using
position_embeddings from the decoder layer.
"""

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class GatedPromptAttnDeltaSteerer:
    def __init__(
        self,
        model: torch.nn.Module,
        v_vector: torch.Tensor,           # (hidden,)
        k_pos: torch.Tensor,              # (num_kv_heads, head_dim) — post-RoPE K_pos mean
        *,
        layer_idx: int,
        base_coef: float,
        scale: float,
        avg_demand: float = 0.0,
        inverse: bool = False,
        debug: bool = False,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.base_coef = float(base_coef)
        self.scale = float(scale)
        self.avg_demand = float(avg_demand)
        self.inverse = bool(inverse)
        self.debug = debug

        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.hidden_size // self.num_heads

        target_mod = model.model.layers[layer_idx].self_attn.q_proj
        target_dtype = target_mod.weight.dtype

        if v_vector.ndim != 1 or v_vector.numel() != cfg.hidden_size:
            raise ValueError(
                f"v_vector must be 1-D length {cfg.hidden_size}, got {v_vector.shape}")
        if k_pos.shape != (self.num_kv_heads, self.head_dim):
            raise ValueError(
                f"k_pos must be ({self.num_kv_heads}, {self.head_dim}), got {k_pos.shape}")

        self.v_vector = v_vector.to(dtype=target_dtype)
        # Broadcast K across query heads once: (nh, hd)
        self.k_per_qhead = k_pos.repeat_interleave(
            self.num_kv_groups, dim=0).to(dtype=target_dtype)

        self._q_captured = None       # pre-RoPE Q, shape (B, T, nh, hd)
        self._pos_emb = None          # (cos, sin), each (B, T, hd)
        self._handles = []
        self._sign = -1.0 if inverse else 1.0

    # ── Hooks ──
    def _attn_pre_hook(self, module, args, kwargs):
        # Grab position_embeddings. Qwen2DecoderLayer passes it as a kwarg.
        pe = kwargs.get("position_embeddings", None)
        if pe is None and len(args) >= 2:
            pe = args[1]
        self._pos_emb = pe

    def _q_hook(self, module, inputs, output):
        # q_proj output: (B, T, nh*hd)
        bsz, seq_len, _ = output.shape
        self._q_captured = output.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _attn_hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            attn_out = output[0]
            rest = output[1:]
        else:
            attn_out = output
            rest = None

        q = self._q_captured
        pe = self._pos_emb
        if q is None or pe is None:
            return output

        cos, sin = pe
        # apply_rotary_pos_emb expects (B, H, T, D)
        q_bhtd = q.transpose(1, 2)
        cos_d = cos.to(device=q.device, dtype=q.dtype)
        sin_d = sin.to(device=q.device, dtype=q.dtype)
        q_rope, _ = apply_rotary_pos_emb(q_bhtd, q_bhtd, cos_d, sin_d)
        q_rope = q_rope.transpose(1, 2)  # (B, T, nh, hd)

        k_ph = self.k_per_qhead.to(device=q_rope.device, dtype=q_rope.dtype)
        v_vec = self.v_vector.to(device=q_rope.device, dtype=q_rope.dtype)

        # demand[b, t] = mean_h (Q_rope[b,t,h,:] . K_per_qhead[h,:]) / sqrt(d_head)
        dot_per_head = (q_rope * k_ph).sum(dim=-1)           # (B, T, H)
        demand = dot_per_head.mean(dim=-1) / (self.head_dim ** 0.5)  # (B, T)

        centered = demand - self.avg_demand
        coef = 2.0 * self.base_coef * torch.sigmoid(self._sign * self.scale * centered)  # (B, T)

        steered = attn_out + coef.unsqueeze(-1) * v_vec

        if self.debug:
            with torch.no_grad():
                print(f"[gated L{self.layer_idx}] demand: mean={demand.mean():.3f} "
                      f"std={demand.std():.3f} min={demand.min():.3f} max={demand.max():.3f}"
                      f" (avg_demand={self.avg_demand:.3f}) | "
                      f"coef: mean={coef.mean():.3f} range=[{coef.min():.3f}, {coef.max():.3f}]")

        self._q_captured = None
        self._pos_emb = None

        if rest is not None:
            return (steered,) + tuple(rest)
        return steered

    # ── Context manager ──
    def __enter__(self):
        self._q_captured = None
        self._pos_emb = None
        attn = self.model.model.layers[self.layer_idx].self_attn
        self._handles.append(
            attn.register_forward_pre_hook(self._attn_pre_hook, with_kwargs=True))
        self._handles.append(attn.q_proj.register_forward_hook(self._q_hook))
        self._handles.append(attn.register_forward_hook(self._attn_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._q_captured = None
        self._pos_emb = None

    def remove(self):
        self.__exit__(None, None, None)


class MultiLayerGatedSteerer:
    """Gated steering across multiple layers. Each layer has its own V, K_pos, avg_demand."""

    def __init__(
        self,
        model: torch.nn.Module,
        v_vectors: dict,        # {layer_idx: (hidden,)}
        k_pos_vectors: dict,    # {layer_idx: (nkv, hd)}
        avg_demands: dict,      # {layer_idx: float}
        *,
        layer_indices: list,
        base_coef: float,
        scale: float,
        inverse: bool = False,
        debug: bool = False,
    ):
        self.model = model
        self.layer_indices = sorted(layer_indices)
        self.base_coef = float(base_coef)
        self.scale = float(scale)
        self.inverse = bool(inverse)
        self.debug = debug

        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.hidden_size // self.num_heads
        self._sign = -1.0 if inverse else 1.0

        # Per-layer tensors (kept on CPU, moved lazily in hooks)
        self._v = {}
        self._k = {}
        self._avg_d = {}
        for l in self.layer_indices:
            target_dtype = model.model.layers[l].self_attn.q_proj.weight.dtype
            v = v_vectors[l]
            k = k_pos_vectors[l]
            if v.ndim != 1 or v.numel() != cfg.hidden_size:
                raise ValueError(f"v_vectors[{l}] must be 1-D length {cfg.hidden_size}, got {v.shape}")
            if k.shape != (self.num_kv_heads, self.head_dim):
                raise ValueError(f"k_pos_vectors[{l}] must be ({self.num_kv_heads}, {self.head_dim}), got {k.shape}")
            self._v[l] = v.to(dtype=target_dtype)
            self._k[l] = k.repeat_interleave(self.num_kv_groups, dim=0).to(dtype=target_dtype)
            self._avg_d[l] = float(avg_demands[l])

        # Per-layer captured state
        self._q_captured = {}
        self._pos_emb = {}
        self._handles = []

    # ── Hook factories (closure over layer index) ──
    def _make_pre_hook(self, l):
        def hook(module, args, kwargs):
            pe = kwargs.get("position_embeddings", None)
            if pe is None and len(args) >= 2:
                pe = args[1]
            self._pos_emb[l] = pe
        return hook

    def _make_q_hook(self, l):
        def hook(module, inputs, output):
            bsz, seq_len, _ = output.shape
            self._q_captured[l] = output.view(bsz, seq_len, self.num_heads, self.head_dim)
        return hook

    def _make_attn_hook(self, l):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                attn_out = output[0]; rest = output[1:]
            else:
                attn_out = output; rest = None

            q = self._q_captured.get(l)
            pe = self._pos_emb.get(l)
            if q is None or pe is None:
                return output

            cos, sin = pe
            q_bhtd = q.transpose(1, 2)
            cos_d = cos.to(device=q.device, dtype=q.dtype)
            sin_d = sin.to(device=q.device, dtype=q.dtype)
            q_rope, _ = apply_rotary_pos_emb(q_bhtd, q_bhtd, cos_d, sin_d)
            q_rope = q_rope.transpose(1, 2)

            k_ph = self._k[l].to(device=q_rope.device, dtype=q_rope.dtype)
            v_vec = self._v[l].to(device=q_rope.device, dtype=q_rope.dtype)

            dot_per_head = (q_rope * k_ph).sum(dim=-1)
            demand = dot_per_head.mean(dim=-1) / (self.head_dim ** 0.5)
            centered = demand - self._avg_d[l]
            coef = 2.0 * self.base_coef * torch.sigmoid(self._sign * self.scale * centered)

            steered = attn_out + coef.unsqueeze(-1) * v_vec

            if self.debug:
                with torch.no_grad():
                    print(f"[gated L{l}] demand: mean={demand.mean():.3f} "
                          f"std={demand.std():.3f} (avg_d={self._avg_d[l]:.3f}) | "
                          f"coef: mean={coef.mean():.3f} range=[{coef.min():.3f}, {coef.max():.3f}]")

            self._q_captured.pop(l, None)
            self._pos_emb.pop(l, None)

            if rest is not None:
                return (steered,) + tuple(rest)
            return steered
        return hook

    # ── Context manager ──
    def __enter__(self):
        self._q_captured.clear()
        self._pos_emb.clear()
        for l in self.layer_indices:
            attn = self.model.model.layers[l].self_attn
            self._handles.append(attn.register_forward_pre_hook(self._make_pre_hook(l), with_kwargs=True))
            self._handles.append(attn.q_proj.register_forward_hook(self._make_q_hook(l)))
            self._handles.append(attn.register_forward_hook(self._make_attn_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._q_captured.clear()
        self._pos_emb.clear()

    def remove(self):
        self.__exit__(None, None, None)
