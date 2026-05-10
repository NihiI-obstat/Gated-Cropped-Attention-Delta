# activation_steering.py  – v0.2
import torch
from contextlib import contextmanager
from typing import Sequence, Union, Iterable


class ActivationSteerer:
    """
    Add (coeff * steering_vector) to a chosen transformer block's output.
    Now handles blocks that return tuples and fails loudly if it can't
    locate a layer list.
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",       # GPT‑2/Neo, Bloom, etc.
        "encoder.layer",       # BERT/RoBERTa
        "model.layers",        # Llama/Mistral
        "gpt_neox.layers",     # GPT‑NeoX
        "block",               # Flan‑T5
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        target_module: str = "layer",
        debug: bool = False,
        decay_rate: float = None,
    ):
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.target_module = target_module
        self.debug = debug
        self.decay_rate = float(decay_rate) if decay_rate is not None else None
        self._handle = None
        self._response_step_count = 0  # for decay: incremented each decode step in response mode

        # --- build vector ---
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1‑D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )
        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

    # ---------- helpers ----------
    def _locate_layer(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:  # found a full match
                if not hasattr(cur, "__getitem__"):
                    continue  # not a list/ModuleList
                if not (-len(cur) <= self.layer_idx < len(cur)):
                    raise IndexError("layer_idx out of range")
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{self.layer_idx}]")
                layer = cur[self.layer_idx]
                if self.target_module == "self_attn":
                    if hasattr(layer, "self_attn"):
                        return layer.self_attn
                    elif hasattr(layer, "attention"):
                        return layer.attention
                    else:
                        raise ValueError(f"Could not find self_attn in layer {self.layer_idx}")
                elif self.target_module == "mlp":
                    if hasattr(layer, "mlp"):
                        return layer.mlp
                    elif hasattr(layer, "feed_forward"):
                        return layer.feed_forward
                    else:
                        raise ValueError(f"Could not find mlp in layer {self.layer_idx}")
                return layer

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _hook_fn(self, module, ins, out):
        # Effective coefficient: decay per response-token step when decay_rate is set
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        if self.positions == "response" and self.decay_rate is not None and torch.is_tensor(out_t) and out_t.shape[1] == 1:
            eff_coeff = self.coeff * (self.decay_rate ** self._response_step_count)
            self._response_step_count += 1
        else:
            eff_coeff = self.coeff
        steer = eff_coeff * self.vector  # (hidden,)

        def _add(t):
            if self.positions == "all":
                return t + steer.to(t.device)
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                else:
                    t2 = t.clone()
                    t2 += steer.to(t.device)
                    return t2
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        # out may be tensor or tuple/list => normalise to tuple
        if torch.is_tensor(out):
            new_out = _add(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                # unusual case – don't touch
                return out
            head = _add(out[0])
            new_out = (head, *out[1:])  # keep other entries
        else:
            return out  # unknown type – leave unchanged

        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out

    def reset_decay_counter(self) -> None:
        """Reset the response-token counter for decay. Call before each new sequence/batch."""
        self._response_step_count = 0

    # ---------- context manager ----------
    def __enter__(self):
        self._response_step_count = 0
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        self.remove()  # always clean up

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


class ActivationSteererMultiple:
    """
    Add multiple (coeff * steering_vector) to chosen transformer block outputs.
    Accepts a list of dicts, each with keys: steering_vector, coeff, layer_idx, positions.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._handles = []
        self._steerers = []

        # Validate and create individual steerers
        for inst in self.instructions:
            steerer = ActivationSteerer(
                model,
                inst["steering_vector"],
                coeff=inst.get("coeff", 0.0),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                target_module=inst.get("target_module", "layer"),
                debug=debug,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        for steerer in self._steerers:
            layer = steerer._locate_layer()
            handle = layer.register_forward_hook(steerer._hook_fn)
            steerer._handle = handle
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for steerer in self._steerers:
            steerer.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Factory helper – single or multi-vector steering
# ---------------------------------------------------------------------------

def make_steerer(
    model: torch.nn.Module,
    vector,
    *,
    coeff: float = 1.0,
    layer_idx: int = -1,
    positions: str = "all",
    target_module: str = "layer",
    debug: bool = False,
):
    """Create the appropriate steerer context manager.

    Parameters
    ----------
    vector : Tensor **or** list[Tensor]
        A single 1-D steering vector, or a list of vectors to apply
        simultaneously (each added independently to the same layer).

    Returns
    -------
    ActivationSteerer or ActivationSteererMultiple
        Both support the context-manager protocol (``with make_steerer(…):``).
    """
    if isinstance(vector, (list, tuple)):
        # If vector is a list of tensors, we apply them all to the same layer
        instructions = [
            {
                "steering_vector": v,
                "coeff": coeff,
                "layer_idx": layer_idx,
                "positions": positions,
                "target_module": target_module,
            }
            for v in vector
        ]
        return ActivationSteererMultiple(model, instructions, debug=debug)
    elif isinstance(vector, dict) and "instructions" in vector:
        # If vector is a dict with instructions, we just pass them
        return ActivationSteererMultiple(model, vector["instructions"], debug=debug)
    
    return ActivationSteerer(
        model, vector, coeff=coeff, layer_idx=layer_idx,
        positions=positions, target_module=target_module, debug=debug,
    )


# ---------------------------------------------------------------------------
# KV-cache utilities (for clean-KV steering)
# ---------------------------------------------------------------------------

def truncate_kv_cache_inplace(cache, end_pos: int) -> None:
    """Truncate a ``DynamicCache`` **in-place** to positions ``[0, end_pos)``.

    This is used by the clean-KV steering strategy: after a *steered* forward
    pass appends a dirty entry to the cache, we truncate it back to the
    previous length and then run a *clean* forward pass that appends the
    correct (un-steered) entry.

    The function updates both the per-layer K/V tensors and the internal
    ``_seen_tokens`` counter so that subsequent forward passes compute the
    correct positional embeddings.
    """
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :end_pos, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :end_pos, :]
    # Keep the internal token counter in sync (used for position IDs).
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = end_pos