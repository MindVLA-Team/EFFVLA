# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# LLM-Initialized Regression (OFT-style) Action Head with KV Cache Sharing
#
# Key property: the transformer blocks are deep-copied from the VLM's last decoder layers
#   1. Action head transformer layers are DEEPCOPIED from the LLM's own DecoderLayers
#      → all weights are strictly loaded from the pretrained LLM, no random init
#   2. No randomly initialized Transformer blocks – LLM weights used directly
#   3. Hidden size / num_heads / head_dim are AUTO-DETECTED from the LLM config
#      → no need to set dit_num_heads etc. in YAML (they are ignored for llm_init)
#   4. Config still controls: dit_num_layers, vlm_kv_layer_offset
#      → same VLM KV layer selection logic as before
#   5. L1 regression loss (identical to KVSharedRegressionActionHead)

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, num_kv_heads, seqlen, head_dim) → (batch, num_heads, seqlen, head_dim)"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─────────────────────────────────────────────────────────────────────────────
# RoPE for action token positions
# ─────────────────────────────────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """Independent RoPE for action positions (offset from VLM positions)."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Args:
            x:            (B, seq_len, _) – used only for device/dtype
            position_ids: (B, seq_len)
        Returns:
            cos, sin each of shape (B, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


# ─────────────────────────────────────────────────────────────────────────────
# Small MLP (state encoder / action decoder)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, output_dim: int = 2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: LLM architecture introspection
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm_decoder_layers(llm_model):
    """Return the nn.ModuleList of transformer decoder layers."""
    if hasattr(llm_model, "model") and hasattr(llm_model.model, "layers"):
        return llm_model.model.layers
    if hasattr(llm_model, "transformer") and hasattr(llm_model.transformer, "h"):
        return llm_model.transformer.h
    if hasattr(llm_model, "layers"):
        return llm_model.layers
    raise ValueError(
        f"Cannot locate decoder layers in {type(llm_model).__name__}. "
        "Expected llm_model.model.layers, llm_model.transformer.h, or llm_model.layers."
    )


def _get_llm_attn_config(llm_model):
    """Extract num_heads, num_kv_heads, head_dim, rope_theta from LLM config."""
    cfg = llm_model.config
    hidden_size = cfg.hidden_size
    num_heads   = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim    = getattr(cfg, "head_dim", hidden_size // num_heads)
    rope_theta  = float(getattr(cfg, "rope_theta", 10000.0))
    return hidden_size, num_heads, num_kv_heads, head_dim, rope_theta


# ─────────────────────────────────────────────────────────────────────────────
# Core: LLM decoder layer wrapper with VLM KV cache sharing
# (shared with FM version – same deepcopy logic, same attention override)
# ─────────────────────────────────────────────────────────────────────────────

class LLMDecoderLayerWithKVShare(nn.Module):
    """
    Wraps a **deep copy** of an LLM decoder layer and replaces the attention
    forward with one that supports VLM KV cache sharing.

    All trainable weights come from the LLM (q/k/v/o projections, MLP, norms).
    Supports GQA and optional per-head q/k norms (Qwen2/3 style).
    """

    def __init__(
        self,
        llm_decoder_layer: nn.Module,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        # ── Deep copy: all LLM weights are strictly preserved ──────────────
        self.decoder_layer = copy.deepcopy(llm_decoder_layer)

        self.hidden_size  = hidden_size
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, T, H]
        vlm_past_key_values: tuple,             # (k_cache, v_cache)
        position_embeddings: tuple,             # (cos, sin)
        attention_mask: torch.Tensor = None,    # [B, 1, T, vlm_len+T] float mask
    ) -> torch.Tensor:
        self_attn = self.decoder_layer.self_attn
        B, T, _  = hidden_states.shape

        # ── 1. Pre-norm ──────────────────────────────────────────────────────
        residual = hidden_states
        normed   = self.decoder_layer.input_layernorm(hidden_states)

        # ── 2. QKV projections ───────────────────────────────────────────────
        q = self_attn.q_proj(normed)
        k = self_attn.k_proj(normed)
        v = self_attn.v_proj(normed)

        q = q.view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── 3. Per-head norms (Qwen2/3 style, optional) ──────────────────────
        if hasattr(self_attn, "q_norm"):
            q = self_attn.q_norm(q)
        if hasattr(self_attn, "k_norm"):
            k = self_attn.k_norm(k)

        # ── 4. RoPE ──────────────────────────────────────────────────────────
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── 5. GQA expand ────────────────────────────────────────────────────
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # ── 6. Concat VLM KV cache ────────────────────────────────────────────
        vlm_k, vlm_v = vlm_past_key_values
        vlm_kv_groups = self.num_heads // vlm_k.shape[1]
        vlm_k = repeat_kv(vlm_k, vlm_kv_groups)
        vlm_v = repeat_kv(vlm_v, vlm_kv_groups)

        k_full = torch.cat([vlm_k, k], dim=2)
        v_full = torch.cat([vlm_v, v], dim=2)

        # ── 7. Scaled dot-product attention ──────────────────────────────────
        attn_out = F.scaled_dot_product_attention(
            query=q, key=k_full, value=v_full,
            attn_mask=attention_mask, dropout_p=0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            B, T, self.num_heads * self.head_dim
        )
        attn_out = self_attn.o_proj(attn_out)

        hidden_states = residual + attn_out

        # ── 8. MLP (uses whatever activation the LLM has: SiLU gate, GELU…) ─
        residual      = hidden_states
        normed        = self.decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.decoder_layer.mlp(normed)

        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Transformer stack with LLM-initialized layers
# ─────────────────────────────────────────────────────────────────────────────

class KVSharedLLMTransformer(nn.Module):
    """
    Drop-in replacement for KVSharedTransformer that uses LLM-initialized layers.

    Selects `dit_num_layers` consecutive layers from the LLM's decoder stack
    (the same layers whose KV cache is shared), deep-copies them, and uses
    them as the action head transformer body.
    """

    def __init__(
        self,
        llm_model: nn.Module,
        dit_num_layers: int,
        vlm_kv_layer_offset=None,
        reinit: bool = False,
    ):
        super().__init__()

        hidden_size, num_heads, num_kv_heads, head_dim, rope_theta = _get_llm_attn_config(llm_model)

        self.hidden_size = hidden_size
        self.num_layers  = dit_num_layers

        # ── Select which LLM layers to deepcopy ──────────────────────────────
        llm_layers     = _get_llm_decoder_layers(llm_model)
        num_llm_layers = len(llm_layers)

        if vlm_kv_layer_offset is not None:
            start_idx = num_llm_layers - vlm_kv_layer_offset - dit_num_layers
            if start_idx < 0:
                raise ValueError(
                    f"vlm_kv_layer_offset ({vlm_kv_layer_offset}) + dit_num_layers "
                    f"({dit_num_layers}) > total LLM layers ({num_llm_layers})"
                )
        else:
            start_idx = num_llm_layers - dit_num_layers
            if start_idx < 0:
                raise ValueError(
                    f"dit_num_layers ({dit_num_layers}) > total LLM layers ({num_llm_layers})"
                )

        print(
            f"[LLMInitOFT] Deepcopying LLM decoder layers [{start_idx}, "
            f"{start_idx + dit_num_layers}) of {num_llm_layers} total layers. "
            f"H={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

        # ── Build action head layers from LLM ────────────────────────────────
        self.layers = nn.ModuleList([
            LLMDecoderLayerWithKVShare(
                llm_decoder_layer=llm_layers[start_idx + i],
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            for i in range(dit_num_layers)
        ])

        # ── Optional: reinitialize the deepcopied weights (ablation) ─────────
        # When reinit=True, the LLM DecoderLayer structure (q/k/v/o projections,
        # GQA, q_norm/k_norm, SiLU-gated MLP, RMSNorm) is preserved exactly, but
        # all weights are reset to the same distribution a freshly-initialized
        # LLM would use. This isolates "architecture" from "weight inheritance".
        if reinit:
            self._reinit_weights(llm_model)
            print(
                f"[LLMInitOFT] Reinitialized {dit_num_layers} deepcopied decoder "
                f"layers with random weights (LLM structure kept, weights reset)."
            )

        # ── RoPE for action positions ─────────────────────────────────────────
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=head_dim,
            max_position_embeddings=2048,
            base=rope_theta,
        )

        # ── Store layer indices for KV extraction ─────────────────────────────
        self.vlm_kv_start_idx = start_idx
        self.dit_num_layers   = dit_num_layers

    def _reinit_weights(self, llm_model: nn.Module):
        """Reset the deepcopied decoder-layer weights to LLM-fresh init.

        Prefers the LLM's own `_init_weights` (HF standard, applies
        `normal_(0, initializer_range)` to Linear/Embedding, `ones_` to RMSNorm
        weight, `zeros_` to biases). Falls back to manual reinit for LLMs
        without `_init_weights` (e.g. MindVLM).
        """
        init_range = float(getattr(llm_model.config, "initializer_range", 0.02))
        has_init_weights = callable(getattr(llm_model, "_init_weights", None))

        for layer in self.layers:
            decoder = layer.decoder_layer
            if has_init_weights:
                for _, sub in decoder.named_modules():
                    llm_model._init_weights(sub)
            else:
                for sub in decoder.modules():
                    if isinstance(sub, (nn.Linear, nn.Embedding)):
                        nn.init.normal_(sub.weight, mean=0.0, std=init_range)
                        if getattr(sub, "bias", None) is not None:
                            nn.init.zeros_(sub.bias)
                    elif isinstance(sub, (nn.LayerNorm, nn.RMSNorm)) or (
                        hasattr(sub, "weight") and sub.__class__.__name__.endswith("Norm")
                    ):
                        if getattr(sub, "weight", None) is not None:
                            nn.init.ones_(sub.weight)
                        if getattr(sub, "bias", None) is not None:
                            nn.init.zeros_(sub.bias)


    def forward(
        self,
        hidden_states: torch.Tensor,         # [B, T, H]
        vlm_past_key_values: list,            # Full VLM past_key_values
        dit_position_ids: torch.Tensor,       # [B, T]
        attention_mask: torch.Tensor = None,  # [B, 1, T, vlm_len+T]
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(hidden_states, dit_position_ids)
        selected_vlm_kv     = self._extract_kv(vlm_past_key_values)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                vlm_past_key_values=selected_vlm_kv[layer_idx],
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        return hidden_states

    def _extract_kv(self, vlm_past_key_values) -> list:
        selected = []
        for i in range(self.vlm_kv_start_idx, self.vlm_kv_start_idx + self.dit_num_layers):
            layer_cache = vlm_past_key_values[i]
            if isinstance(layer_cache, tuple):
                k, v = layer_cache
            elif hasattr(layer_cache, "key_cache") and hasattr(layer_cache, "value_cache"):
                k, v = layer_cache.key_cache, layer_cache.value_cache
            elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                k, v = layer_cache.keys, layer_cache.values
            elif hasattr(layer_cache, "key") and hasattr(layer_cache, "value"):
                k, v = layer_cache.key, layer_cache.value
            else:
                raise TypeError(
                    f"Unknown KV cache type: {type(layer_cache)}\n"
                    f"Attributes: {[a for a in dir(layer_cache) if not a.startswith('_')]}"
                )
            selected.append((k, v))
        return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main action head: LLM-initialized regression (OFT)
# ─────────────────────────────────────────────────────────────────────────────

class LLMInitRegressionActionHead(nn.Module):
    """
    L1 regression action head whose transformer body is initialized from the
    LLM's own DecoderLayers (via deep copy).

    Replaces KVSharedRegressionActionHead with:
      - LLM-initialized transformer (no random init)
      - Same VLM KV cache sharing
      - Same learnable action queries, state encoder, future tokens, pos embed
      - Same L1 regression loss and single-pass inference
    """

    def __init__(self, global_config, llm_model: nn.Module, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model

        self.hidden_size    = global_config.framework.qwenvl.vl_hidden_dim
        self.action_dim     = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1

        # ── LLM-initialized transformer ───────────────────────────────────────
        # action_head_type: "llm_init"      → deepcopy LLM layers, keep weights
        #                   "llm_init_random" → deepcopy LLM layers, reinit weights
        #                       (LLM architecture preserved, weights reset to fresh LLM init)
        action_head_type = getattr(action_config, "action_head_type", "llm_init")
        reinit = (action_head_type == "llm_init_random")
        self.transformer = KVSharedLLMTransformer(
            llm_model=llm_model,
            dit_num_layers=action_config.dit_num_layers,
            vlm_kv_layer_offset=action_config.vlm_kv_layer_offset,
            reinit=reinit,
        )

        # ── State encoder (optional) ─────────────────────────────────────────
        self.state_encoder = (
            MLP(input_dim=action_config.state_dim, output_dim=self.hidden_size)
            if action_config.state_dim
            else None
        )

        # ── Learnable action query tokens ─────────────────────────────────────
        self.action_queries = nn.Embedding(self.action_horizon, self.hidden_size)
        nn.init.normal_(self.action_queries.weight, mean=0.0, std=0.02)

        # ── Future context tokens (optional) ─────────────────────────────────
        if action_config.num_target_vision_tokens > 0:
            self.future_tokens = nn.Embedding(
                action_config.num_target_vision_tokens, self.hidden_size
            )
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        else:
            self.future_tokens = None

        # ── Positional embedding ──────────────────────────────────────────────
        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(2048, self.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ── Action decoder ────────────────────────────────────────────────────
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )

        # ── Multi-pass iterative refinement (ablation vs flow matching) ──────
        # num_repeat_passes=1 (default): single pass, identical to original.
        #   → existing checkpoints load with strict=True, zero architecture change.
        # num_repeat_passes>1: the output hidden states of the action tokens from
        #   pass k are fed back directly as the action token inputs of pass k+1
        #   (prefix tokens are re-prepended each pass from their embeddings).
        #   No new parameters are introduced – old checkpoints remain fully
        #   compatible even with num_repeat_passes>1.
        #   L1 loss is computed on the FINAL pass output only.
        self.num_repeat_passes = int(getattr(action_config, "num_repeat_passes", 1))

        self.action_config   = action_config
        self.dit_num_layers  = action_config.dit_num_layers
        self.vlm_kv_layer_offset = action_config.vlm_kv_layer_offset

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_attention_mask(
        self,
        vlm_attention_mask: torch.Tensor,
        dit_len: int,
        device,
        vlm_kv_len: int = None,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        """
        [B, 1, dit_len, vlm_kv_len + dit_len]  (float, 0 / -inf)
        Rules identical to KVSharedRegressionActionHead.
        """
        batch_size = vlm_attention_mask.shape[0]
        if vlm_kv_len is None:
            vlm_kv_len = vlm_attention_mask.shape[1]

        action_len = dit_len - prefix_len

        vlm_mask = torch.ones(
            batch_size, dit_len, vlm_kv_len, device=device, dtype=torch.bool
        )

        if prefix_len > 0:
            prefix_causal = torch.tril(
                torch.ones(batch_size, prefix_len, prefix_len, device=device, dtype=torch.bool)
            )
            prefix_to_action = torch.zeros(
                batch_size, prefix_len, action_len, device=device, dtype=torch.bool
            )
            top = torch.cat([prefix_causal, prefix_to_action], dim=-1)

        action_to_all = torch.ones(
            batch_size, action_len, dit_len, device=device, dtype=torch.bool
        )

        dit_mask = torch.cat([top, action_to_all], dim=1) if prefix_len > 0 else action_to_all
        full_mask = torch.cat([vlm_mask, dit_mask], dim=-1).unsqueeze(1)

        float_mask = torch.zeros_like(full_mask, dtype=torch.float32)
        float_mask = float_mask.masked_fill(~full_mask, float("-inf"))
        return float_mask

    def _prepare_transformer_input(
        self,
        batch_size: int,
        device,
        state: torch.Tensor = None,        # [B, 1, state_dim] or None
        action_input: torch.Tensor = None, # [B, T, H] or None
    ) -> torch.Tensor:
        """Build input: [state_embed] + [future_tokens] + [action_tokens].

        Args:
            action_input: If None, use learnable action_queries (pass 1).
                          If provided [B, T, H], use directly as action tokens
                          (pass 2+: hidden states from previous transformer output).
        """
        if action_input is None:
            ids      = torch.arange(self.action_horizon, dtype=torch.long, device=device)
            action_tokens = self.action_queries(ids).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            action_tokens = action_input  # [B, T, H]

        state_embed = self.state_encoder(state) if state is not None else None

        ft = (
            self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            if self.future_tokens is not None
            else None
        )

        parts = []
        if state_embed is not None:
            parts.append(state_embed)
        if ft is not None:
            parts.append(ft)
        parts.append(action_tokens)

        return torch.cat(parts, dim=1)

    # ── forward (training) ───────────────────────────────────────────────────

    def forward(
        self,
        vlm_past_key_values: list,
        vlm_attention_mask: torch.Tensor,   # [B, vlm_len]
        actions: torch.Tensor,              # [B, T, action_dim]
        state: torch.Tensor = None,         # [B, 1, state_dim] or None
    ) -> torch.Tensor:
        """
        Compute L1 regression loss with optional multi-pass refinement.

        When num_repeat_passes=1 (default), behaviour is identical to the
        original single-pass implementation (backward compatible).
        When num_repeat_passes>1, the action-token hidden states from pass k
        are fed back as action token inputs of pass k+1; no new parameters
        are introduced.

        Returns:
            loss (torch.Tensor): scalar L1 loss on the final pass output.
        """
        device     = actions.device
        batch_size = actions.shape[0]

        # ── Pre-compute components shared across all passes ───────────────────
        dit_len = (
            (1 if state is not None else 0)
            + (self.action_config.num_target_vision_tokens if self.future_tokens is not None else 0)
            + self.action_horizon
        )

        # Extract selected VLM KV once (for vlm_kv_len only; transformer re-extracts internally)
        selected_kv = self.transformer._extract_kv(vlm_past_key_values)
        vlm_kv_len  = selected_kv[0][0].shape[2]

        prefix_len = (1 if state is not None else 0) + (
            self.action_config.num_target_vision_tokens if self.future_tokens is not None else 0
        )
        attention_mask = self._build_attention_mask(
            vlm_attention_mask, dit_len, device,
            vlm_kv_len=vlm_kv_len, prefix_len=prefix_len,
        )

        vlm_max_pos      = vlm_attention_mask.sum(dim=1).max().item()
        dit_position_ids = (
            torch.arange(dit_len, device=device, dtype=torch.long)
            .unsqueeze(0).expand(batch_size, -1)
            + vlm_max_pos + 1
        )

        if self.action_config.add_pos_embed:
            pos_ids  = torch.arange(dit_len, device=device, dtype=torch.long)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)  # [1, dit_len, H]
        else:
            pos_embs = None

        # ── Pass 1: learnable action queries ─────────────────────────────────
        hidden_states = self._prepare_transformer_input(batch_size, device, state,
                                                        action_input=None)
        if pos_embs is not None:
            hidden_states = hidden_states + pos_embs

        hidden_states = self.transformer(
            hidden_states=hidden_states,
            vlm_past_key_values=vlm_past_key_values,
            dit_position_ids=dit_position_ids,
            attention_mask=attention_mask,
        )
        pred_actions = self.action_decoder(hidden_states[:, -self.action_horizon:])

        # ── Passes 2..N: feed back action-portion hidden states ───────────────
        for _ in range(1, self.num_repeat_passes):
            # No detach: gradients flow back through all passes (end-to-end).
            # Stable for small num_repeat_passes due to residual connections.
            action_input = hidden_states[:, -self.action_horizon:]  # [B, T, H]
            hidden_states = self._prepare_transformer_input(batch_size, device, state,
                                                            action_input=action_input)
            if pos_embs is not None:
                hidden_states = hidden_states + pos_embs

            hidden_states = self.transformer(
                hidden_states=hidden_states,
                vlm_past_key_values=vlm_past_key_values,
                dit_position_ids=dit_position_ids,
                attention_mask=attention_mask,
            )
            pred_actions = self.action_decoder(hidden_states[:, -self.action_horizon:])

        # L1 loss on final pass output
        loss = F.l1_loss(pred_actions, actions)
        return loss

    # ── predict_action (inference) ───────────────────────────────────────────

    @torch.no_grad()
    def predict_action(
        self,
        vlm_past_key_values: list,
        vlm_attention_mask: torch.Tensor,
        state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Multi-pass inference: each pass refines action hidden states.

        When num_repeat_passes=1 (default), identical to original single-pass.
        """
        device     = vlm_attention_mask.device
        batch_size = vlm_attention_mask.shape[0]

        # ── Pre-compute shared components ─────────────────────────────────────
        dit_len = (
            (1 if state is not None else 0)
            + (self.action_config.num_target_vision_tokens if self.future_tokens is not None else 0)
            + self.action_horizon
        )

        selected_kv = self.transformer._extract_kv(vlm_past_key_values)
        vlm_kv_len  = selected_kv[0][0].shape[2]

        prefix_len = (1 if state is not None else 0) + (
            self.action_config.num_target_vision_tokens if self.future_tokens is not None else 0
        )
        attention_mask = self._build_attention_mask(
            vlm_attention_mask, dit_len, device,
            vlm_kv_len=vlm_kv_len, prefix_len=prefix_len,
        )

        vlm_max_pos      = vlm_attention_mask.sum(dim=1).max().item()
        dit_position_ids = (
            torch.arange(dit_len, device=device, dtype=torch.long)
            .unsqueeze(0).expand(batch_size, -1)
            + vlm_max_pos + 1
        )

        if self.action_config.add_pos_embed:
            pos_ids  = torch.arange(dit_len, device=device, dtype=torch.long)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        else:
            pos_embs = None

        # ── Pass 1 ────────────────────────────────────────────────────────────
        hidden_states = self._prepare_transformer_input(batch_size, device, state,
                                                        action_input=None)
        if pos_embs is not None:
            hidden_states = hidden_states + pos_embs

        hidden_states = self.transformer(
            hidden_states=hidden_states,
            vlm_past_key_values=vlm_past_key_values,
            dit_position_ids=dit_position_ids,
            attention_mask=attention_mask,
        )

        # ── Passes 2..N ───────────────────────────────────────────────────────
        for _ in range(1, self.num_repeat_passes):
            action_input = hidden_states[:, -self.action_horizon:]  # [B, T, H]
            hidden_states = self._prepare_transformer_input(batch_size, device, state,
                                                            action_input=action_input)
            if pos_embs is not None:
                hidden_states = hidden_states + pos_embs

            hidden_states = self.transformer(
                hidden_states=hidden_states,
                vlm_past_key_values=vlm_past_key_values,
                dit_position_ids=dit_position_ids,
                attention_mask=attention_mask,
            )

        pred_actions = self.action_decoder(hidden_states[:, -self.action_horizon:])
        return pred_actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_init_oft_action_model(config, llm_model: nn.Module):
    """
    Factory: build LLM-initialized regression action head.

    Args:
        config:    Global framework config (OmegaConf / dict).
        llm_model: The language model (AutoModelForCausalLM or similar).
                   Its decoder layers will be deep-copied into the action head.
    Returns:
        LLMInitRegressionActionHead
    """
    return LLMInitRegressionActionHead(global_config=config, llm_model=llm_model)


if __name__ == "__main__":
    pass
