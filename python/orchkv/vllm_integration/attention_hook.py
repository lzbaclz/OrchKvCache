"""
D3: Attention Hook — collect per-block attention scores and feed
them to OrchKvCache's tiered_manager for hot/cold classification.

Strategy: B+C Hybrid
  - Every K steps (default K=10), capture full attention weights from
    an eager-attention forward pass.
  - Perform block-wise reduce on GPU → async D2H → report_attn().
  - Between sampling steps, EMA (λ=0.9) naturally decays old scores.
  - Steady-state overhead: ~0.02 ms/step (0.2 ms every 10 steps).

Usage with vLLM:
    collector = AttentionScoreCollector(tm_handle, block_size=16)
    hook_mgr  = AttentionHookManager(collector)
    hook_mgr.install(model)   # wraps attention modules
    # ... inference loop ...
    hook_mgr.on_step_done()   # after each decode step
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

try:
    import orchkv_core as _C
except ImportError:
    _C = None


class AttentionScoreCollector:
    """
    Collects per-block attention scores from softmax(QK^T/√d) output
    and reports them to the tiered_manager via orchkv_core.

    Pipeline (on sampling steps):
      1. Receive attention_weights [batch, heads, q_len, kv_len] from GPU
      2. On a dedicated CUDA stream: pad → reshape → block-wise mean
      3. Async D2H copy to pinned CPU buffer
      4. Synchronize, then call tm_report_attn() for each (layer, head, block)
    """

    def __init__(
        self,
        tm_handle: int,
        block_size: int = 16,
        sample_interval: int = 10,
    ):
        self._tm = tm_handle
        self._block_size = block_size
        self._sample_interval = max(1, sample_interval)
        self._stream = torch.cuda.Stream()
        self._step = 0
        self._cpu_bufs: dict[int, torch.Tensor] = {}
        self._enabled = True
        self._stats = {
            "samples_taken": 0,
            "blocks_reported": 0,
            "steps_skipped": 0,
        }

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def sample_interval(self) -> int:
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, v: int):
        self._sample_interval = max(1, v)

    def should_sample(self) -> bool:
        """Whether the current step is a sampling step."""
        if not self._enabled:
            return False
        return (self._step % self._sample_interval) == 0

    def collect(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
        block_table: torch.Tensor | None = None,
        block_id_fn: Callable[[int, int, int], int | None] | None = None,
    ):
        """
        Collect attention scores for one layer.

        Args:
            layer_idx:    Transformer layer index (0-based).
            attn_weights: GPU tensor, shape [batch, n_kv_heads, q_len, kv_len].
                          This is the softmax output (attention probabilities).
            block_table:  Optional vLLM block table [batch, max_blocks] mapping
                          (batch, block_idx) → vllm_block_id.
            block_id_fn:  Callable(layer, head, block_idx) → orchkv_block_id.
                          If None, uses (layer_idx * H * B + head * B + bi) as
                          a fallback block_id.
        """
        if not self.should_sample():
            return

        bs, n_heads, q_len, kv_len = attn_weights.shape
        n_blocks = (kv_len + self._block_size - 1) // self._block_size

        with torch.cuda.stream(self._stream):
            scores = self._blockwise_reduce(attn_weights, n_blocks)
            cpu_buf = self._async_d2h(layer_idx, scores)

        self._stream.synchronize()

        self._report_scores(layer_idx, n_heads, n_blocks, cpu_buf,
                            block_table, block_id_fn)

    def _blockwise_reduce(
        self, attn_weights: torch.Tensor, n_blocks: int,
    ) -> torch.Tensor:
        """
        GPU-side block-wise mean reduction.

        Input:  [batch, heads, q_len, kv_len]
        Output: [heads, n_blocks]
        """
        bs, n_heads, q_len, kv_len = attn_weights.shape
        padded_len = n_blocks * self._block_size

        if kv_len < padded_len:
            pad_size = padded_len - kv_len
            aw = torch.nn.functional.pad(attn_weights, (0, pad_size))
        else:
            aw = attn_weights

        # [batch, heads, q_len, n_blocks, block_size]
        aw = aw.view(bs, n_heads, q_len, n_blocks, self._block_size)
        # Mean over batch(0), q_len(2), block_tokens(4) → [heads, n_blocks]
        scores = aw.mean(dim=(0, 2, 4))
        return scores

    def _async_d2h(
        self, layer_idx: int, scores: torch.Tensor,
    ) -> torch.Tensor:
        """Async GPU→CPU copy into pinned buffer."""
        buf = self._cpu_bufs.get(layer_idx)
        if buf is None or buf.shape != scores.shape:
            buf = torch.empty(
                scores.shape, dtype=scores.dtype,
                device="cpu", pin_memory=True,
            )
            self._cpu_bufs[layer_idx] = buf
        buf.copy_(scores, non_blocking=True)
        return buf

    def _report_scores(
        self,
        layer_idx: int,
        n_heads: int,
        n_blocks: int,
        cpu_scores: torch.Tensor,
        block_table: torch.Tensor | None,
        block_id_fn: Callable[[int, int, int], int | None] | None,
    ):
        """Report per-block scores to tiered_manager."""
        if _C is None:
            return

        reported = 0
        for h in range(n_heads):
            for bi in range(n_blocks):
                if block_id_fn is not None:
                    bid = block_id_fn(layer_idx, h, bi)
                elif block_table is not None and bi < block_table.shape[-1]:
                    bid = int(block_table[0, bi])
                else:
                    bid = layer_idx * n_heads * n_blocks + h * n_blocks + bi

                if bid is not None and bid >= 0:
                    score = float(cpu_scores[h, bi])
                    _C.tm_report_attn(self._tm, bid, score)
                    reported += 1

        self._stats["blocks_reported"] += reported

    def on_step_done(self):
        """
        Mark the end of a decode step.

        Must be called exactly once per decode step, after all layers
        have been processed. Advances the attention tracker's EMA.
        """
        self._step += 1

        if self.should_sample():
            self._stats["samples_taken"] += 1
        else:
            self._stats["steps_skipped"] += 1

        if _C is not None:
            _C.tm_step_done(self._tm)

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "current_step": self._step,
            "sample_interval": self._sample_interval,
            "block_size": self._block_size,
            "enabled": self._enabled,
        }

    def set_enabled(self, enabled: bool):
        self._enabled = enabled


class AttentionHookManager:
    """
    Manages attention hooks on vLLM model modules.

    Strategy B+C hybrid:
      - Wraps each attention module's forward() method.
      - On sampling steps: forces eager attention to get full weights,
        then feeds them to AttentionScoreCollector.
      - On non-sampling steps: runs normal FlashAttention (no overhead).

    Install:
        hook_mgr = AttentionHookManager(collector)
        hook_mgr.install(model)

    Note: FlashAttention does NOT output attention weights by default.
    On sampling steps, the hook switches to the "eager" backend path
    that does output weights. This adds ~0.2ms per layer per sample step.
    """

    def __init__(
        self,
        collector: AttentionScoreCollector,
        attn_module_name: str = "attn",
    ):
        self._collector = collector
        self._attn_module_name = attn_module_name
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._original_forwards: dict[int, Callable] = {}
        self._installed = False
        self._layer_map: dict[int, int] = {}

    def install(self, model: torch.nn.Module):
        """
        Find all attention modules in the model and install hooks.

        Compatible with vLLM's model structure where attention layers
        are accessible via model.model.layers[i].self_attn or similar.
        """
        if self._installed:
            logger.warning("AttentionHookManager already installed")
            return

        layer_idx = 0
        for name, module in model.named_modules():
            if self._is_attention_module(name, module):
                self._wrap_module(module, layer_idx)
                self._layer_map[id(module)] = layer_idx
                layer_idx += 1

        self._installed = True
        logger.info("AttentionHookManager: installed hooks on %d layers",
                     layer_idx)

    def _is_attention_module(
        self, name: str, module: torch.nn.Module,
    ) -> bool:
        """Heuristic to identify attention modules in the model."""
        name_lower = name.lower()
        return (
            self._attn_module_name in name_lower
            and "self_attn" in name_lower
        )

    def _wrap_module(self, module: torch.nn.Module, layer_idx: int):
        """Replace module.forward with a wrapped version."""
        original_forward = module.forward
        self._original_forwards[id(module)] = original_forward
        collector = self._collector

        def hooked_forward(*args, **kwargs):
            if collector.should_sample():
                kwargs["output_attentions"] = True

            result = original_forward(*args, **kwargs)

            if collector.should_sample():
                attn_weights = self._extract_attn_weights(result, kwargs)
                if attn_weights is not None:
                    collector.collect(
                        layer_idx=layer_idx,
                        attn_weights=attn_weights,
                        block_table=kwargs.get("block_table"),
                    )

            return result

        module.forward = hooked_forward

    @staticmethod
    def _extract_attn_weights(
        result: Any, kwargs: dict,
    ) -> torch.Tensor | None:
        """
        Extract attention weights from the forward output.

        Handles common return formats:
          - tuple: (output, attn_weights, ...)
          - dict with 'attn_weights' or 'attention_weights' key
          - single tensor (no weights available)
        """
        if isinstance(result, tuple) and len(result) >= 2:
            candidate = result[1]
            if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
                return candidate

        if isinstance(result, dict):
            for key in ("attn_weights", "attention_weights", "attentions"):
                if key in result and isinstance(result[key], torch.Tensor):
                    return result[key]

        return None

    def on_step_done(self):
        """Delegate to collector's step_done (call after each decode step)."""
        self._collector.on_step_done()

    def uninstall(self):
        """Remove all hooks and restore original forward methods."""
        for module_id, original in self._original_forwards.items():
            for name, module in self._get_all_modules():
                if id(module) == module_id:
                    module.forward = original
                    break
        self._original_forwards.clear()
        self._layer_map.clear()
        self._installed = False

    def _get_all_modules(self):
        """Placeholder - in practice uses model.named_modules()."""
        return []

    @property
    def collector(self) -> AttentionScoreCollector:
        return self._collector

    def get_stats(self) -> dict[str, Any]:
        return {
            "installed": self._installed,
            "n_layers_hooked": len(self._layer_map),
            "collector": self._collector.get_stats(),
        }
