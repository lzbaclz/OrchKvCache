"""
D3: Unit tests for AttentionScoreCollector and AttentionHookManager.

Tests the attention hook pipeline WITHOUT requiring vLLM.
Uses orchkv_core's tiered_manager for real C-side integration.

Run with: conda run -n orchkv python -m pytest test/test_attention_hook.py -v
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import orchkv_core as _C
from orchkv.vllm_integration.attention_hook import (
    AttentionScoreCollector,
    AttentionHookManager,
)


@pytest.fixture
def tm_handle():
    """Create a tiered_manager for testing."""
    tm = _C.tm_create(tracker_cap=1024, ema_lambda=0.9)
    yield tm
    _C.tm_destroy(tm)


# ==================== AttentionScoreCollector ====================

class TestCollectorBasic:
    def test_init(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=16, sample_interval=5)
        assert c.block_size == 16
        assert c.sample_interval == 5
        assert c._step == 0

    def test_should_sample_step0(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, sample_interval=10)
        assert c.should_sample()  # step 0 is a sampling step

    def test_should_sample_interval(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, sample_interval=3)
        samples = []
        for i in range(9):
            samples.append(c.should_sample())
            c.on_step_done()
        # step 0,3,6 → sample; 1,2,4,5,7,8 → skip
        assert samples == [True, False, False, True, False, False,
                           True, False, False]

    def test_disabled(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, sample_interval=1)
        assert c.should_sample()
        c.set_enabled(False)
        assert not c.should_sample()
        c.set_enabled(True)
        assert c.should_sample()


class TestCollectorReduce:
    """Test the GPU-side block-wise reduction pipeline."""

    def test_blockwise_reduce_exact(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        # [batch=1, heads=2, q_len=1, kv_len=8] → n_blocks=2
        aw = torch.ones(1, 2, 1, 8, device="cuda") * 0.5
        scores = c._blockwise_reduce(aw, n_blocks=2)
        assert scores.shape == (2, 2)
        assert torch.allclose(scores, torch.tensor(0.5, device="cuda"),
                              atol=1e-5)

    def test_blockwise_reduce_with_padding(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        # kv_len=6 → padded to 8, n_blocks=2
        aw = torch.ones(1, 1, 1, 6, device="cuda")
        scores = c._blockwise_reduce(aw, n_blocks=2)
        assert scores.shape == (1, 2)
        # block 0: mean of [1,1,1,1] = 1.0
        # block 1: mean of [1,1,0,0] = 0.5
        assert abs(float(scores[0, 0]) - 1.0) < 1e-5
        assert abs(float(scores[0, 1]) - 0.5) < 1e-5

    def test_blockwise_reduce_multihead(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        aw = torch.zeros(2, 4, 3, 12, device="cuda")
        # head 0 blocks get 1.0, head 1 gets 0.5, etc
        for h in range(4):
            aw[:, h, :, :] = (h + 1) * 0.1
        scores = c._blockwise_reduce(aw, n_blocks=3)
        assert scores.shape == (4, 3)
        for h in range(4):
            expected = (h + 1) * 0.1
            assert torch.allclose(scores[h], torch.tensor(expected, device="cuda"),
                                  atol=1e-5)

    def test_async_d2h(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        gpu_scores = torch.randn(4, 8, device="cuda")
        cpu_buf = c._async_d2h(layer_idx=0, scores=gpu_scores)
        torch.cuda.synchronize()
        assert cpu_buf.device == torch.device("cpu")
        assert cpu_buf.is_pinned()
        assert torch.allclose(cpu_buf, gpu_scores.cpu(), atol=1e-6)

    def test_d2h_buffer_reuse(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        s1 = torch.randn(2, 4, device="cuda")
        buf1 = c._async_d2h(0, s1)
        s2 = torch.randn(2, 4, device="cuda")
        buf2 = c._async_d2h(0, s2)
        assert buf1.data_ptr() == buf2.data_ptr()  # same pinned buffer reused


class TestCollectorE2E:
    """End-to-end collect + report_attn integration tests."""

    def test_collect_reports_to_tm(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        # [batch=1, heads=2, q_len=1, kv_len=8] → 2 blocks per head
        aw = torch.rand(1, 2, 1, 8, device="cuda")
        block_ids_called = []

        def mock_bid_fn(layer, head, bi):
            bid = layer * 100 + head * 10 + bi
            block_ids_called.append(bid)
            return bid

        c.collect(layer_idx=0, attn_weights=aw, block_id_fn=mock_bid_fn)
        # 2 heads × 2 blocks = 4 calls
        assert len(block_ids_called) == 4
        assert set(block_ids_called) == {0, 1, 10, 11}

    def test_collect_skip_non_sample_step(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=3)
        aw = torch.rand(1, 1, 1, 4, device="cuda")
        calls = []

        def bid_fn(l, h, bi):
            calls.append((l, h, bi))
            return l * 10 + bi

        c.collect(0, aw, block_id_fn=bid_fn)  # step=0 → sample
        assert len(calls) == 1

        c.on_step_done()  # step becomes 1
        calls.clear()
        c.collect(0, aw, block_id_fn=bid_fn)  # step=1 → skip
        assert len(calls) == 0

        c.on_step_done()  # step becomes 2
        calls.clear()
        c.collect(0, aw, block_id_fn=bid_fn)  # step=2 → skip
        assert len(calls) == 0

        c.on_step_done()  # step becomes 3
        calls.clear()
        c.collect(0, aw, block_id_fn=bid_fn)  # step=3 → sample
        assert len(calls) == 1

    def test_collect_with_block_table(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        aw = torch.rand(1, 1, 1, 8, device="cuda")
        block_table = torch.tensor([[100, 200]], dtype=torch.int64)

        c.collect(layer_idx=0, attn_weights=aw, block_table=block_table)
        # Should have reported for block_ids 100 and 200
        stats = c.get_stats()
        assert stats["blocks_reported"] == 2

    def test_step_done_advances_tm(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        aw = torch.ones(1, 1, 1, 4, device="cuda") * 0.9

        c.collect(0, aw, block_id_fn=lambda l, h, bi: bi)
        c.on_step_done()

        stats = _C.tm_get_stats(tm_handle)
        assert isinstance(stats, dict)

    def test_multi_layer_collect(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        all_bids = []

        def bid_fn(l, h, bi):
            bid = l * 100 + h * 10 + bi
            all_bids.append(bid)
            return bid

        for layer in range(4):
            aw = torch.rand(1, 2, 1, 8, device="cuda")
            c.collect(layer, aw, block_id_fn=bid_fn)

        # 4 layers × 2 heads × 2 blocks = 16
        assert len(all_bids) == 16

    def test_stats(self, tm_handle):
        c = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=2)
        aw = torch.rand(1, 1, 1, 4, device="cuda")

        c.collect(0, aw, block_id_fn=lambda l, h, bi: bi)  # step 0 → sample
        c.on_step_done()

        c.collect(0, aw, block_id_fn=lambda l, h, bi: bi)  # step 1 → skip
        c.on_step_done()

        c.collect(0, aw, block_id_fn=lambda l, h, bi: bi)  # step 2 → sample
        c.on_step_done()

        stats = c.get_stats()
        assert stats["current_step"] == 3
        assert stats["samples_taken"] >= 1
        assert stats["blocks_reported"] >= 1


# ==================== AttentionHookManager ====================

class MockSelfAttn(torch.nn.Module):
    """Mock attention module that returns (output, attn_weights)."""
    def __init__(self, n_heads, d_head, kv_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.kv_len = kv_len
        self.forward_count = 0

    def forward(self, x, output_attentions=False, **kwargs):
        self.forward_count += 1
        out = torch.randn(1, self.n_heads, 1, self.d_head, device="cuda")
        if output_attentions:
            weights = torch.rand(1, self.n_heads, 1, self.kv_len, device="cuda")
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return out, weights
        return out,


class MockModel(torch.nn.Module):
    """Mock transformer model with self_attn modules."""
    def __init__(self, n_layers, n_heads, d_head, kv_len):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            layer = torch.nn.Module()
            layer.self_attn = MockSelfAttn(n_heads, d_head, kv_len)
            self.layers.append(layer)


class TestHookManager:
    def test_install_finds_attn_modules(self, tm_handle):
        collector = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        mgr = AttentionHookManager(collector)
        model = MockModel(n_layers=3, n_heads=2, d_head=64, kv_len=16)
        mgr.install(model)

        stats = mgr.get_stats()
        assert stats["installed"] is True
        assert stats["n_layers_hooked"] == 3

    def test_hooked_forward_collects_on_sample_step(self, tm_handle):
        collector = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        mgr = AttentionHookManager(collector)
        model = MockModel(n_layers=2, n_heads=2, d_head=64, kv_len=8)
        mgr.install(model)

        x = torch.randn(1, 1, 128, device="cuda")
        for layer in model.layers:
            result = layer.self_attn(x)
            assert isinstance(result, tuple)

        stats = collector.get_stats()
        assert stats["blocks_reported"] > 0

    def test_hooked_forward_skips_non_sample(self, tm_handle):
        collector = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=5)
        mgr = AttentionHookManager(collector)
        model = MockModel(n_layers=1, n_heads=1, d_head=64, kv_len=8)
        mgr.install(model)

        collector.on_step_done()  # step → 1 (non-sample)

        x = torch.randn(1, 1, 128, device="cuda")
        result = model.layers[0].self_attn(x)
        assert isinstance(result, tuple)
        assert len(result) == 1  # no attn weights output

    def test_step_done_delegated(self, tm_handle):
        collector = AttentionScoreCollector(tm_handle, block_size=4, sample_interval=1)
        mgr = AttentionHookManager(collector)

        mgr.on_step_done()
        mgr.on_step_done()
        mgr.on_step_done()

        assert collector._step == 3

    def test_full_decode_loop(self, tm_handle):
        """Simulate a 20-step decode loop with sample_interval=5."""
        collector = AttentionScoreCollector(
            tm_handle, block_size=4, sample_interval=5)
        mgr = AttentionHookManager(collector)
        model = MockModel(n_layers=2, n_heads=2, d_head=64, kv_len=16)
        mgr.install(model)

        x = torch.randn(1, 1, 128, device="cuda")

        for step in range(20):
            for layer in model.layers:
                layer.self_attn(x)
            mgr.on_step_done()

        stats = collector.get_stats()
        assert stats["current_step"] == 20
        # Steps 0,5,10,15 are sample steps = 4 samples
        assert stats["samples_taken"] == 4
        assert stats["blocks_reported"] > 0

    def test_collector_property(self, tm_handle):
        collector = AttentionScoreCollector(tm_handle)
        mgr = AttentionHookManager(collector)
        assert mgr.collector is collector


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
