"""
D1: Python binding unit tests for orchkv_core.

Tests the pybind11 binding without requiring full vLLM integration.
Run with: python -m pytest test/test_binding.py -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'bindings'))

import orchkv_core as _C  # noqa: E402


class TestEnums:
    def test_storage_tier_values(self):
        assert _C.StorageTier.GPU_HBM.value == 0
        assert _C.StorageTier.HOST_DRAM.value == 1
        assert _C.StorageTier.NVM.value == 2
        assert _C.StorageTier.SSD.value == 3

    def test_data_type_values(self):
        assert _C.DataType.FP16.value == 0
        assert _C.DataType.FP32.value == 2

    def test_error_codes(self):
        assert _C.OK == 0
        assert _C.ERR_OOM < 0
        assert _C.ERR_CUDA < 0


class TestConfig:
    def test_default_config(self):
        cfg = _C.Config()
        assert cfg.d_head == 128
        assert cfg.tokens_per_block == 64
        assert cfg.gpu_pool_bytes > 0
        assert cfg.dram_pool_bytes > 0

    def test_modify_config(self):
        cfg = _C.Config()
        cfg.gpu_pool_bytes = 4 * (1 << 30)
        cfg.d_head = 64
        assert cfg.gpu_pool_bytes == 4 * (1 << 30)
        assert cfg.d_head == 64

    def test_config_repr(self):
        cfg = _C.Config()
        cfg.gpu_pool_bytes = 32 << 20
        r = repr(cfg)
        assert 'Config' in r
        assert '32MB' in r


class TestLifecycle:
    def test_init_shutdown(self):
        cfg = _C.Config()
        cfg.gpu_pool_bytes = 32 << 20
        cfg.dram_pool_bytes = 32 << 20
        cfg.d_head = 128
        cfg.num_cuda_streams = 2
        cfg.orchfs_io_workers = 2
        cfg.max_blocks_per_head = 64

        assert not _C.is_initialized()
        rc = _C.init(cfg)
        assert rc == _C.OK
        assert _C.is_initialized()

        rc = _C.shutdown()
        assert rc == _C.OK
        assert not _C.is_initialized()


class TestRequestAndData:
    @classmethod
    def setup_class(cls):
        cfg = _C.Config()
        cfg.gpu_pool_bytes = 32 << 20
        cfg.dram_pool_bytes = 32 << 20
        cfg.d_head = 128
        cfg.num_cuda_streams = 2
        cfg.orchfs_io_workers = 2
        cfg.max_blocks_per_head = 64
        _C.init(cfg)

    @classmethod
    def teardown_class(cls):
        _C.shutdown()

    def test_request_create_destroy(self):
        ctx = _C.request_create(request_id=1, n_layers=2, n_kv_heads=2)
        assert ctx != 0
        rc = _C.request_destroy(ctx)
        assert rc == _C.OK

    def test_stats(self):
        stats = _C.get_stats()
        assert isinstance(stats, dict)
        assert 'gpu_slabs_total' in stats
        assert 'dram_slabs_total' in stats
        assert 'blocks_on_gpu' in stats

    def test_prefill_and_get_block(self):
        import torch
        ctx = _C.request_create(request_id=10, n_layers=1, n_kv_heads=2)
        assert ctx != 0

        seq_len = 64
        d_head = 128
        n_heads = 2
        k = torch.randn(seq_len * n_heads, d_head, dtype=torch.float16, device='cuda')
        v = torch.randn(seq_len * n_heads, d_head, dtype=torch.float16, device='cuda')

        rc = _C.prefill(ctx, layer=0,
                        k_ptr=k.data_ptr(), v_ptr=v.data_ptr(),
                        seq_len=seq_len)
        assert rc == _C.OK

        rc2, k_ptr, v_ptr = _C.get_kv_block(ctx, layer=0, head=0, block_idx=0)
        assert rc2 == _C.OK
        assert k_ptr != 0
        assert v_ptr != 0

        _C.request_destroy(ctx)

    def test_evict_promote_roundtrip(self):
        import torch
        ctx = _C.request_create(request_id=20, n_layers=1, n_kv_heads=1)
        seq_len = 64
        d_head = 128
        k = torch.randn(seq_len, d_head, dtype=torch.float16, device='cuda')
        v = torch.randn(seq_len, d_head, dtype=torch.float16, device='cuda')
        _C.prefill(ctx, 0, k.data_ptr(), v.data_ptr(), seq_len)

        rc = _C.evict_to_dram(ctx, layer=0, head=0, block_idx=0)
        assert rc == _C.OK

        rc = _C.promote_to_gpu(ctx, layer=0, head=0, block_idx=0)
        assert rc == _C.OK

        _C.request_destroy(ctx)


class TestTieredManager:
    def test_create_destroy(self):
        tm = _C.tm_create()
        assert tm != 0
        _C.tm_destroy(tm)

    def test_report_attn_and_step(self):
        tm = _C.tm_create(tracker_cap=256)
        for step in range(10):
            _C.tm_report_attn(tm, block_id=42, attn_weight=0.8)
            _C.tm_step_done(tm)

        stats = _C.tm_get_stats(tm)
        assert isinstance(stats, dict)
        assert stats['schedule_cycles'] == 0
        _C.tm_destroy(tm)

    def test_set_usage_and_schedule(self):
        tm = _C.tm_create(tracker_cap=256)
        _C.tm_set_usage(tm, gpu_ratio=0.5, dram_ratio=0.5)
        _C.tm_schedule_once(tm)

        stats = _C.tm_get_stats(tm)
        assert stats['schedule_cycles'] == 1
        _C.tm_destroy(tm)

    def test_set_policy(self):
        tm = _C.tm_create()
        _C.tm_set_policy(tm, alpha=0.8, beta=0.1, gamma=0.1)
        _C.tm_destroy(tm)

    def test_start_stop(self):
        tm = _C.tm_create(schedule_interval_us=500)
        rc = _C.tm_start(tm)
        assert rc == _C.OK

        import time
        time.sleep(0.01)  # let it run a few cycles

        _C.tm_stop(tm)
        stats = _C.tm_get_stats(tm)
        assert stats['schedule_cycles'] >= 1
        _C.tm_destroy(tm)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
