"""
D2: Unit tests for OrchKvOffloadingConnector.

Tests the connector WITHOUT requiring vLLM installation.
Uses the fallback stub base class for all tests.

Run with: conda run -n orchkv python -m pytest test/test_connector.py -v
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from orchkv.vllm_integration.connector import (
    OrchKvConnectorWorker,
    OrchKvConnectorScheduler,
    OrchKvOffloadingConnector,
    OrchKvConnectorMetadata,
    KVConnectorRole,
)


DEFAULT_CFG = {
    "dram_pool_bytes": 32 << 20,
    "d_head": 128,
    "tokens_per_block": 16,
    "num_cuda_streams": 2,
    "io_workers": 2,
    "n_layers": 4,
}


# ===== Mock types for vLLM =====

class MockRequest:
    def __init__(self, request_id: str):
        self.request_id = request_id


class MockVllmConfig:
    def __init__(self, extra_config=None):
        self.kv_transfer_config = MockKVTransferConfig(extra_config)
        self.model_config = MockModelConfig()


class MockKVTransferConfig:
    def __init__(self, extra_config=None):
        self.kv_connector_extra_config = extra_config or {}


class MockModelConfig:
    head_dim = 128
    num_layers = 4


class MockSchedulerOutput:
    pass


class MockForwardContext:
    pass


# ===== Worker tests =====

class TestWorker:
    def setup_method(self):
        self.worker = OrchKvConnectorWorker(DEFAULT_CFG)

    def teardown_method(self):
        self.worker.shutdown()

    def test_register_kv_caches(self):
        kv = {
            "layer_0": torch.randn(8, 2, 16, 128, device="cuda"),
            "layer_1": torch.randn(8, 2, 16, 128, device="cuda"),
        }
        self.worker.register_kv_caches(kv)
        assert len(self.worker._kv_caches) == 2

    def test_save_and_load_roundtrip(self):
        n_blocks, kv_dim, block_size, d_head = 4, 2, 16, 128
        kv_gpu = torch.randn(n_blocks, kv_dim, block_size, d_head, device="cuda")
        original = kv_gpu.clone()

        self.worker.register_kv_caches({"layer_0": kv_gpu})

        meta = OrchKvConnectorMetadata()
        meta.blocks_to_save["layer_0"] = [0, 2]
        self.worker.save_kv_layer("layer_0", kv_gpu, meta,
                                    block_ids=[0, 2])
        self.worker.wait_for_save()

        assert ("layer_0", 0) in self.worker._dram_buffers
        assert ("layer_0", 2) in self.worker._dram_buffers

        kv_gpu[0].zero_()
        kv_gpu[2].zero_()

        meta_load = OrchKvConnectorMetadata()
        meta_load.blocks_to_load["layer_0"] = [0, 2]
        self.worker.start_load_kv(meta_load)
        self.worker.wait_for_layer_load("layer_0")

        assert torch.allclose(kv_gpu[0].cpu(), original[0].cpu(), atol=1e-6)
        assert torch.allclose(kv_gpu[2].cpu(), original[2].cpu(), atol=1e-6)
        assert torch.equal(kv_gpu[1].cpu(), original[1].cpu())

    def test_save_no_blocks(self):
        kv_gpu = torch.randn(4, 2, 16, 128, device="cuda")
        self.worker.register_kv_caches({"layer_0": kv_gpu})

        meta = OrchKvConnectorMetadata()
        self.worker.save_kv_layer("layer_0", kv_gpu, meta)
        self.worker.wait_for_save()
        assert len(self.worker._dram_buffers) == 0

    def test_free_blocks(self):
        kv_gpu = torch.randn(4, 2, 16, 128, device="cuda")
        self.worker.register_kv_caches({"layer_0": kv_gpu})

        meta = OrchKvConnectorMetadata()
        meta.blocks_to_save["layer_0"] = [1, 3]
        self.worker.save_kv_layer("layer_0", kv_gpu, meta,
                                    block_ids=[1, 3])
        self.worker.wait_for_save()

        assert len(self.worker._dram_buffers) == 2
        self.worker.free_blocks("layer_0", [1])
        assert len(self.worker._dram_buffers) == 1
        assert ("layer_0", 3) in self.worker._dram_buffers

    def test_stats(self):
        kv_gpu = torch.randn(4, 2, 16, 128, device="cuda")
        self.worker.register_kv_caches({"layer_0": kv_gpu})

        meta = OrchKvConnectorMetadata()
        self.worker.save_kv_layer("layer_0", kv_gpu, meta,
                                    block_ids=[0])
        self.worker.wait_for_save()

        stats = self.worker.get_stats()
        assert stats["save_count"] == 1
        assert stats["dram_buffers"] == 1

    def test_multi_layer_save_load(self):
        layers = {}
        originals = {}
        for i in range(3):
            name = f"layer_{i}"
            t = torch.randn(4, 2, 16, 128, device="cuda")
            layers[name] = t
            originals[name] = t.clone()

        self.worker.register_kv_caches(layers)

        meta = OrchKvConnectorMetadata()
        for name in layers:
            meta.blocks_to_save[name] = [1]
            self.worker.save_kv_layer(name, layers[name], meta,
                                        block_ids=[1])
        self.worker.wait_for_save()

        for name in layers:
            layers[name][1].zero_()

        meta_load = OrchKvConnectorMetadata()
        for name in layers:
            meta_load.blocks_to_load[name] = [1]
        self.worker.start_load_kv(meta_load)

        for name in layers:
            self.worker.wait_for_layer_load(name)
            assert torch.allclose(
                layers[name][1].cpu(), originals[name][1].cpu(), atol=1e-6)


# ===== Scheduler tests =====

class TestScheduler:
    def setup_method(self):
        self.sched = OrchKvConnectorScheduler(DEFAULT_CFG)

    def test_no_cached_tokens(self):
        req = MockRequest("req1")
        n, async_ = self.sched.get_num_new_matched_tokens(req, 0)
        assert n == 0
        assert not async_

    def test_cached_tokens_after_finish(self):
        req = MockRequest("req2")
        self.sched.request_finished(req, [0, 1, 2, 3])

        n, async_ = self.sched.get_num_new_matched_tokens(req, 0)
        assert n == 4
        assert async_

        n2, _ = self.sched.get_num_new_matched_tokens(req, 2)
        assert n2 == 2

    def test_build_connector_meta_empty(self):
        meta = self.sched.build_connector_meta(MockSchedulerOutput())
        assert isinstance(meta, OrchKvConnectorMetadata)
        assert len(meta.blocks_to_save) == 0
        assert len(meta.blocks_to_load) == 0

    def test_handle_preemptions(self):
        req = MockRequest("req3")
        self.sched.request_finished(req, [10, 11])
        self.sched.handle_preemptions({"req3"})

        meta = self.sched.build_connector_meta(MockSchedulerOutput())
        total_offload_blocks = sum(
            len(v) for v in meta.blocks_to_save.values())
        assert total_offload_blocks > 0


# ===== Full Connector tests =====

class TestConnectorFull:
    def test_init_worker_role(self):
        cfg = MockVllmConfig()
        conn = OrchKvOffloadingConnector(cfg, KVConnectorRole.WORKER)
        assert conn._worker is not None
        assert conn._scheduler is None
        conn.shutdown()

    def test_init_scheduler_role(self):
        cfg = MockVllmConfig()
        conn = OrchKvOffloadingConnector(cfg, KVConnectorRole.SCHEDULER)
        assert conn._scheduler is not None
        assert conn._worker is None

    def test_prefer_cross_layer_blocks(self):
        cfg = MockVllmConfig()
        conn = OrchKvOffloadingConnector(cfg, KVConnectorRole.WORKER)
        assert conn.prefer_cross_layer_blocks is True
        conn.shutdown()

    def test_worker_save_load_via_connector(self):
        cfg = MockVllmConfig({"dram_pool_gb": 1})
        conn = OrchKvOffloadingConnector(cfg, KVConnectorRole.WORKER)

        kv = {"layer_0": torch.randn(4, 2, 16, 128, device="cuda")}
        original = kv["layer_0"].clone()
        conn.register_kv_caches(kv)

        meta = OrchKvConnectorMetadata()
        meta.blocks_to_save["layer_0"] = [0]
        conn.bind_connector_metadata(meta)

        conn.save_kv_layer("layer_0", kv["layer_0"], None)
        conn.wait_for_save()

        kv["layer_0"][0].zero_()

        meta_load = OrchKvConnectorMetadata()
        meta_load.blocks_to_load["layer_0"] = [0]
        conn.bind_connector_metadata(meta_load)
        conn.start_load_kv(MockForwardContext())
        conn.wait_for_layer_load("layer_0")

        assert torch.allclose(
            kv["layer_0"][0].cpu(), original[0].cpu(), atol=1e-6)

        conn.clear_connector_metadata()
        conn.shutdown()

    def test_scheduler_request_lifecycle(self):
        cfg = MockVllmConfig()
        conn = OrchKvOffloadingConnector(cfg, KVConnectorRole.SCHEDULER)

        req = MockRequest("r1")
        n, _ = conn.get_num_new_matched_tokens(req, 0)
        assert n == 0

        conn.request_finished(req, [0, 1, 2])

        n2, async_ = conn.get_num_new_matched_tokens(req, 0)
        assert n2 == 3
        assert async_

    def test_custom_extra_config(self):
        extra = {
            "dram_pool_gb": 16,
            "d_head": 64,
            "tokens_per_block": 32,
        }
        cfg = MockVllmConfig(extra)
        conn = OrchKvOffloadingConnector(cfg, KVConnectorRole.WORKER)
        assert conn._worker._orchkv_cfg["dram_pool_bytes"] == 16 * (1 << 30)
        conn.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
