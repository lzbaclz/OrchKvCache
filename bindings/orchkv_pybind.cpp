/*
 * Phase D (D1): Python binding for OrchKvCache via pybind11.
 *
 * Exposes the C API (Phase A/B) and tiered_manager (Phase C)
 * as the Python module `orchkv_core`.
 *
 * GPU pointer interop: PyTorch tensor.data_ptr() → uintptr_t → void*.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

extern "C" {
#include "core/kv_types.h"
#include "core/kv_block.h"
#include "core/kv_request.h"
#include "api/orchkv_api.h"
#include "scheduler/tiered_manager.h"
}

PYBIND11_MODULE(orchkv_core, m) {
    m.doc() = "OrchKvCache: Tiered KV-Cache management for LLM inference";

    /* ==== Enumerations ==== */

    py::enum_<StorageTier>(m, "StorageTier")
        .value("GPU_HBM",   TIER_GPU_HBM)
        .value("HOST_DRAM",  TIER_HOST_DRAM)
        .value("NVM",        TIER_NVM)
        .value("SSD",        TIER_SSD)
        .value("NONE",       TIER_NONE)
        .export_values();

    py::enum_<DataType>(m, "DataType")
        .value("FP16", DTYPE_FP16)
        .value("BF16", DTYPE_BF16)
        .value("FP32", DTYPE_FP32)
        .value("INT8", DTYPE_INT8)
        .value("INT4", DTYPE_INT4)
        .export_values();

    /* ==== Error codes ==== */

    m.attr("OK")            = ORCHKV_OK;
    m.attr("ERR_OOM")       = ORCHKV_ERR_OOM;
    m.attr("ERR_INVALID")   = ORCHKV_ERR_INVALID;
    m.attr("ERR_NOT_FOUND") = ORCHKV_ERR_NOT_FOUND;
    m.attr("ERR_TIER_FULL") = ORCHKV_ERR_TIER_FULL;
    m.attr("ERR_CUDA")      = ORCHKV_ERR_CUDA;
    m.attr("ERR_IO")        = ORCHKV_ERR_IO;

    /* ==== Configuration ==== */

    py::class_<orchkv_config_t>(m, "Config")
        .def(py::init([]() {
            orchkv_config_t cfg;
            orchkv_config_default(&cfg);
            return cfg;
        }))
        .def_readwrite("gpu_device_id",    &orchkv_config_t::gpu_device_id)
        .def_readwrite("gpu_pool_bytes",   &orchkv_config_t::gpu_pool_bytes)
        .def_readwrite("dram_pool_bytes",  &orchkv_config_t::dram_pool_bytes)
        .def_readwrite("dram_use_pinned",  &orchkv_config_t::dram_use_pinned)
        .def_readwrite("num_cuda_streams", &orchkv_config_t::num_cuda_streams)
        .def_readwrite("tokens_per_block", &orchkv_config_t::tokens_per_block)
        .def_readwrite("d_head",           &orchkv_config_t::d_head)
        .def_readwrite("dtype",            &orchkv_config_t::dtype)
        .def_readwrite("gpu_hwm",          &orchkv_config_t::gpu_hwm)
        .def_readwrite("gpu_lwm",          &orchkv_config_t::gpu_lwm)
        .def_readwrite("dram_hwm",         &orchkv_config_t::dram_hwm)
        .def_readwrite("dram_lwm",         &orchkv_config_t::dram_lwm)
        .def_readwrite("orchfs_io_workers",     &orchkv_config_t::orchfs_io_workers)
        .def_readwrite("max_blocks_per_head",   &orchkv_config_t::max_blocks_per_head)
        .def("__repr__", [](const orchkv_config_t &c) {
            return "<orchkv.Config gpu_pool=" + std::to_string(c.gpu_pool_bytes >> 20) +
                   "MB dram_pool=" + std::to_string(c.dram_pool_bytes >> 20) +
                   "MB d_head=" + std::to_string(c.d_head) + ">";
        });

    /* ==== System lifecycle ==== */

    m.def("init", &orchkv_init,
          py::arg("config"),
          "Initialize the OrchKvCache system");

    m.def("shutdown", &orchkv_shutdown,
          "Shutdown and free all resources");

    m.def("is_initialized", &orchkv_is_initialized,
          "Check if the system is initialized");

    /* ==== Request lifecycle ==== */

    m.def("request_create",
          [](uint64_t request_id, uint32_t n_layers, uint32_t n_kv_heads) -> uintptr_t {
              kv_request_ctx_t *ctx = orchkv_request_create(request_id, n_layers, n_kv_heads);
              return reinterpret_cast<uintptr_t>(ctx);
          },
          py::arg("request_id"), py::arg("n_layers"), py::arg("n_kv_heads"),
          "Create a new KV request context (returns opaque handle)");

    m.def("request_destroy",
          [](uintptr_t ctx_ptr) {
              return orchkv_request_destroy(reinterpret_cast<kv_request_ctx_t*>(ctx_ptr));
          },
          py::arg("ctx"),
          "Destroy a request and release its blocks");

    /* ==== Data operations ==== */

    m.def("prefill",
          [](uintptr_t ctx_ptr, uint32_t layer,
             uintptr_t k_ptr, uintptr_t v_ptr, uint32_t seq_len) {
              return orchkv_prefill(
                  reinterpret_cast<kv_request_ctx_t*>(ctx_ptr), layer,
                  reinterpret_cast<const void*>(k_ptr),
                  reinterpret_cast<const void*>(v_ptr),
                  seq_len);
          },
          py::arg("ctx"), py::arg("layer"),
          py::arg("k_ptr"), py::arg("v_ptr"), py::arg("seq_len"),
          "Prefill KV data (pass tensor.data_ptr() as k_ptr/v_ptr)");

    m.def("get_kv_block",
          [](uintptr_t ctx_ptr, uint32_t layer,
             uint32_t head, uint32_t block_idx) -> py::tuple {
              void *k_out = nullptr, *v_out = nullptr;
              int rc = orchkv_get_kv_block(
                  reinterpret_cast<kv_request_ctx_t*>(ctx_ptr),
                  layer, head, block_idx, &k_out, &v_out);
              return py::make_tuple(rc,
                                   reinterpret_cast<uintptr_t>(k_out),
                                   reinterpret_cast<uintptr_t>(v_out));
          },
          py::arg("ctx"), py::arg("layer"), py::arg("head"), py::arg("block_idx"),
          "Get KV block GPU pointers as (rc, k_ptr, v_ptr)");

    /* ==== Migration: GPU <-> DRAM ==== */

    m.def("evict_to_dram",
          [](uintptr_t ctx, uint32_t l, uint32_t h, uint32_t bi) {
              return orchkv_evict_to_dram(
                  reinterpret_cast<kv_request_ctx_t*>(ctx), l, h, bi);
          },
          py::arg("ctx"), py::arg("layer"), py::arg("head"), py::arg("block_idx"));

    m.def("promote_to_gpu",
          [](uintptr_t ctx, uint32_t l, uint32_t h, uint32_t bi) {
              return orchkv_promote_to_gpu(
                  reinterpret_cast<kv_request_ctx_t*>(ctx), l, h, bi);
          },
          py::arg("ctx"), py::arg("layer"), py::arg("head"), py::arg("block_idx"));

    /* ==== Migration: DRAM <-> Storage ==== */

    m.def("evict_to_storage",
          [](uintptr_t ctx, uint32_t l, uint32_t h, uint32_t bi) {
              return orchkv_evict_to_storage(
                  reinterpret_cast<kv_request_ctx_t*>(ctx), l, h, bi);
          },
          py::arg("ctx"), py::arg("layer"), py::arg("head"), py::arg("block_idx"));

    m.def("promote_from_storage",
          [](uintptr_t ctx, uint32_t l, uint32_t h, uint32_t bi) {
              return orchkv_promote_from_storage(
                  reinterpret_cast<kv_request_ctx_t*>(ctx), l, h, bi);
          },
          py::arg("ctx"), py::arg("layer"), py::arg("head"), py::arg("block_idx"));

    m.def("evict_cold",
          [](uintptr_t ctx, uint32_t l, uint32_t h, uint32_t bi) {
              return orchkv_evict_cold(
                  reinterpret_cast<kv_request_ctx_t*>(ctx), l, h, bi);
          },
          py::arg("ctx"), py::arg("layer"), py::arg("head"), py::arg("block_idx"));

    m.def("storage_flush", &orchkv_storage_flush);

    /* ==== Statistics (Phase A/B) ==== */

    m.def("get_stats", []() -> py::dict {
        orchkv_stats_t s;
        orchkv_get_stats(&s);
        py::dict d;
        d["gpu_pool_total"]   = s.gpu_pool_total;
        d["gpu_pool_used"]    = s.gpu_pool_used;
        d["gpu_slabs_total"]  = s.gpu_slabs_total;
        d["gpu_slabs_used"]   = s.gpu_slabs_used;
        d["dram_pool_total"]  = s.dram_pool_total;
        d["dram_pool_used"]   = s.dram_pool_used;
        d["dram_slabs_total"] = s.dram_slabs_total;
        d["dram_slabs_used"]  = s.dram_slabs_used;
        d["total_blocks"]     = s.total_blocks;
        d["blocks_on_gpu"]    = s.blocks_on_gpu;
        d["blocks_on_dram"]   = s.blocks_on_dram;
        d["blocks_on_nvm"]    = s.blocks_on_nvm;
        d["blocks_on_ssd"]    = s.blocks_on_ssd;
        d["transfers_d2h"]    = s.transfers_d2h;
        d["transfers_h2d"]    = s.transfers_h2d;
        d["bytes_d2h"]        = s.bytes_d2h;
        d["bytes_h2d"]        = s.bytes_h2d;
        d["active_requests"]  = s.active_requests;
        return d;
    }, "Return system statistics as a dict");

    /* ========================================================
     *  Phase C: tiered_manager
     * ======================================================== */

    m.def("tm_create",
          [](uint32_t tracker_cap, float ema_lambda,
             float alpha, float beta, float gamma,
             uint32_t prefetch_budget,
             uint32_t schedule_interval_us) -> uintptr_t {
              auto *m_ptr = new tiered_manager_t();
              tm_config_t cfg;
              tm_config_default(&cfg);
              cfg.tracker_capacity      = tracker_cap;
              cfg.ema_lambda            = ema_lambda;
              cfg.hcc_params.alpha      = alpha;
              cfg.hcc_params.beta       = beta;
              cfg.hcc_params.gamma      = gamma;
              cfg.prefetch_budget       = prefetch_budget;
              cfg.schedule_interval_us  = schedule_interval_us;
              cfg.auto_schedule         = false;
              int rc = tm_init(m_ptr, &cfg);
              if (rc != ORCHKV_OK) {
                  delete m_ptr;
                  throw std::runtime_error("tm_init failed: " + std::to_string(rc));
              }
              return reinterpret_cast<uintptr_t>(m_ptr);
          },
          py::arg("tracker_cap") = 4096,
          py::arg("ema_lambda") = 0.9f,
          py::arg("alpha") = 0.5f,
          py::arg("beta") = 0.3f,
          py::arg("gamma") = 0.2f,
          py::arg("prefetch_budget") = 16,
          py::arg("schedule_interval_us") = 1000,
          "Create a tiered_manager (returns opaque handle)");

    m.def("tm_destroy",
          [](uintptr_t tm_ptr) {
              auto *m_ptr = reinterpret_cast<tiered_manager_t*>(tm_ptr);
              tm_destroy(m_ptr);
              delete m_ptr;
          },
          py::arg("tm"),
          "Destroy a tiered_manager");

    m.def("tm_report_attn",
          [](uintptr_t tm_ptr, uint64_t block_id, float weight) {
              tm_notify_attn(reinterpret_cast<tiered_manager_t*>(tm_ptr),
                             block_id, weight);
          },
          py::arg("tm"), py::arg("block_id"), py::arg("attn_weight"),
          "Report attention score for a block in the current step");

    m.def("tm_step_done",
          [](uintptr_t tm_ptr) {
              tm_step_done(reinterpret_cast<tiered_manager_t*>(tm_ptr));
          },
          py::arg("tm"),
          "Mark current decode step as complete");

    m.def("tm_set_usage",
          [](uintptr_t tm_ptr, float gpu, float dram) {
              tm_set_usage(reinterpret_cast<tiered_manager_t*>(tm_ptr), gpu, dram);
          },
          py::arg("tm"), py::arg("gpu_ratio"), py::arg("dram_ratio"),
          "Update GPU/DRAM usage ratios");

    m.def("tm_schedule_once",
          [](uintptr_t tm_ptr) {
              tm_schedule_once(reinterpret_cast<tiered_manager_t*>(tm_ptr));
          },
          py::arg("tm"),
          "Run one scheduling iteration");

    m.def("tm_start",
          [](uintptr_t tm_ptr) {
              return tm_start(reinterpret_cast<tiered_manager_t*>(tm_ptr));
          },
          py::arg("tm"),
          "Start background scheduler thread");

    m.def("tm_stop",
          [](uintptr_t tm_ptr) {
              tm_stop(reinterpret_cast<tiered_manager_t*>(tm_ptr));
          },
          py::arg("tm"),
          "Stop background scheduler thread");

    m.def("tm_set_policy",
          [](uintptr_t tm_ptr, float a, float b, float g) {
              tm_set_policy(reinterpret_cast<tiered_manager_t*>(tm_ptr), a, b, g);
          },
          py::arg("tm"), py::arg("alpha"), py::arg("beta"), py::arg("gamma"),
          "Adjust hotness formula weights at runtime");

    m.def("tm_get_stats",
          [](uintptr_t tm_ptr) -> py::dict {
              tm_stats_t s;
              tm_get_stats(reinterpret_cast<tiered_manager_t*>(tm_ptr), &s);
              py::dict d;
              d["schedule_cycles"]       = s.schedule_cycles;
              d["gpu_demotes"]           = s.gpu_demotes;
              d["dram_demotes"]          = s.dram_demotes;
              d["prefetches_dispatched"] = s.prefetches_dispatched;
              d["gpu_used_ratio"]        = s.gpu_used_ratio;
              d["dram_used_ratio"]       = s.dram_used_ratio;
              d["blocks_migrated"]       = s.migration_stats.blocks_migrated;
              d["migration_errors"]      = s.migration_stats.op_errors;
              d["prefetch_hits"]         = s.prefetch_stats.prefetch_hits;
              d["prefetch_wasted"]       = s.prefetch_stats.prefetch_wasted;
              d["prefetch_hit_rate"]     = s.prefetch_stats.hit_rate;
              d["n_hot"]                 = s.hcc_stats.n_hot;
              d["n_warm"]               = s.hcc_stats.n_warm;
              d["n_cold"]                = s.hcc_stats.n_cold;
              return d;
          },
          py::arg("tm"),
          "Return tiered_manager statistics as a dict");
}
