#!/usr/bin/env bash
#
# SIGMETRICS 2027 — Full Experiment Matrix
#
# Run all baseline comparisons across models, workloads, and budgets.
# Parallelisable across GPUs via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   bash benchmarks/sigmetrics/run_all.sh              # full sweep (all GPUs)
#   bash benchmarks/sigmetrics/run_all.sh --quick       # smoke test
#   bash benchmarks/sigmetrics/run_all.sh --gpu 0       # single GPU
#   bash benchmarks/sigmetrics/run_all.sh --stage 2     # only stage 2
#
# Estimated wall time (1× A100-80G):
#   Stage 1 (workload characterization):  ~30 min
#   Stage 2 (single-model quick sweep):   ~1 hour
#   Stage 3 (full 4-model sweep):         ~6-8 hours
#   Stage 4 (plot generation):            ~2 min
#   Total:                                ~8-10 hours
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

MODELS=("qwen2.5-7b" "llama-3.1-8b" "llama-2-7b" "mistral-7b")
WORKLOADS=("sharegpt" "longbench" "ruler" "rag" "agentic")
BASELINES=("gpu_only" "fifo_offload" "orchkv" "orchkv_sampling" "orchkv_qk_proxy")
BUDGETS=("0.05" "0.10" "0.25" "0.50" "0.75")

GPU_ID=0
STAGE=0          # 0 = all stages
QUICK=false
NUM_PROMPTS=32
MAX_NEW=256

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)    QUICK=true; shift ;;
        --gpu)      GPU_ID="$2"; shift 2 ;;
        --stage)    STAGE="$2"; shift 2 ;;
        --prompts)  NUM_PROMPTS="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if $QUICK; then
    MODELS=("qwen2.5-7b")
    WORKLOADS=("sharegpt" "ruler")
    BASELINES=("gpu_only" "fifo_offload" "orchkv")
    BUDGETS=("0.25" "0.50")
    NUM_PROMPTS=8
    MAX_NEW=64
    echo "[run_all] Quick mode: 1 model, 2 workloads, 3 baselines, 2 budgets"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
DEVICE="cuda:0"

PYTHON="${PYTHON:-python3}"
RUN_CMD="$PYTHON -m benchmarks.sigmetrics.run_baseline"
MEASURE_CMD="$PYTHON -m benchmarks.sigmetrics.measurement"
PLOT_CMD="$PYTHON -m benchmarks.sigmetrics.plot_figures"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_with_log() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo "[$(timestamp)] START: $label"
    echo "[$(timestamp)] CMD: $*"
    if "$@" > "$logfile" 2>&1; then
        echo "[$(timestamp)] DONE:  $label"
    else
        echo "[$(timestamp)] FAIL:  $label (see $logfile)"
        return 1
    fi
}

# =====================================================================
#  Stage 1: Workload Characterization (~30 min)
# =====================================================================

stage1() {
    echo ""
    echo "================================================================="
    echo "  Stage 1: Workload Characterization"
    echo "  Estimated time: ~30 minutes"
    echo "================================================================="

    for wl in "${WORKLOADS[@]}"; do
        run_with_log "measure_lengths_${wl}" \
            $MEASURE_CMD --model "${MODELS[0]}" --workload "$wl" \
                         --num_prompts 64 --length-only \
                         --output "$RESULTS_DIR/char_lengths_${wl}.json"
    done

    for model in "${MODELS[@]}"; do
        for wl in "sharegpt" "longbench"; do
            run_with_log "measure_attn_${model}_${wl}" \
                $MEASURE_CMD --model "$model" --workload "$wl" \
                             --num_prompts 4 --max_new_tokens 32 \
                             --sample_interval 1 --device "$DEVICE"
        done
    done
}

# =====================================================================
#  Stage 2: Quick Sweep — single model, core baselines (~1 hour)
# =====================================================================

stage2() {
    echo ""
    echo "================================================================="
    echo "  Stage 2: Quick Sweep (${MODELS[0]})"
    echo "  Estimated time: ~1 hour"
    echo "================================================================="

    run_with_log "quick_sweep" \
        $RUN_CMD --sweep \
            --models "${MODELS[0]}" \
            --workloads "${WORKLOADS[@]}" \
            --baselines "gpu_only" "fifo_offload" "orchkv" \
            --budgets "0.10" "0.25" "0.50" \
            --num_prompts "$NUM_PROMPTS" \
            --max_new_tokens "$MAX_NEW" \
            --device "$DEVICE" \
            --tag "sigmetrics_stage2"
}

# =====================================================================
#  Stage 3: Full Sweep — all models × workloads × baselines (~6-8 hours)
# =====================================================================

stage3() {
    echo ""
    echo "================================================================="
    echo "  Stage 3: Full Sweep (all ${#MODELS[@]} models)"
    echo "  Estimated time: ~6-8 hours"
    echo "================================================================="

    for model in "${MODELS[@]}"; do
        run_with_log "full_sweep_${model}" \
            $RUN_CMD --sweep \
                --models "$model" \
                --workloads "${WORKLOADS[@]}" \
                --baselines "${BASELINES[@]}" \
                --budgets "${BUDGETS[@]}" \
                --num_prompts "$NUM_PROMPTS" \
                --max_new_tokens "$MAX_NEW" \
                --device "$DEVICE" \
                --tag "sigmetrics_full_${model}"
    done
}

# =====================================================================
#  Stage 4: Plot Generation (~2 min)
# =====================================================================

stage4() {
    echo ""
    echo "================================================================="
    echo "  Stage 4: Generating Paper Figures"
    echo "================================================================="

    run_with_log "plot_figures" \
        $PLOT_CMD --results_dir "$RESULTS_DIR"
}

# =====================================================================
#  GPU parallel helper
# =====================================================================

run_parallel_gpus() {
    echo ""
    echo "================================================================="
    echo "  Parallel mode: one model per GPU"
    echo "================================================================="

    local n_gpus
    n_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "  Detected $n_gpus GPUs"

    local pids=()
    for i in "${!MODELS[@]}"; do
        local gpu=$((i % n_gpus))
        local model="${MODELS[$i]}"
        echo "  Launching ${model} on GPU ${gpu}..."
        CUDA_VISIBLE_DEVICES="$gpu" \
        $RUN_CMD --sweep \
            --models "$model" \
            --workloads "${WORKLOADS[@]}" \
            --baselines "${BASELINES[@]}" \
            --budgets "${BUDGETS[@]}" \
            --num_prompts "$NUM_PROMPTS" \
            --max_new_tokens "$MAX_NEW" \
            --device "cuda:0" \
            --tag "sigmetrics_full_${model}" \
            > "$LOG_DIR/parallel_${model}.log" 2>&1 &
        pids+=($!)
    done

    echo "  Waiting for ${#pids[@]} jobs..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        echo "  WARNING: $failed jobs failed. Check logs in $LOG_DIR/"
    else
        echo "  All parallel jobs completed successfully."
    fi
}

# =====================================================================
#  Main
# =====================================================================

main() {
    echo "================================================================="
    echo "  OrchKvCache SIGMETRICS 2027 Experiment Runner"
    echo "  GPU: $GPU_ID  Quick: $QUICK  Stage: ${STAGE:-all}"
    echo "  Models: ${MODELS[*]}"
    echo "  Workloads: ${WORKLOADS[*]}"
    echo "  Baselines: ${BASELINES[*]}"
    echo "  Budgets: ${BUDGETS[*]}"
    echo "  Prompts: $NUM_PROMPTS  Max new: $MAX_NEW"
    echo "================================================================="

    local t_start
    t_start=$(date +%s)

    if [[ $STAGE -eq 0 || $STAGE -eq 1 ]]; then stage1; fi
    if [[ $STAGE -eq 0 || $STAGE -eq 2 ]]; then stage2; fi
    if [[ $STAGE -eq 0 || $STAGE -eq 3 ]]; then stage3; fi
    if [[ $STAGE -eq 0 || $STAGE -eq 4 ]]; then stage4; fi

    local t_end
    t_end=$(date +%s)
    local elapsed=$(( (t_end - t_start) / 60 ))
    echo ""
    echo "================================================================="
    echo "  All stages complete. Total time: ${elapsed} minutes."
    echo "  Results: $RESULTS_DIR/"
    echo "  Logs:    $LOG_DIR/"
    echo "================================================================="
}

main
