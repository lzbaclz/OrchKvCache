#!/bin/bash
# InfiniGen vs FlexGen baseline throughput comparison on OPT-1.3B
#
# Prerequisites:
#   - conda env "infinigen" with torch 2.0.x, numpy<2
#   - InfiniGen repo at /home/lzq/codes/InfiniGen/speedup
#
# Usage: bash benchmarks/run_infinigen_comparison.sh

set -e

INFINIGEN_DIR=/home/lzq/codes/InfiniGen/speedup
FLEXGEN_DIR=$INFINIGEN_DIR/flexgen
INPUT_FILE=$FLEXGEN_DIR/pg19_firstbook.txt
RESULTS_DIR=$(dirname $0)/results
PYTHON=/home/lzq/miniconda3/envs/infinigen/bin/python
mkdir -p $RESULTS_DIR

export PYTHONPATH=$INFINIGEN_DIR/infinigen:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

MODEL="facebook/opt-1.3b"
BSZ=4
GEN_LEN=128

switch_scheme() {
    local scheme=$1
    cd $FLEXGEN_DIR
    rm -f flexgen/flex_opt.py flexgen/pytorch_backend.py
    ln -s ../$scheme/flex_opt.py flexgen/flex_opt.py
    ln -s ../$scheme/pytorch_backend.py flexgen/pytorch_backend.py
    echo "  Switched to scheme: $scheme"
    cd -
}

run_config() {
    local PROMPT_LEN=$1
    echo ""
    echo "============================================================"
    echo "  Config: $MODEL  bs=$BSZ  prompt=$PROMPT_LEN  gen=$GEN_LEN"
    echo "============================================================"

    # 1. FlexGen Original (baseline)
    echo ""
    echo ">>> [1/3] FlexGen Original (baseline) ..."
    switch_scheme "original"
    cd $FLEXGEN_DIR
    $PYTHON -u -m flexgen.flex_opt \
        --model $MODEL \
        --percent 100 0 0 100 100 0 \
        --overlap false \
        --gpu-batch-size $BSZ \
        --num-gpu-batches 1 \
        --prompt-len $PROMPT_LEN \
        --gen-len $GEN_LEN \
        --warmup-input-path $INPUT_FILE \
        --test-input-path $INPUT_FILE \
        2>&1 | tee $RESULTS_DIR/infinigen_original_p${PROMPT_LEN}.log
    cd -

    # 2. InfiniGen
    echo ""
    echo ">>> [2/3] InfiniGen ..."
    switch_scheme "infinigen"
    cd $FLEXGEN_DIR
    $PYTHON -u -m flexgen.flex_opt \
        --model $MODEL \
        --percent 100 0 0 100 100 0 \
        --overlap false \
        --gpu-batch-size $BSZ \
        --num-gpu-batches 1 \
        --prompt-len $PROMPT_LEN \
        --gen-len $GEN_LEN \
        --warmup-input-path $INPUT_FILE \
        --test-input-path $INPUT_FILE \
        --alpha 4 \
        --partial-weight-ratio 0.2 \
        --max-num-kv 200 \
        2>&1 | tee $RESULTS_DIR/infinigen_infinigen_p${PROMPT_LEN}.log
    cd -

    # 3. H2O
    echo ""
    echo ">>> [3/3] H2O ..."
    switch_scheme "h2o"
    cd $FLEXGEN_DIR
    $PYTHON -u -m flexgen.flex_opt \
        --model $MODEL \
        --percent 100 0 0 100 100 0 \
        --overlap false \
        --gpu-batch-size $BSZ \
        --num-gpu-batches 1 \
        --prompt-len $PROMPT_LEN \
        --gen-len $GEN_LEN \
        --warmup-input-path $INPUT_FILE \
        --test-input-path $INPUT_FILE \
        --max-num-kv 400 \
        --hh-ratio 0.1 \
        --hh-all \
        2>&1 | tee $RESULTS_DIR/infinigen_h2o_p${PROMPT_LEN}.log
    cd -
}

for PLEN in 512 1024 1536; do
    run_config $PLEN
done

echo ""
echo "============================================================"
echo "  Done. Extracting results..."
echo "============================================================"
grep -i "Total:\|FlexGen\|InfiniGen\|H2O\|input:" $RESULTS_DIR/infinigen_*_p*.log 2>/dev/null || echo "  (parse logs manually)"
