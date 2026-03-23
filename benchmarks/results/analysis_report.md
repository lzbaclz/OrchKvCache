# OrchKvCache Experiment Analysis Report
## E5: Hot/Cold Policy Sweep
- **Best config (fixed pattern)**: α=0.5 β=0.1 γ=0.4 → Hot=24, Warm=0, Cold=40
- **High α (≥0.7) avg hot (fixed)**: 16.0
- **Low α (≤0.3) avg hot (fixed)**: 24.0
- Total configs tested: 27

## E7: Prefetch Effectiveness
- **Saturation budget**: 0
- **Scheduling overhead**: 5.7~5.9 μs

## E8: Storage Bandwidth
- **GPU↔DRAM**: D2H=22.04GB/s, H2D=23.56GB/s
- **DRAM↔tmpfs**: Write=3.7GB/s, Read=14.38GB/s
- **Tier gap**: GPU↔DRAM is 1.6× faster than DRAM↔tmpfs

## E9: Scheduling Scalability
- **Range**: 64→4096 (64.0× blocks → 22.5× latency)
- **Scaling exponent**: 0.749 (1.0 = linear)
- **@4096 blocks**: avg=38.33μs, p99=57.88μs (PASS < 100μs)
- **Per-block cost**: 9.36ns
