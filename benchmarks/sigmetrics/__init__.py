"""
OrchKvCache SIGMETRICS 2027 benchmark harness.

Unified baseline comparison for head-to-head evaluation of OrchKvCache
against production baselines across models, workloads, and memory budgets.

Modules
-------
config           Experiment grid definitions (models, budgets, workloads, baselines, metrics).
workload_loader  Dataset loading and prompt preparation for each workload type.
measurement      Workload characterization: attention profiling, reuse distance, Gini/Jaccard.
run_baseline     Single-run and sweep experiment driver with metric collection.
plot_figures     Paper-ready figure generation (Figs 2-9, Table 2).
"""
