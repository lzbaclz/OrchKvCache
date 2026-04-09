# Improvement Plan v5 — Response to Reviewer (Borderline/Weak Reject, High Confidence)

## Reviewer's Core Assessment
> "Strong prototype + systems diagnosis, not yet fully compelling production-grade systems paper."
> Rating: Borderline / Weak Reject. Confidence: High.

The paper is close. The reviewer explicitly says it moved from "promising but unstable" to "serious and discussable." The gap is **framing sharpness** + **2-3 additional evidence pieces**, not fundamental rearchitecting.

---

## P0: Must-Fix (Address directly to flip the verdict)

### P0.1 — Sharpen central thesis to ONE sentence (W3, 5.1, Q1)
**Problem:** Paper tries to do three things at once (prototype system, vLLM diagnosis, future kernels argument). Reviewer cannot identify the primary takeaway.

**Action:** Rewrite abstract, final paragraph of Introduction, and Conclusion around a single thesis:

> *"Block-level, lossless, three-tier KV migration reduces unnecessary data movement by 139–597× and is highly effective at the mechanism level; current production runtimes cannot realize this value because they force request-level preemption rather than mixed-tier block access—a runtime abstraction barrier that mixed-tier attention kernels would remove."*

**Where:** Abstract (line 69), Intro final para (line 128-130), Conclusion (line 792).

### P0.2 — De-emphasize prototype throughput, lead with migration reduction (W2, 5.2)
**Problem:** 1.28–1.77× throughput headline is confounded by Python overhead. Reviewer says "migration reduction is much cleaner."

**Action:**
- Abstract: lead with "139–597× fewer migrations" and "100% lossless." Mention throughput as secondary, qualified.
- Introduction bullet list: reorder contributions so migration reduction + lossless + SSD come first; throughput is supporting evidence.
- Conclusion: same reordering.

**Where:** Abstract, Intro contributions list, Conclusion.

### P0.3 — Reframe vLLM section as deliberate "runtime abstraction barrier" finding (W1, 5.3, Q4)
**Problem:** vLLM result sits between "integration success" and "diagnostic limitation." Reviewer wants it sharpened into an explicit systems takeaway.

**Action:** Add a framing sentence at the start of §vLLM Integration:

> *"The goal of this experiment is not to demonstrate throughput gains via victim selection, but to diagnose where the abstraction barrier lies between OrchKvCache's block-level intelligence and current production runtimes."*

Then restructure the section conclusion into a crisp 3-step argument:
1. Better victim selection reduces preemption count (28%).
2. But preemption cost (recompute/swap volume) dominates throughput → FIFO is near-optimal for victim selection.
3. Online serving avoids preemption entirely → victim selection is the wrong optimization surface.
4. **Conclusion: the production path is mixed-tier block access inside the attention kernel, not better request-level preemption.**

This turns a "weakness" into a "systems insight contribution."

**Where:** §5.8 (vLLM Integration Analysis), lines 668-731.

### P0.4 — Add native-path throughput estimate (Q2, W2)
**Problem:** Reviewer asks "can you show how much of the gap disappears when Python is removed?"

**Action:** Add 2-3 sentences + 1 row in the fair-baseline table (Table XI) showing an **estimated native throughput**:
- C classifier: <40µs/step
- model.forward: 19.2ms/step
- Eliminate build_past_kv (48.3ms → ~0 with native integration)
- Estimated native step time: ~20ms → estimated throughput: ~3,000+ tok/s
- Compare to vLLM native 4,201 tok/s → gap is now only 1.4× (batching gap only)

This is a **projection**, clearly labeled as such, but it directly answers Q2.

**Where:** After Table XI (fair-baseline), add "Native-path projection" paragraph.

---

## P1: Should-Fix (Materially improves the paper)

### P1.1 — Add SSD end-to-end ablation table (W5, 5.4, Q3)
**Problem:** SSD contribution lacks clean decomposition of batching vs OrchFS vs async.

**Action:** Add a small table or extend Table II:

| SSD Write Mode | BW (GB/s) | Util. | E2E tok/s | Match |
|----------------|-----------|-------|-----------|-------|
| Per-block POSIX (1×32KB) | 0.23 | 4.3% | X | 100% |
| Batched 8 blocks (POSIX) | 2.16 | 40.8% | Y | 100% |
| Batched 8 + OrchFS align | 3.91+ | 73.8% | Z | 100% |

The microbenchmark data already exists in Table II. Need to run 2-3 E2E trials with different SSD write modes to fill in tok/s columns. This may require a small code change to force per-block vs batched SSD writes.

**Where:** New table in §5.7 (SSD Tier Validation) or extend existing.

### P1.2 — Reorganize evaluation into 3 clear groups (5.5, W6)
**Problem:** Evaluation narrative is complex (eager vs SDPA, full vs fallback, trace vs E2E, prototype vs vLLM).

**Action:** Add explicit group headers at the start of §5:

> "We organize evaluation into three groups:
> (I) **Mechanism validation** (§5.1–5.4): migration reduction, lossless correctness, throughput, ablation.
> (II) **Robustness and scaling** (§5.5–5.8): scheduling overhead, competitive ratio, hyperparameter sensitivity, signal ablation, realistic workloads, extended context.
> (III) **Production-runtime diagnosis** (§5.9): overhead decomposition, fair-baseline landscape, vLLM integration."

Then reorder subsections to match (may require moving §overhead and §fair-baseline after the robustness group).

**Where:** §5 opening paragraph (line 301).

### P1.3 — Strengthen InfiniGen comparison acknowledgment (W4)
**Problem:** Comparison is architectural, not head-to-head. Reviewer knows it's hard but flags it.

**Action:** Add 1-2 sentences in Discussion explicitly acknowledging this:

> *"A unified-stack head-to-head comparison with InfiniGen remains future work, as it would require re-implementing InfiniGen's cross-layer predictor within OrchKvCache's framework. The architectural comparison (Table XIV) and the perplexity comparison (Table XIII) are the closest feasible approximations."*

**Where:** Discussion, after InfiniGen comparison paragraph.

---

## P2: Nice-to-Have (Polish if space permits)

### P2.1 — Simplify signal ablation framing
Remove the dual "trace-level vs E2E" framing. Just present the E2E result and say trace confirms the mechanism. Currently the section switches between two evaluation modes which adds complexity.

### P2.2 — Add "intended primary takeaway" callout
Directly answer Q1 in the Scope paragraph or Conclusion. One sentence:

> *"The primary takeaway is (1): block-level three-tier migration is highly effective and lossless. The production diagnosis (§5.9) identifies the specific runtime abstraction barrier that prevents current systems from realizing this value."*

---

## Estimated Effort

| Item | Lines changed | New data needed? | Difficulty |
|------|--------------|------------------|-----------|
| P0.1 Thesis sharpening | ~15 lines rewrite | No | Easy |
| P0.2 De-emphasize throughput | ~10 lines rewrite | No | Easy |
| P0.3 Reframe vLLM | ~10 lines rewrite | No | Easy |
| P0.4 Native-path estimate | ~5 lines + 1 table row | Calculation only | Easy |
| P1.1 SSD ablation table | ~15 lines new | 2-3 E2E runs | Medium |
| P1.2 Evaluation reorg | ~5 lines + section reorder | No | Medium |
| P1.3 InfiniGen ack | ~3 lines | No | Easy |
| P2.1 Signal ablation simplify | ~5 lines cut | No | Easy |
| P2.2 Takeaway callout | ~2 lines | No | Easy |

**Total: ~70 lines of text changes + 1 new small table (SSD ablation) + optional section reordering.**

---

## Execution Order
1. P0.1 + P0.2 + P0.3 (thesis/framing — do together, ~30 min)
2. P0.4 (native-path estimate — ~15 min calculation)
3. P1.1 (SSD ablation — needs experiment, ~1 hour)
4. P1.2 (eval reorg — ~20 min)
5. P1.3 + P2.1 + P2.2 (polish — ~15 min)
