---
license: apache-2.0
task_categories: [feature-extraction]
tags: [interpretability, mechanistic-interpretability, induction-heads, training-dynamics, pythia, controls]
pretty_name: Induction-head emergence across Pythia training
size_categories: [1K<n<10K]
---

# Induction-head emergence across Pythia training

Per-head **induction**, **previous-token**, and **in-context-learning** scores across the full
training checkpoint sequence of Pythia models — plus the **causal ablation control** that turns
a correlational score into a mechanism.

**This is a data layer, not a new finding.** Induction-head emergence on Pythia has been studied
before (see *Prior work*); what has never been published is the computed scores themselves, as a
tidy downloadable table. Olsson et al. (2022), the origin of the phase-change result, used 34
**internally-trained** models — no weights, no checkpoints, no per-head data were ever released.

## Contents

| file | rows | what |
|---|---|---|
| `induction_pythia-160m.jsonl` | **154 checkpoints** | all of `step0…step143000`, incl. the log-spaced `step1–512` region |
| `induction_pythia-160m-seed{1..9}.jsonl` | 11 ckpts × 9 | **PolyPythias seed axis** |
| `induction_pythia-{410m,1b,1.4b}.jsonl` | 11 ckpts each | scale axis |
| `induction_pythia-{1b,1.4b}_fp32.jsonl` | 11 each | fp32 re-runs (dtype control) |
| `induction_pythia-160m_bf16.jsonl` | 11 | bf16/fp32 comparison |
| `ablation_pythia-160m.jsonl` | 5 ckpts | causal control |

Each row: `revision`, `step`, `dtype`, `batch`, `seqlen`, `icl_score_mean`, `icl_score_seeds`,
and a `heads` list of `{layer, head, induction_mean, induction_std, prev_token_mean}` over
3 stimulus seeds.

## What the data shows

**A sharp phase change**, flat until step 512 then complete by step 1000 (induction 0.035 →
0.963; ICL −0.02 → −9.97). Bounded by Pythia's checkpoint spacing — no checkpoints exist between
512 and 1000, so that interval is the finest timing statement this model permits.

**Invariant timing, arbitrary implementation.** All **10 seeds** and all **4 sizes** cross in
the same interval. Since Pythia uses identical data order and batch size across sizes, same step
= same tokens, so the transition is **data-determined**. Yet the top induction head is a
*different* (layer, head) in every one of the 10 seeds, spanning layers 4–8.

**A visible precursor**: at step 512 the previous-token score is 10–17× the induction score in
every seed — the prerequisite circuit forms first.

**The ablation control** tracks the phase change: ablating the top-5 induction heads costs
**−0.003** at step 512 but **+9.48** at step 1000 (+10.1 at step 143000); five random heads
≈ 0 throughout. Before the transition the top-*scoring* heads have no causal role at all.

## Caveats

- **dtype matters at the final checkpoint.** bf16 and fp32 agree to ±0.001 through the phase
  change, but at step 143000 on 160m, 22/144 heads shift by >0.05 and the argmax head changes.
  1b and 1.4b show no such effect. Use the fp32 files for mature-checkpoint claims.
- A late, gradual ICL improvement (−12.3 → −18.8 from ~step 105000 on 160m) coincides with the
  tail of Pythia's cosine LR decay — **confounded**, and it is also dtype-sensitive.
- Induction score is correlational; the ablation file is what makes it causal.
- Yin & Steinhardt argue *function-vector* heads, not induction heads, drive few-shot ICL.

## Prior work (please cite these too)

Olsson et al., *In-context Learning and Induction Heads* (2022) · Tigges, Hanna, Yu & Biderman,
*LLM Circuit Analyses Are Consistent Across Training and Scale* (NeurIPS 2024) · Yin &
Steinhardt, *Which Attention Heads Matter for In-Context Learning?* · Feucht et al., *Dual-Route
Model of Induction* (nearest prior artifact: 7 of 154 checkpoints, copying scores) · Aoyama,
Wilcox & Schneider, *Predicting the Emergence of Induction Heads* (ICML 2026).

## Reproduce

`scripts/sweep_induction.py` and `scripts/ablate_induction.py`. Probe is seconds per checkpoint;
the whole sweep is bandwidth-bound and cost under $20.

Part of *Controls & Trajectories* — publishing the null distributions and developmental
trajectories that interpretability papers rely on but rarely ship.
[Curriculum](https://github.com/m9h/spinning-up-in-mech-interp) · Morgan Hough, Orthogonal
Research and Education Lab (OREL).
