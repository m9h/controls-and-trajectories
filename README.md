# Controls & Trajectories

Code behind the *Controls & Trajectories* artifact program: publishing the **null distributions**
and **developmental trajectories** that interpretability papers rely on but rarely ship.

Every source paper's evidence splits in two — the **substrate** (weights, SAEs, transcoders,
feature visualizations), which is largely public, and the **control layer** (nulls, baselines,
emergence-across-training curves), which mostly is not. This publishes the second.

## Datasets on the Hub

| dataset | what |
|---|---|
| [induction-emergence-pythia](https://huggingface.co/datasets/mhough/induction-emergence-pythia) | per-head induction / prev-token / ICL scores across **154 Pythia checkpoints**, 9 PolyPythias seeds, 4 model sizes, fp32 dtype controls, and the causal ablation |
| [inceptionv1-tuning-atlas](https://huggingface.co/datasets/mhough/inceptionv1-tuning-atlas) | **5,808 InceptionV1 units**, orientation selectivity at each unit's preferred spatial frequency, with **two** null distributions × 5 seeds |
| [trilens-instrument-agreement](https://huggingface.co/datasets/mhough/trilens-instrument-agreement) | three instruments read the same activation; mismatch nulls and steering ground truth |

## Selected findings

- **The induction phase change is invariant to seed and scale, but its implementation is
  arbitrary.** All 10 seeds and all 4 sizes cross in the same interval (step 512→1000) — and
  since Pythia uses identical data order across sizes, same step = same tokens, so the
  transition is *data-determined*. Yet the top induction head is a different (layer, head) in
  **every one** of the 10 seeds.
- **The score precedes the mechanism.** Ablating the top-scoring induction heads costs −0.003 at
  step 512 and +9.48 one checkpoint later.
- **A dead null is not a null.** 100% of `inception5b` units beat a randomly-initialized network;
  only 46% beat a weight-shuffled one, because a random deep net is nearly inert.

## Scripts

`sweep_induction.py` · `ablate_induction.py` · `inceptionv1_atlas.py` · `export_viewer_payload.py`

Apache-2.0. Morgan Hough, Orthogonal Research and Education Lab (OREL).
Curriculum: [spinning-up-in-mech-interp](https://github.com/m9h/spinning-up-in-mech-interp).

## Related

Part of one program — a controls-first, open-weights attempt to make the 2025–26 interpretability
claims checkable:

| repo | what |
|---|---|
| [spinning-up-in-mech-interp](https://github.com/m9h/spinning-up-in-mech-interp) | the **curriculum** — 8 rungs, 6 runnable on a laptop, each ending in its own null |
| [jacobian-lens](https://github.com/m9h/jacobian-lens) | the **research** — OLMo post-training ladder, metacognition, the Consciousness-Indicator Scorecard |
| [tri-lens](https://github.com/m9h/tri-lens) | do **three instruments agree** about the same activation? |
| [societies-of-thought](https://github.com/m9h/societies-of-thought) | the **adversarial replication** — rebuild a no-code/no-data paper, then try to break it |
| [controls-and-trajectories](https://github.com/m9h/controls-and-trajectories) | the **published datasets** — nulls and developmental trajectories |

Datasets: [induction-emergence-pythia](https://huggingface.co/datasets/mhough/induction-emergence-pythia) ·
[inceptionv1-tuning-atlas](https://huggingface.co/datasets/mhough/inceptionv1-tuning-atlas) ·
[trilens-instrument-agreement](https://huggingface.co/datasets/mhough/trilens-instrument-agreement) ·
[olmo3-jacobian-lenses](https://huggingface.co/mhough/olmo3-jacobian-lenses)
