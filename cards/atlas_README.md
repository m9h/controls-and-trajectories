---
license: apache-2.0
task_categories: [image-feature-extraction]
tags: [interpretability, mechanistic-interpretability, vision, circuits, inceptionv1, controls]
pretty_name: InceptionV1 tuning atlas with randomization nulls
size_categories: [1K<n<10K]
---

# InceptionV1 tuning atlas — every unit, with its nulls

Quantitative per-unit orientation and curvature tuning for **all 5,808 units** of InceptionV1
(torchvision `googlenet`), each shipped with **two null distributions**.

**Why this exists.** The Distill *Circuits* thread established oriented-edge and curve detectors
with feature visualizations and rendered tuning-curve widgets. None of it was published as
numbers, no randomization control appears anywhere in the thread, and OpenAI Microscope — the
visualization layer — has returned HTTP 503 since roughly January 2025. The founding rung of
mechanistic interpretability is currently its least reproducible.

## Fields (one JSON object per unit)

`layer`, `unit`, `osi` (orientation selectivity, 1 − circular variance, **measured at that
unit's preferred spatial frequency**), `pref_orientation_deg`, `pref_curvature`,
`curve_selectivity_index`, `response_magnitude`, and per-null:
`null_osi_{mean,std,max}`, `osi_z_vs_null`, `osi_exceeds_all_nulls`, `null_alive`,
`shuffle_osi_{mean,max}`, `osi_exceeds_all_shuffles`, `shuffle_alive`.

## Two nulls, because the standard one is too weak

- **Random-init** (Adebayo et al. 2018): same architecture, random weights.
- **Weight-shuffle**: permute each trained kernel's weights within-channel — preserves the
  weight distribution and keeps activations alive.

Both are distributions over 5 seeds. The distinction matters:

| | beats random-init | beats weight-shuffle |
|---|---|---|
| `conv1` | 34/64 | 40/64 |
| `inception5b` | **100%** | **46%** |

A randomly-initialized network is nearly **dead** below `inception3a` (0% of units respond), so
"beats the random-init null" is trivially true there. `null_alive` flags exactly where the
weaker null is uninformative.

## Honest headline

Trained top OSI **1.000** vs random-init **0.528** and shuffle **0.980** — the sharpest
detectors are real. But the **median** conv1 unit sits at ~0.25: *"conv1 is all Gabor filters"*
is too strong a reading, and the numbers say so where the pictures could not.

## Method traps this cost us

1. Measure orientation tuning at each unit's **preferred spatial frequency** — averaging across
   frequencies inverted the conv1 result entirely.
2. Use a **bounded** curve index, not a ratio; a near-zero straight-line response produced
   "selectivities" of 2,000,000.
3. A **dead null is not a null**.

## Caveat on unit indices

These are **torchvision `googlenet`** channel indices. They are *not* known to correspond to the
unit numbering in the Distill articles, which refers to the lucid/TF-slim InceptionV1
checkpoint. Do not join these to those labels without first establishing the mapping.

Reproduce: `scripts/inceptionv1_atlas.py` (CPU, minutes).
Interactive viewer and curriculum: [spinning-up-in-mech-interp](https://github.com/m9h/spinning-up-in-mech-interp).
Morgan Hough, Orthogonal Research and Education Lab (OREL).
