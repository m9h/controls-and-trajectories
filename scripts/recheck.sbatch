#!/bin/bash
#SBATCH --job-name=recheck160
#SBATCH --partition=quick
#SBATCH --mem=12G
#SBATCH --time=0:30:00
#SBATCH --output=/home/mhough/Workspace/mech-interp-artifacts/logs/recheck-%j.out
cd /home/mhough/Workspace/mech-interp-artifacts
export HF_HUB_DISABLE_XET=1
PY=/home/mhough/Workspace/jacobian-lens/.venv/bin/python
# Is 160m step143000's bf16 anomaly reproducible, or was it a fluke? 8 stimulus seeds each.
$PY scripts/sweep_induction.py --model EleutherAI/pythia-160m --steps 143000 --dtype bfloat16 \
  --seeds 8 --out /tmp/recheck_bf16.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
$PY scripts/sweep_induction.py --model EleutherAI/pythia-160m --steps 143000 --dtype float32 \
  --seeds 8 --out /tmp/recheck_fp32.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
