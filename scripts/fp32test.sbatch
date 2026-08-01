#!/bin/bash
#SBATCH --job-name=fp32test
#SBATCH --partition=quick
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/home/mhough/Workspace/mech-interp-artifacts/logs/fp32test-%j.out
cd /home/mhough/Workspace/mech-interp-artifacts
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/mhough/Workspace/jacobian-lens/.venv/bin/python scripts/sweep_induction.py \
  --model EleutherAI/pythia-1.4b --steps 143000 --dtype float32 --batch 4 --seeds 2 \
  --out /tmp/fp32_fit_test.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
