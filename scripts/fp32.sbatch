#!/bin/bash
#SBATCH --job-name=ind-fp32
#SBATCH --partition=batch
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=/home/mhough/Workspace/mech-interp-artifacts/logs/fp32-%j.out
set -u
cd /home/mhough/Workspace/mech-interp-artifacts
STEPS=64,128,256,512,1000,2000,3000,4000,8000,16000,143000
PY=/home/mhough/Workspace/jacobian-lens/.venv/bin/python
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# fp32 for the big models so the whole scale axis is one dtype (bf16 corrupted 160m's mature
# checkpoint: 22/144 heads shifted >0.05). batch 4 + 6 stimulus seeds = 24 sequences, to fit 8GB.
for M in pythia-1b pythia-1.4b; do
  echo "=== $M fp32 ($(date +%H:%M:%S)) ==="
  $PY scripts/sweep_induction.py --model EleutherAI/$M --steps $STEPS \
    --dtype float32 --batch 4 --seeds 6 \
    --out data/induction_${M}_fp32.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
done
echo "FP32 DONE"
