#!/bin/bash
#SBATCH --job-name=ind-seeds
#SBATCH --partition=batch
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=/home/mhough/Workspace/mech-interp-artifacts/logs/seeds-%j.out
set -u
cd /home/mhough/Workspace/mech-interp-artifacts
STEPS=64,128,256,512,1000,2000,3000,4000,8000,16000,143000
PY=/home/mhough/Workspace/jacobian-lens/.venv/bin/python
export HF_HUB_DISABLE_XET=1
for s in 1 2 3 4 5 6 7 8 9; do
  echo "=== seed $s ($(date +%H:%M:%S)) ==="
  $PY scripts/sweep_induction.py \
    --model EleutherAI/pythia-160m-seed$s \
    --steps $STEPS \
    --out data/induction_pythia-160m-seed$s.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
done
echo "ALL SEEDS DONE"
