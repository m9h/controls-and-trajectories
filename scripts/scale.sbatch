#!/bin/bash
#SBATCH --job-name=ind-scale
#SBATCH --partition=batch
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/mhough/Workspace/mech-interp-artifacts/logs/scale-%j.out
set -u
cd /home/mhough/Workspace/mech-interp-artifacts
STEPS=64,128,256,512,1000,2000,3000,4000,8000,16000,143000
PY=/home/mhough/Workspace/jacobian-lens/.venv/bin/python
export HF_HUB_DISABLE_XET=1
# dtype consistency control: same model+ckpts in bf16, to show the score is dtype-robust
echo "=== dtype control: pythia-160m in bfloat16 ==="
$PY scripts/sweep_induction.py --model EleutherAI/pythia-160m --steps $STEPS \
  --dtype bfloat16 --out data/induction_pythia-160m_bf16.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
# scale axis. 8GB card: fp32 up to 410m, bf16 above (1.4b fp32 = 5.7GB, will not fit).
echo "=== pythia-410m (fp32) ==="
$PY scripts/sweep_induction.py --model EleutherAI/pythia-410m --steps $STEPS \
  --out data/induction_pythia-410m.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
for M in pythia-1b pythia-1.4b; do
  echo "=== $M (bf16) ==="
  $PY scripts/sweep_induction.py --model EleutherAI/$M --steps $STEPS \
    --dtype bfloat16 --batch 8 --out data/induction_$M.jsonl 2>&1 | grep -vE "Loading|%\||deprecated|warn"
done
echo "SCALE DONE"
