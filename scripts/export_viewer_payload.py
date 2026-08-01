"""Compact JSON payload for the atlas viewer: scalars from the JSONL + each unit's tuning
curve, normalized and quantized to one base36 char per orientation (~70KB for 5808 units)."""
import json, math, sys
import torch
from torchvision.models import googlenet, GoogLeNet_Weights

sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("atlas", "scripts/inceptionv1_atlas.py")
src = open("scripts/inceptionv1_atlas.py").read().split('if __name__')[0]
src = src.replace('p.add_argument("--out", required=True)', 'p.add_argument("--out", default="x")')
G = {}
exec(compile(src, "atlas", "exec"), G)

LAYERS, N_ORI, FREQS, PHASES = G["LAYERS"], G["N_ORI"], G["FREQS"], 4
dev = G["dev"]
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"

stim = G["grating_stimuli"](128)[0]
net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval().to(dev)
R = G["layer_responses"](net, stim)

curves = {}
for L in LAYERS:
    t = R[L].view(N_ORI, len(FREQS), PHASES, -1).mean(2)          # [ORI, F, C]
    best = t.mean(0).argmax(0)
    tf = t.gather(1, best.view(1, 1, -1).expand(N_ORI, 1, -1)).squeeze(1)   # [ORI, C]
    norm = tf / tf.max(0).values.clamp_min(1e-9)
    q = (norm * 35).round().clamp(0, 35).to(torch.int64)
    curves[L] = ["".join(B36[int(q[o, c])] for o in range(N_ORI)) for c in range(q.shape[1])]
    print(f"  {L}: {len(curves[L])} curves", flush=True)

rows = [json.loads(l) for l in open("data/inceptionv1_atlas.jsonl")]
by = {}
for r in rows:
    by.setdefault(r["layer"], []).append(r)

payload = {"n_ori": N_ORI, "freqs": FREQS, "layers": []}
for L in LAYERS:
    us = []
    for r in sorted(by[L], key=lambda r: r["unit"]):
        us.append([round(r["osi"], 3), r["pref_orientation_deg"], r["pref_curvature"],
                   round(r["curve_selectivity_index"], 3), round(r["null_osi_max"], 3),
                   round(r["shuffle_osi_max"], 3), int(r["null_alive"]),
                   int(r["shuffle_alive"]), curves[L][r["unit"]]])
    payload["layers"].append({"name": L, "units": us})

json.dump(payload, open("data/viewer_payload.json", "w"), separators=(",", ":"))
import os
print("payload KB:", round(os.path.getsize("data/viewer_payload.json") / 1024))
