"""Quantitative per-unit tuning atlas for InceptionV1 (GoogLeNet), with a randomized-network null.

The Distill Circuits thread established early-vision features (edge/orientation detectors, curve
detectors) with feature visualizations, dataset examples and rendered tuning-curve widgets --
none of which were ever published as numbers, and none of which shipped a null. This computes
the numbers, for every unit, and ships the null with them.

Per unit (every channel of every top-level layer):
  * orientation selectivity index (OSI, 1 - circular variance) + preferred orientation, from
    sinusoidal gratings (orientation x phase x spatial frequency)
  * curvature preference + curve-vs-straight ratio, from rendered arcs (curvature x orientation)
NULL: the identical probe on N randomly-initialized copies of the same architecture, giving a
null DISTRIBUTION -- so each unit gets a percentile against chance, not just a point comparison.

    python inceptionv1_atlas.py --out atlas.jsonl --null-seeds 5
"""
import argparse, json, math
import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--null-seeds", type=int, default=5)
p.add_argument("--size", type=int, default=128)
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
N_ORI = 12
ORIS = [i * math.pi / N_ORI for i in range(N_ORI)]                    # gratings: 180-periodic
CURV_ORIS = [i * 2 * math.pi / N_ORI for i in range(N_ORI)]           # curves: 360
CURVATURES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
FREQS = [2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0]   # must span the units' preferred
#                                                     spatial frequency, or tuning is understated
LAYERS = ["conv1", "conv2", "conv3", "inception3a", "inception3b", "inception4a",
          "inception4b", "inception4c", "inception4d", "inception4e", "inception5a",
          "inception5b"]


def _norm(img):                                                       # img in [-1,1], [3,H,W]
    return (img * 0.5 + 0.5 - MEAN) / STD


def grating_stimuli(size):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size),
                            indexing="ij")
    out = []
    for th in ORIS:
        for fq in FREQS:
            for ph in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
                g = torch.sin(2 * math.pi * fq * (xs * math.cos(th) + ys * math.sin(th)) + ph)
                out.append(_norm(g.expand(3, size, size)))
    return torch.stack(out), len(FREQS) * 4                           # [N_ORI*F*P, 3, H, W]


def curve_stimuli(size, width=0.06):
    """Arcs of given curvature k (k=0 is a straight line), rotated to each orientation."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size),
                            indexing="ij")
    out = []
    for k in CURVATURES:
        for th in CURV_ORIS:
            xr = xs * math.cos(th) - ys * math.sin(th)
            yr = xs * math.sin(th) + ys * math.cos(th)
            if k == 0.0:
                d = yr.abs()
            else:
                R = 1.0 / k
                d = ((xr ** 2 + (yr - R) ** 2).sqrt() - R).abs()      # distance to the arc
            stroke = torch.exp(-(d / width) ** 2) * 2 - 1             # bright arc on dark field
            out.append(_norm(stroke.expand(3, size, size)))
    return torch.stack(out)                                           # [K*N_ORI, 3, H, W]


@torch.no_grad()
def layer_responses(model, stim, chunk=64):
    """Mean spatial response of every channel of every LAYER. -> {layer: [N_stim, C]}"""
    acts = {}
    hooks = [getattr(model, L).register_forward_hook(
        lambda m, i, o, L=L: acts.__setitem__(L, torch.relu(o).mean(dim=(2, 3)).cpu()))
        for L in LAYERS]
    outs = {L: [] for L in LAYERS}
    for i in range(0, stim.shape[0], chunk):
        model(stim[i:i + chunk].to(dev))
        for L in LAYERS:
            outs[L].append(acts[L])
    for h in hooks:
        h.remove()
    return {L: torch.cat(v) for L, v in outs.items()}


def orientation_metrics(resp, reps):
    """resp [N_ORI*F*P, C] -> OSI [C] measured AT EACH UNIT'S PREFERRED SPATIAL FREQUENCY
    (averaging tuning across frequencies blurs units tuned to one of them), preferred
    orientation [C], and mean response magnitude [C]."""
    F, P = len(FREQS), reps // len(FREQS)
    t = resp.view(N_ORI, F, P, -1).mean(2)                            # [N_ORI, F, C] avg phase
    best_f = t.mean(0).argmax(0)                                      # [C] preferred frequency
    idx = best_f.view(1, 1, -1).expand(N_ORI, 1, -1)
    tf = t.gather(1, idx).squeeze(1)                                  # [N_ORI, C] at best freq
    ang = torch.linspace(0, math.pi, N_ORI + 1)[:-1]
    vec = (tf * torch.exp(2j * ang)[:, None]).sum(0)
    osi = vec.abs() / tf.sum(0).clamp_min(1e-9)
    return osi, tf.argmax(0), tf.mean(0)


def curvature_metrics(resp):
    """resp [K*N_ORI, C] -> preferred curvature [C], normalized curve-selectivity index [C].
    Index = (curved - straight)/(curved + straight) in [-1,1]; a ratio would blow up whenever
    the straight-line response is ~0."""
    t = resp.view(len(CURVATURES), N_ORI, -1).max(1).values           # [K, C] best orientation
    pref = t.argmax(0)
    straight = t[0]                                                   # k = 0
    curved = t[1:].max(0).values
    csi = (curved - straight) / (curved + straight).clamp_min(1e-9)
    return pref, csi, t.mean(0)


def analyze(model, gr, greps, cu):
    R_g = layer_responses(model, gr)
    R_c = layer_responses(model, cu)
    out = {}
    for L in LAYERS:
        osi, pref_ori, mag = orientation_metrics(R_g[L], greps)
        pref_k, csi, cmag = curvature_metrics(R_c[L])
        out[L] = dict(osi=osi, pref_ori=pref_ori, pref_k=pref_k, csi=csi, mag=mag, cmag=cmag)
    return out


if __name__ == "__main__":
    gr, greps = grating_stimuli(args.size)
    cu = curve_stimuli(args.size)
    print(f"stimuli: {gr.shape[0]} gratings, {cu.shape[0]} curves, {args.size}px", flush=True)

    trained = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval().to(dev)
    T = analyze(trained, gr, greps, cu)
    print("trained network probed", flush=True)

    def shuffled_net(seed):
        """Stronger null: permute each conv kernel's weights within-channel. Preserves the weight
        distribution and keeps deep layers ALIVE (a randomly-initialized googlenet is dead below
        inception3a, so it is not a fair baseline there)."""
        g = torch.Generator().manual_seed(seed)
        net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval()
        with torch.no_grad():
            for mod in net.modules():
                if isinstance(mod, nn.Conv2d):
                    w = mod.weight.data
                    flat = w.view(w.shape[0], -1)
                    for i in range(flat.shape[0]):
                        flat[i] = flat[i][torch.randperm(flat.shape[1], generator=g)]
        return net.to(dev)

    nulls, shuf = [], []
    for s in range(args.null_seeds):
        torch.manual_seed(1000 + s)
        rnet = googlenet(weights=None, init_weights=True).eval().to(dev)
        nulls.append(analyze(rnet, gr, greps, cu))
        shuf.append(analyze(shuffled_net(2000 + s), gr, greps, cu))
        print(f"  null seed {s} probed (random-init + weight-shuffle)", flush=True)

    n_units = 0
    with open(args.out, "w") as f:
        for L in LAYERS:
            C = T[L]["osi"].shape[0]
            null_osi = torch.stack([n[L]["osi"] for n in nulls])       # [seeds, C]
            null_csi = torch.stack([n[L]["csi"] for n in nulls])
            null_mag = torch.stack([n[L]["mag"] for n in nulls])
            shuf_osi = torch.stack([n[L]["osi"] for n in shuf])
            shuf_mag = torch.stack([n[L]["mag"] for n in shuf])
            for c in range(C):
                osi = float(T[L]["osi"][c]); csi = float(T[L]["csi"][c])
                no = null_osi[:, c]; nr = null_csi[:, c]
                nmag = float(null_mag[:, c].mean()); tmag = float(T[L]["mag"][c])
                f.write(json.dumps({
                    "layer": L, "unit": c,
                    "osi": osi,
                    "pref_orientation_deg": int(T[L]["pref_ori"][c]) * 180 // N_ORI,
                    "pref_curvature": CURVATURES[int(T[L]["pref_k"][c])],
                    "curve_selectivity_index": csi,
                    "response_magnitude": tmag,
                    "null_response_magnitude": nmag,
                    "null_alive": bool(nmag > 1e-4),   # a DEAD null unit is not a fair baseline
                    "null_osi_mean": float(no.mean()), "null_osi_std": float(no.std()),
                    "null_osi_max": float(no.max()),
                    "osi_z_vs_null": float((osi - no.mean()) / no.std().clamp_min(1e-6)),
                    "osi_exceeds_all_nulls": bool(osi > float(no.max())),
                    "shuffle_osi_mean": float(shuf_osi[:, c].mean()),
                    "shuffle_osi_max": float(shuf_osi[:, c].max()),
                    "osi_exceeds_all_shuffles": bool(osi > float(shuf_osi[:, c].max())),
                    "shuffle_alive": bool(float(shuf_mag[:, c].mean()) > 1e-4),
                    "pref_freq_railed": None,
                    "null_csi_mean": float(nr.mean()),
                    "csi_z_vs_null": float((csi - nr.mean()) / nr.std().clamp_min(1e-6)),
                }) + "\n")
                n_units += 1
    print(f"DONE -- {n_units} units written to {args.out}", flush=True)
