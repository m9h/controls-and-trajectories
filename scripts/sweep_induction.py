"""Per-head induction / prev-token / ICL scores across an open model's training checkpoints.

Writes one JSONL row per (checkpoint, layer, head) plus a per-checkpoint summary row.
Resumable: skips checkpoints already present in the output file.
Purges each checkpoint from the HF cache after probing (154 ckpts would otherwise be ~58GB).
"""
import argparse, json, os, shutil, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs

p = argparse.ArgumentParser()
p.add_argument("--model", default="EleutherAI/pythia-160m")
p.add_argument("--out", required=True)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--seqlen", type=int, default=64)
p.add_argument("--seeds", type=int, default=3)      # stimulus seeds -> error bars on every score
p.add_argument("--limit", type=int, default=0)
p.add_argument("--steps", default="", help="comma-separated step numbers; default = all")
p.add_argument("--dtype", default="float32", choices=["float32","bfloat16","float16"])
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
V = tok.vocab_size


def stimulus(seed):
    """[random tokens][same tokens again] -- the induction probe."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randint(0, V, (args.batch, args.seqlen), generator=g)
    return torch.cat([h, h], 1).to(dev)


STIM = [stimulus(s) for s in range(args.seeds)]
L = args.seqlen
dest = torch.arange(L, 2 * L - 1)
src_ind = dest - L + 1          # induction: attend to token AFTER previous occurrence
src_prev = dest - 1             # previous-token head


@torch.no_grad()
def probe(rev):
    m = AutoModelForCausalLM.from_pretrained(
        args.model, revision=rev, attn_implementation="eager",
        dtype=getattr(torch, args.dtype)
    ).to(dev).eval()
    ind, prev, icl = [], [], []
    for ids in STIM:
        out = m(ids, output_attentions=True)
        NL = len(out.attentions); NH = out.attentions[0].shape[1]
        i = torch.stack([out.attentions[l][:, :, dest, src_ind].mean(dim=(0, 2)) for l in range(NL)])
        p_ = torch.stack([out.attentions[l][:, :, dest, src_prev].mean(dim=(0, 2)) for l in range(NL)])
        ind.append(i.float().cpu()); prev.append(p_.float().cpu())
        lg = out.logits[:, :-1].float(); tg = ids[:, 1:]
        lp = torch.log_softmax(lg, -1).gather(2, tg.unsqueeze(2)).squeeze(2)
        icl.append(((-lp[:, L:].mean()) - (-lp[:, :L].mean())).item())
        del out
    ind = torch.stack(ind); prev = torch.stack(prev)     # [seeds, NL, NH]
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return ind, prev, icl


def purge(model_id):
    d = os.path.expanduser("~/.cache/huggingface/hub/models--" + model_id.replace("/", "--"))
    shutil.rmtree(d, ignore_errors=True)


steps = sorted([b.name for b in list_repo_refs(args.model).branches if b.name.startswith("step")],
               key=lambda s: int(s[4:]))
if args.steps:
    want = {int(x) for x in args.steps.split(",")}
    steps = [s for s in steps if int(s[4:]) in want]
if args.limit:
    steps = steps[:args.limit]

done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        try:
            done.add(json.loads(line)["revision"])
        except Exception:
            pass

print(f"{args.model}: {len(steps)} checkpoints, {len(done)} already done", flush=True)
with open(args.out, "a") as f:
    for k, rev in enumerate(steps):
        if rev in done:
            continue
        t0 = time.time()
        try:
            ind, prev, icl = probe(rev)
        except Exception as e:
            print(f"  {rev}: FAILED {type(e).__name__}: {e}", flush=True)
            purge(args.model); continue
        S, NL, NH = ind.shape
        rec = {"revision": rev, "step": int(rev[4:]), "model": args.model,
               "dtype": args.dtype, "batch": args.batch, "seqlen": args.seqlen,
               "icl_score_mean": sum(icl) / len(icl), "icl_score_seeds": icl,
               "n_layers": NL, "n_heads": NH,
               "heads": [{"layer": l, "head": h,
                          "induction_mean": float(ind[:, l, h].mean()),
                          "induction_std": float(ind[:, l, h].std()),
                          "prev_token_mean": float(prev[:, l, h].mean())}
                         for l in range(NL) for h in range(NH)]}
        f.write(json.dumps(rec) + "\n"); f.flush()
        purge(args.model)
        print(f"  [{k+1}/{len(steps)}] {rev}: max induction {float(ind.mean(0).max()):.3f}  "
              f"ICL {rec['icl_score_mean']:+.2f}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
