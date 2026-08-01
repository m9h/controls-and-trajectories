"""Causal control for the induction atlas: ablate the top-k induction heads vs k RANDOM heads
and measure the damage to in-context (2nd-copy) loss. An induction score is correlational;
this is what makes it a mechanism."""
import json, os, shutil, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
REVS = sys.argv[1:] or ["step512", "step1000", "step2000", "step16000", "step143000"]
K = 5
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
g = torch.Generator().manual_seed(0)
half = torch.randint(0, tok.vocab_size, (16, 64), generator=g)
ids = torch.cat([half, half], 1).to(dev)
L = 64


def install(model, heads):
    """Zero each (layer,head)'s slice of the attention output before the dense projection."""
    cfg = model.config
    dh = cfg.hidden_size // cfg.num_attention_heads
    by = {}
    for (l, h) in heads:
        by.setdefault(l, []).append(h)
    hs = []
    for l, hd in by.items():
        dense = model.gpt_neox.layers[l].attention.dense
        def pre(mod, args, hd=hd):
            x = args[0].clone()
            for h in hd:
                x[..., h * dh:(h + 1) * dh] = 0
            return (x,) + tuple(args[1:])
        hs.append(dense.register_forward_pre_hook(pre))
    return hs


@torch.no_grad()
def second_copy_loss(model):
    lg = model(ids).logits[:, :-1].float()
    lp = torch.log_softmax(lg, -1).gather(2, ids[:, 1:].unsqueeze(2)).squeeze(2)
    return float(-lp[:, L:].mean())


scores = {r["revision"]: r for r in
          (json.loads(l) for l in open("data/induction_pythia-160m.jsonl"))}
out = []
for rev in REVS:
    m = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev, attn_implementation="eager",
                                             dtype=torch.float32).to(dev).eval()
    base = second_copy_loss(m)
    hs = sorted(scores[rev]["heads"], key=lambda h: -h["induction_mean"])[:K]
    top = [(h["layer"], h["head"]) for h in hs]
    gg = torch.Generator().manual_seed(1)
    NL, NH = scores[rev]["n_layers"], scores[rev]["n_heads"]
    rnd = [(int(torch.randint(0, NL, (1,), generator=gg)),
            int(torch.randint(0, NH, (1,), generator=gg))) for _ in range(K)]

    hk = install(m, top); abl_i = second_copy_loss(m); [h.remove() for h in hk]
    hk = install(m, rnd); abl_r = second_copy_loss(m); [h.remove() for h in hk]
    rec = {"revision": rev, "step": scores[rev]["step"], "baseline_2nd_copy_loss": base,
           "ablate_induction": abl_i, "ablate_random": abl_r,
           "delta_induction": abl_i - base, "delta_random": abl_r - base,
           "top_heads": top, "random_heads": rnd}
    out.append(rec)
    print(f"{rev:>12} base {base:6.3f} | ablate induction {abl_i:6.3f} ({abl_i-base:+.3f}) "
          f"| random {abl_r:6.3f} ({abl_r-base:+.3f})", flush=True)
    del m
    if dev == "cuda": torch.cuda.empty_cache()
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--pythia-160m"),
                  ignore_errors=True)

with open("data/ablation_pythia-160m.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("written")
