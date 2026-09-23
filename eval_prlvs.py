"""
Evaluate trained PRLVS on TVSum: rank correlation (Table 3) and F-score (Table 2).

Rank correlation needs a per-frame score, but PRLVS emits a K-frame selection.
Frame scores are therefore estimated as selection frequency over R stochastic
rollouts, smoothed over a small temporal window.              # INTERPRETATION

    python eval_prlvs.py features_tvsum.npz prlvs_tvsum.pt <tvsum_root>
"""
import sys, os, csv, json
import numpy as np, torch
from scipy.stats import kendalltau, spearmanr
from scipy.ndimage import uniform_filter1d
from prlvs import PRLVSAgent, PRLVSEnv, K_DEFAULT, T_MAX

ROLLOUTS, BUDGET, SMOOTH = 30, 0.15, 5


def l2norm(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def load_anno(root):
    per = {}
    for r in csv.reader(open(os.path.join(root, "data_ex/data/ydata-tvsum50-anno.tsv")),
                        delimiter="\t"):
        per.setdefault(r[0], []).append(np.fromstring(r[2], sep=",", dtype=np.float32))
    return {v: np.stack(a) for v, a in per.items()}


@torch.no_grad()
def frame_scores(agent, feats, rng):
    """Selection frequency over stochastic rollouts -> per-sampled-frame score."""
    counts = np.zeros(len(feats), dtype=np.float32)
    for _ in range(ROLLOUTS):
        env = PRLVSEnv(feats, K=K_DEFAULT, rng=rng)
        env.reset()
        for _ in range(T_MAX):
            s = agent.encode(torch.from_numpy(env.x[env.idx]))
            p_h, _ = agent.horizontal(s)
            a_h = torch.multinomial(p_h, 1).item()
            p_v, _ = agent.vertical(s, torch.from_numpy(env.x[env.idx[a_h]]))
            a_v = torch.multinomial(p_v, 1).item()
            _, term = env.step(a_h, a_v)
            if term: break
        counts[env.idx] += 1
    return uniform_filter1d(counts / ROLLOUTS, SMOOTH)


def bin_to(scores, n):
    e = np.linspace(0, len(scores), n + 1).astype(int)
    return np.array([scores[a:b].mean() if b > a else 0.0 for a, b in zip(e[:-1], e[1:])])


def f1(pred, gold):
    tp = np.sum(pred & gold)
    if tp == 0: return 0.0
    p, r = tp / pred.sum(), tp / gold.sum()
    return 2 * p * r / (p + r)


def topk(sc, frac=BUDGET):
    k = max(1, int(round(frac * len(sc))))
    m = np.zeros(len(sc), bool); m[np.argsort(-sc)[:k]] = True
    return m


def main(npz, ckpt, root):
    d = np.load(npz, allow_pickle=True)
    anno = load_anno(root)
    agent = PRLVSAgent(); agent.load_state_dict(torch.load(ckpt, map_location="cpu")["model"]); agent.eval()
    rng = np.random.default_rng(0)

    out = {k: {"tau": [], "rho": [], "f1": []} for k in ["prlvs", "random", "human"]}
    for v in sorted(anno):
        key = f"{v}__feat"
        if key not in d.files: continue
        feats = l2norm(d[key]).astype(np.float32)
        if len(feats) <= K_DEFAULT + 5: continue
        N = len(feats)
        human = np.stack([bin_to(anno[v][j], N) for j in range(anno[v].shape[0])])
        mean_h = human.mean(0)

        sc = frame_scores(agent, feats, rng)
        t, _ = kendalltau(sc, mean_h); r, _ = spearmanr(sc, mean_h)
        out["prlvs"]["tau"].append(t); out["prlvs"]["rho"].append(r)
        golds = [topk(human[j]) for j in range(len(human))]
        out["prlvs"]["f1"].append(np.mean([f1(topk(sc), g) for g in golds]))

        rs = rng.random(N)
        t, _ = kendalltau(rs, mean_h); r, _ = spearmanr(rs, mean_h)
        out["random"]["tau"].append(t); out["random"]["rho"].append(r)
        out["random"]["f1"].append(np.mean([f1(topk(rs), g) for g in golds]))

        ts, rr, ff = [], [], []
        for a in range(len(human)):
            o = np.delete(human, a, 0).mean(0)
            tt, _ = kendalltau(human[a], o); pp, _ = spearmanr(human[a], o)
            if not np.isnan(tt): ts.append(tt); rr.append(pp)
            ff.append(np.mean([f1(golds[a], golds[b]) for b in range(len(golds)) if b != a]))
        out["human"]["tau"].append(np.mean(ts)); out["human"]["rho"].append(np.mean(rr))
        out["human"]["f1"].append(np.mean(ff))

    n = len(out["prlvs"]["tau"])
    print(f"\nTVSum, {n} videos, GoogLeNet 1024-d @2fps, K={K_DEFAULT}\n")
    print(f"{'':<34}{'tau':>9}{'rho':>9}{'F1 %':>9}")
    print("-" * 61)
    for lab, k in [("PRLVS (this reproduction)", "prlvs"), ("random", "random"),
                   ("human", "human")]:
        print(f"{lab:<34}{np.nanmean(out[k]['tau']):>9.3f}"
              f"{np.nanmean(out[k]['rho']):>9.3f}{100*np.mean(out[k]['f1']):>9.1f}")
    print(f"{'PRLVS as published (Table 2/3)':<34}{0.080:>9.3f}{0.131:>9.3f}{63.0:>9.1f}")
    print(f"{'human as published (Table 3)':<34}{0.177:>9.3f}{0.204:>9.3f}{'-':>9}")
    json.dump({k: {m: list(map(float, x)) for m, x in vv.items()} for k, vv in out.items()},
              open("prlvs_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
