"""
The paper's F-score, under the standard benchmark protocol (Zhang et al. 2016; Zhou et al. 2018),
so it can be compared with the published 63.0% on TVSum.

eval_prlvs.py's F1 is a frame-level top-15% overlap. The benchmark F1 is different:
  1. Shots: Kernel Temporal Segmentation (Potapov et al. 2014) on the frame features,
     linear kernel, penalty-selected number of change points (cpd_auto, vmax = 1).
  2. Machine summary: mean frame score per shot, then a 0/1 knapsack picks whole shots
     up to 15% of the video's length.
  3. User summaries: each annotator's scores go through the same shots and knapsack.
  4. F1 of the machine summary against each user summary, reduced by mean ("avg", the
     usual TVSum convention) or by max ("max", as the reference reproduction reports).

Everything runs at the 2 fps feature resolution, for every selector alike.

    python std_f1.py features_tvsum.npz <tvsum_root>
"""
import sys, json
import numpy as np, torch
import prlvs
from prlvs import PRLVSAgent
from eval_prlvs import l2norm, load_anno, bin_to, frame_scores
from train_importance import importance

BUDGET = 0.15


# ---- Kernel Temporal Segmentation (Potapov, Douze, Harchaoui & Schmid, ECCV 2014) ----

def _segment_costs(K):
    """C[i, j] = within-segment scatter of frames [i, j), for all i < j."""
    n = len(K)
    S = np.zeros((n + 1, n + 1)); S[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)
    d = np.concatenate([[0.0], np.cumsum(np.diag(K))])
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    block = S[j, j] - S[i, j] - S[j, i] + S[i, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        C = (d[j] - d[i]) - block / (j - i)
    C[j <= i] = np.inf
    return C


def _dp(C, m):
    """Best cost of splitting [0, n) into k+1 segments, k = 0..m, and the argmins."""
    n = len(C) - 1
    I = np.full((m + 1, n + 1), np.inf); P = np.zeros((m + 1, n + 1), int)
    I[0] = C[0]
    for k in range(1, m + 1):
        tot = I[k - 1][:, None] + C                        # split at s, next segment [s, t)
        P[k] = np.argmin(tot, 0); I[k] = tot[P[k], np.arange(n + 1)]
    return I, P


def kts(X, max_ncp):
    K = X @ X.T
    n = len(K)
    C = _segment_costs(K)
    m = min(max_ncp, n - 1)
    I, P = _dp(C, m)
    scores = I[:, n]
    ncp = np.arange(1, m + 1)
    pen = np.zeros(m + 1); pen[1:] = ncp / (2.0 * n) * (np.log(n / ncp) + 1)   # vmax = 1
    k = int(np.argmin(scores / n + pen))
    cps, t = [], n
    for kk in range(k, 0, -1):
        t = P[kk, t]; cps.append(t)
    b = [0] + sorted(cps) + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]   # [start, end) shots


# ---- summaries and F1 ----

def knapsack(values, weights, cap):
    n = len(values)
    V = np.zeros((n + 1, cap + 1))
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        V[i] = V[i - 1]
        if w <= cap:
            V[i, w:] = np.maximum(V[i - 1, w:], V[i - 1, :cap + 1 - w] + v)
    pick, c = [], cap
    for i in range(n, 0, -1):
        if V[i, c] != V[i - 1, c]:
            pick.append(i - 1); c -= weights[i - 1]
    return pick


def summary(scores, shots):
    n = shots[-1][1]
    vals = [float(scores[a:b].mean()) for a, b in shots]
    lens = [b - a for a, b in shots]
    m = np.zeros(n, bool)
    for s in knapsack(vals, lens, int(np.floor(BUDGET * n))):
        a, b = shots[s]; m[a:b] = True
    return m


def f1(pred, gold):
    tp = np.sum(pred & gold)
    if tp == 0: return 0.0
    p, r = tp / pred.sum(), tp / gold.sum()
    return 2 * p * r / (p + r)


def main(npz, root):
    d = np.load(npz, allow_pickle=True)
    anno = load_anno(root)
    agents = {}
    for mode in ["faithful", "repaired"]:
        a = PRLVSAgent(); a.load_state_dict(torch.load(f"prlvs_{mode}.pt", map_location="cpu")["model"]); a.eval()
        agents[mode] = a
    folds = json.load(open("importance_cv.json"))["folds"]
    fold_of = {v: f for f, vs in enumerate(folds) for v in vs}
    pred = np.load("importance_cv_pred.npz")
    rng = np.random.default_rng(0)

    names = ["prlvs_faithful", "prlvs_repaired", "random", "importance_model", "human_loo"]
    out = {k: {"avg": [], "max": []} for k in names}
    shots_per_video = []
    for v in sorted(anno):
        key = f"{v}__feat"
        if key not in d.files: continue
        X = l2norm(d[key]).astype(np.float64)
        n = len(X)
        shots = kts(X, max_ncp=n - 1)
        shots_per_video.append(len(shots))
        users = [summary(bin_to(u, n), shots) for u in anno[v]]
        sc = {}
        for mode, agent in agents.items():
            prlvs.MODE = mode
            sc[f"prlvs_{mode}"] = frame_scores(agent, X.astype(np.float32), rng)
        sc["random"] = rng.random(n)
        if v in fold_of:
            sc["importance_model"] = importance(fold_of[v], v, pred)
        for k, s in sc.items():
            fs = [f1(summary(s, shots), u) for u in users]
            out[k]["avg"].append(float(np.mean(fs))); out[k]["max"].append(float(np.max(fs)))
        loo = [[f1(users[a], users[b]) for b in range(len(users)) if b != a] for a in range(len(users))]
        out["human_loo"]["avg"].append(float(np.mean([np.mean(x) for x in loo])))
        out["human_loo"]["max"].append(float(np.mean([np.max(x) for x in loo])))
        print(f"{v}: {len(shots)} shots, " + ", ".join(f"{k} {100 * out[k]['avg'][-1]:.1f}" for k in sc), flush=True)

    print(f"\nTVSum, {len(shots_per_video)} videos, KTS shots per video: median {int(np.median(shots_per_video))}, "
          f"budget {BUDGET:.0%}\n")
    print(f"{'':<22}{'F1 avg %':>10}{'F1 max %':>10}")
    for k in names:
        print(f"{k:<22}{100 * np.mean(out[k]['avg']):>10.1f}{100 * np.mean(out[k]['max']):>10.1f}")
    print(f"{'PRLVS as published':<22}{63.0:>10.1f}{'':>10}")
    json.dump({"shots_per_video": shots_per_video, **out}, open("std_f1.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
