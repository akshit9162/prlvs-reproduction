"""
Feasibility check for an importance-aware reward.

The diagnosis showed PRLVS's reward carries no information about what humans find
important. This trains a small frame-importance regressor on GoogLeNet features
with 5-fold cross-validation over videos (the standard TVSum protocol), so every
prediction is for a video the model never saw, and reports Kendall tau of the
held-out predictions against the mean human score.

    python importance_cv.py features_tvsum.npz <tvsum_root>
"""
import sys, json
import numpy as np, torch, torch.nn as nn
from scipy.stats import kendalltau
from eval_prlvs import l2norm, load_anno, bin_to
from prlvs import K_DEFAULT

FOLDS, EPOCHS, SEED = 5, 60, 0


def mlp(d):
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))


def zscore(y):
    return (y - y.mean()) / (y.std() + 1e-8)


def main(npz, root):
    d = np.load(npz, allow_pickle=True)
    anno = load_anno(root)
    vids, X, Y = [], {}, {}
    for v in sorted(anno):
        k = f"{v}__feat"
        if k not in d.files or d[k].shape[0] <= K_DEFAULT + 5: continue
        X[v] = l2norm(d[k]).astype(np.float32)
        Y[v] = np.stack([bin_to(a, len(X[v])) for a in anno[v]]).mean(0).astype(np.float32)
        vids.append(v)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(vids)
    folds = [list(order[i::FOLDS]) for i in range(FOLDS)]

    pred, taus, per_fold = {}, {}, {}
    for f, test in enumerate(folds):
        train = [v for v in vids if v not in test]
        xs = torch.from_numpy(np.concatenate([X[v] for v in train]))
        ys = torch.from_numpy(np.concatenate([zscore(Y[v]) for v in train]))   # per-video normalised
        torch.manual_seed(SEED + f)
        net = mlp(xs.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(xs))
            for i in range(0, len(xs), 256):
                b = perm[i:i + 256]
                loss = ((net(xs[b]).squeeze(-1) - ys[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            # every video scored by this fold's model: train videos feed the reward
            # during RL training, test videos only ever see a model that never saw them
            for v in vids:
                per_fold[f"{f}__{v}"] = net(torch.from_numpy(X[v])).squeeze(-1).numpy()
            for v in test:
                pred[v] = per_fold[f"{f}__{v}"]
                taus[v] = float(kendalltau(pred[v], Y[v])[0])
        print(f"fold {f}: tau = {np.mean([taus[v] for v in test]):.3f} on {len(test)} held-out videos", flush=True)

    t = np.array([taus[v] for v in vids])
    b = np.random.default_rng(1).choice(t, (10000, len(t))).mean(1)
    print(f"\nheld-out importance regressor: tau = {t.mean():.3f} "
          f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
    np.savez("importance_cv_pred.npz", **per_fold)
    json.dump({"folds": folds, "tau": taus}, open("importance_cv.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
