"""
PRLVS + importance: the fix suggested by diagnose_eval.py.

The paper's vertical reward (Eq. 5, diversity + representativeness) carries no
information about what humans find important. This adds one term:

    r_v' = 0.5 * r_v (Eq. 5, as printed)  +  0.5 * mean importance of the K selected frames

where importance comes from importance_cv.py's regressor for this fold, which
never saw this fold's test videos. Everything else -- agent, Eq. 6/7 horizontal
reward, Alg. 1 alternation, optimiser, 300 epochs -- is unchanged, and the
equations are used as printed (faithful mode), so the importance term is the only
difference from the baseline.

Training and evaluation both follow the 5 folds in importance_cv.json: each fold's
agent trains on 40 videos and is scored only on its 10 held-out videos.

    python train_importance.py features_tvsum.npz <tvsum_root> <fold> [epochs]   # train one fold
    python train_importance.py features_tvsum.npz <tvsum_root> eval [R]          # evaluate all folds
"""
import sys, json, time
import numpy as np, torch
from scipy.stats import kendalltau
import prlvs
from prlvs import PRLVSAgent, PRLVSEnv, K_DEFAULT, GAMMA, returns_from
from train_prlvs import load, episode, LR, WD, E_C
from eval_prlvs import load_anno, bin_to
from diagnose_eval import agent_counts, greedy_counts, sampler_counts, score, boot_ci

W_IMP = 0.5


def importance(fold, v, pred):
    z = pred[f"{fold}__{v}"]
    z = (z - z.mean()) / (z.std() + 1e-8)
    return 1.0 / (1.0 + np.exp(-z))                 # (0, 1), per-video normalised


class ImpEnv(PRLVSEnv):
    def __init__(self, feats, imp, **kw):
        super().__init__(feats, **kw)
        self.imp = imp

    def r_v(self):
        return (1 - W_IMP) * super().r_v() + W_IMP * float(self.imp[self.idx].mean())


def train(npz, fold, epochs):
    torch.set_num_threads(1)
    prlvs.MODE = "faithful"
    feats = load(npz)
    folds = json.load(open("importance_cv.json"))["folds"]
    pred = np.load("importance_cv_pred.npz")
    vids = sorted(v for v in feats if v not in folds[fold])
    imp = {v: importance(fold, v, pred) for v in vids}
    rng = np.random.default_rng(fold); torch.manual_seed(fold)
    agent = PRLVSAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=LR, weight_decay=WD)
    batch, hist, t0 = 0, [], time.time()
    print(f"fold {fold}: training on {len(vids)} videos, {epochs} epochs, w_imp={W_IMP}", flush=True)
    for ep in range(epochs):
        rng.shuffle(vids)
        ep_rv = []
        for v in vids:
            lam = float((batch // E_C) % 2)                                   # Eq. 13
            env = ImpEnv(feats[v], imp[v], K=K_DEFAULT, rng=rng)
            lph, lpv, rh, rv, Vh, Vv = episode(agent, env, None, rng)
            ep_rv.append(np.mean(rv))
            Vh_t, Vv_t = torch.stack(Vh), torch.stack(Vv)
            Rh = torch.tensor(returns_from(rh, Vh_t[-1].item(), GAMMA), dtype=torch.float32)
            Rv = torch.tensor(returns_from(rv, Vv_t[-1].item(), GAMMA), dtype=torch.float32)
            actor = -(torch.stack(lph) * (Rh - Vh_t.detach())).sum() - (torch.stack(lpv) * (Rv - Vv_t.detach())).sum()
            critic = ((Rh - Vh_t) ** 2).sum() + ((Rv - Vv_t) ** 2).sum()
            loss = lam * actor + (1 - lam) * critic                           # Eq. 14
            opt.zero_grad(); loss.backward(); opt.step()
            batch += 1
        hist.append(float(np.mean(ep_rv)))
        if (ep + 1) % 20 == 0:
            print(f"  fold {fold} epoch {ep+1}/{epochs}  mean r_v' = {hist[-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": agent.state_dict(), "reward_hist": hist, "fold": fold, "w_imp": W_IMP}, f"prlvs_imp_fold{fold}.pt")
    print(f"fold {fold}: saved prlvs_imp_fold{fold}.pt", flush=True)


def evaluate(npz, root, R):
    prlvs.MODE = "faithful"
    feats = load(npz)
    anno = load_anno(root)
    folds = json.load(open("importance_cv.json"))["folds"]
    pred = np.load("importance_cv_pred.npz")
    base = PRLVSAgent(); base.load_state_dict(torch.load("prlvs_faithful.pt", map_location="cpu")["model"]); base.eval()
    taus = {k: [] for k in ["oracle", "uniform", "importance_only", "greedy_paper", "greedy_imp",
                            "prlvs_paper", "prlvs_imp"]}
    for f, test in enumerate(folds):
        agent = PRLVSAgent(); agent.load_state_dict(torch.load(f"prlvs_imp_fold{f}.pt", map_location="cpu")["model"]); agent.eval()
        for v in test:
            vi = sorted(feats).index(v)
            x = feats[v]; N = len(x)
            h = np.stack([bin_to(a, N) for a in anno[v]]).mean(0)
            imp = importance(f, v, pred)
            g = lambda i: np.random.default_rng([i, vi])
            ph = h - h.min() + 1e-6; ph /= ph.sum()
            sc = {"oracle": score(sampler_counts(ph, R, g(0)), R),
                  "uniform": score(sampler_counts(np.full(N, 1 / N), R, g(1)), R),
                  "importance_only": imp}
            sc["greedy_paper"] = score(greedy_counts(x, R, g(2), "faithful"), R)
            # reward-greedy on the augmented reward: swap in ImpEnv
            orig = prlvs.PRLVSEnv
            import diagnose_eval as de
            de.PRLVSEnv = lambda feats, **kw: ImpEnv(feats, imp, **kw)
            sc["greedy_imp"] = score(greedy_counts(x, R, g(3), "faithful"), R)
            de.PRLVSEnv = orig
            torch.manual_seed(100 * vi + 1); sc["prlvs_paper"] = score(agent_counts(base, x, R, g(4)), R)
            torch.manual_seed(100 * vi + 2); sc["prlvs_imp"] = score(agent_counts(agent, x, R, g(5)), R)
            for k, s in sc.items():
                t = kendalltau(s, h)[0]
                taus[k].append(0.0 if np.isnan(t) else float(t))
        print(f"fold {f} done", flush=True)
    from scipy.stats import wilcoxon
    brng = np.random.default_rng(1)
    out = {k: {"per_video": v, "mean_ci": boot_ci(v, brng)} for k, v in taus.items()}
    out["prlvs_imp"]["wilcoxon_vs_prlvs_paper_p"] = float(wilcoxon(taus["prlvs_imp"], taus["prlvs_paper"]).pvalue)
    out["prlvs_imp"]["wilcoxon_vs_uniform_p"] = float(wilcoxon(taus["prlvs_imp"], taus["uniform"]).pvalue)
    print(f"\n5-fold held-out, R = {R}\n{'selector':<18}{'tau':>8}   95% CI")
    for k, r in out.items():
        m, lo, hi = r["mean_ci"]; print(f"{k:<18}{m:>8.3f}   [{lo:.3f}, {hi:.3f}]")
    print(f"  prlvs_imp vs prlvs_paper: Wilcoxon p = {out['prlvs_imp']['wilcoxon_vs_prlvs_paper_p']:.2g}")
    print(f"  prlvs_imp vs uniform:     Wilcoxon p = {out['prlvs_imp']['wilcoxon_vs_uniform_p']:.2g}")
    json.dump({"R": R, **out}, open("importance_eval.json", "w"), indent=2)


if __name__ == "__main__":
    if sys.argv[3] == "eval":
        evaluate(sys.argv[1], sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 150)
    else:
        train(sys.argv[1], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 300)
