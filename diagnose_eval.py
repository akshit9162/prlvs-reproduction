"""
Is tau ~ 0 a property of PRLVS, or of the way frame scores are estimated?

eval_prlvs.py turns a K-frame selector into per-frame scores via selection
frequency over R rollouts. With K=10 and R=30 that is 300 picks spread over a
median of ~400 frames, so most frames tie at 0 or 1. This script pushes several
selectors through the *same* scoring pipeline:

  oracle    samples K frames with probability proportional to the mean human
            score -- the ceiling this pipeline can reach
  uniform   samples K frames uniformly -- the floor
  untrained PRLVS agent at random init -- did training change anything?
  faithful / repaired   the trained checkpoints

and reports mean tau with a bootstrap 95% CI, plus a paired Wilcoxon test of
each trained agent against the untrained one.

    python diagnose_eval.py features_tvsum.npz <tvsum_root> [R ...]
"""
import sys, json
import numpy as np, torch
from scipy.stats import kendalltau, wilcoxon
from scipy.ndimage import uniform_filter1d
import prlvs
from prlvs import PRLVSAgent, PRLVSEnv, K_DEFAULT, T_MAX, MOVES
from eval_prlvs import l2norm, load_anno, bin_to, SMOOTH


@torch.no_grad()
def agent_counts(agent, feats, R, rng):
    counts = np.zeros(len(feats), dtype=np.float32)
    for _ in range(R):
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
    return counts


def greedy_counts(feats, R, rng, mode):
    """Hill-climb Eq.5 directly: at each step apply the single +-1/+-5 move that
    most increases r_v, stopping at a local optimum. No RL -- this is what the
    reward itself asks for."""
    prlvs.MODE = mode
    counts = np.zeros(len(feats), dtype=np.float32)
    for _ in range(R):
        env = PRLVSEnv(feats, K=K_DEFAULT, rng=rng)
        env.reset()
        cur = env.r_v()
        for _ in range(T_MAX):
            best, best_idx = cur, None
            for i in range(K_DEFAULT):
                for mv in MOVES:
                    cand = env.idx.copy()
                    cand[i] = int(np.clip(cand[i] + mv, 0, env.N - 1))
                    if np.any(np.diff(cand) <= 0): continue
                    old, env.idx = env.idx, cand
                    r = env.r_v()
                    env.idx = old
                    if r > best: best, best_idx = r, cand
            if best_idx is None: break
            env.idx, cur = best_idx, best
        counts[env.idx] += 1
    return counts


def sampler_counts(p, R, rng):
    counts = np.zeros(len(p), dtype=np.float32)
    for _ in range(R):
        counts[rng.choice(len(p), K_DEFAULT, replace=False, p=p)] += 1
    return counts


def score(counts, R):
    return uniform_filter1d(counts / R, SMOOTH)


def boot_ci(x, rng, n=10000):
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(npz, root, Rs):
    d = np.load(npz, allow_pickle=True)
    anno = load_anno(root)
    torch.manual_seed(0)
    agents = {"untrained": PRLVSAgent()}
    for m in ["faithful", "repaired"]:
        a = PRLVSAgent(); a.load_state_dict(torch.load(f"prlvs_{m}.pt", map_location="cpu")["model"])
        agents[m] = a
    for a in agents.values(): a.eval()

    res = {}
    for R in Rs:
        taus = {k: [] for k in ["oracle", "uniform", "greedy_faithful", "greedy_repaired", *agents]}
        for vi, v in enumerate(sorted(anno)):
            key = f"{v}__feat"
            if key not in d.files: continue
            feats = l2norm(d[key]).astype(np.float32)
            N = len(feats)
            if N <= K_DEFAULT + 5: continue
            mean_h = np.stack([bin_to(a, N) for a in anno[v]]).mean(0)
            # an independent stream per selector, so adding or reordering
            # selectors never changes another selector's result
            g = lambda i: np.random.default_rng([i, vi])
            p = mean_h - mean_h.min() + 1e-6; p /= p.sum()
            sc = {"oracle": score(sampler_counts(p, R, g(0)), R),
                  "uniform": score(sampler_counts(np.full(N, 1 / N), R, g(1)), R)}
            for i, m in enumerate(["faithful", "repaired"]):
                sc[f"greedy_{m}"] = score(greedy_counts(feats, R, g(2 + i), m), R)
            for i, (k, a) in enumerate(agents.items()):
                prlvs.MODE = k if k != "untrained" else "faithful"
                torch.manual_seed(100 * vi + i)
                sc[k] = score(agent_counts(a, feats, R, g(4 + i)), R)
            for k, s in sc.items():
                t, _ = kendalltau(s, mean_h)
                taus[k].append(0.0 if np.isnan(t) else float(t))
        brng = np.random.default_rng(1)
        res[R] = {k: {"per_video": v, "mean_ci": boot_ci(v, brng)} for k, v in taus.items()}
        print(f"\nR = {R} rollouts ({R * K_DEFAULT} picks/video)")
        print(f"{'selector':<18}{'tau':>8}   95% CI")
        for k, r in res[R].items():
            m, lo, hi = r["mean_ci"]
            print(f"{k:<18}{m:>8.3f}   [{lo:.3f}, {hi:.3f}]")
        for k in ["faithful", "repaired"]:
            p = wilcoxon(taus[k], taus["untrained"]).pvalue
            res[R][k]["wilcoxon_vs_untrained_p"] = float(p)
            print(f"  {k} vs untrained: Wilcoxon p = {p:.3g}")
        sys.stdout.flush()
    json.dump({str(R): r for R, r in res.items()}, open("diagnose_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]] or [30, 300])
