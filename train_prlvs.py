"""
PRLVS training — Algorithm 1 of Wang et al. (2024).

Alternating A2C: every E_c batches the trained policy flips and the other is frozen
(Alg.1 lines 5-9). Loss combines actor and critic terms via the balance factor
lambda = floor(E_t/E_c) mod 2  (Eq.13, Eq.14).

    python train_prlvs.py features_tvsum.npz [epochs]
"""
import sys, time, json
import numpy as np, torch
import prlvs
from prlvs import (PRLVSAgent, PRLVSEnv, information, returns_from,
                   K_DEFAULT, T_MAX, GAMMA)

LR, WD, EPOCHS, E_C = 1e-5, 1e-5, 300, 10      # §4.3


def l2norm(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def load(npz):
    d = np.load(npz, allow_pickle=True)
    # TVSum ids can themselves contain "__" (e.g. Hl-__g2gn_A), so split from
    # the right on the final separator only.
    vids = sorted({k.rsplit("__", 1)[0] for k in d.files})
    return {v: l2norm(d[f"{v}__feat"]).astype(np.float32) for v in vids
            if d[f"{v}__feat"].shape[0] > K_DEFAULT + 5}


def episode(agent, env, train_policy, rng):
    """One trajectory (Alg.1 lines 10-18). train_policy in {'h','v'}."""
    env.reset()
    logp_h, logp_v, r_h, r_v, V_h, V_v = [], [], [], [], [], []
    I_prev = None
    for t in range(T_MAX):
        s = agent.encode(torch.from_numpy(env.x[env.idx]))
        p_h, v_h = agent.horizontal(s)
        a_h = torch.multinomial(p_h, 1).item()

        I_t = information(p_h)                                    # Eq.6
        r_h.append(0.0 if I_prev is None else I_t - I_prev)       # Eq.7
        I_prev = I_t
        logp_h.append(torch.log(p_h[a_h].clamp_min(1e-12))); V_h.append(v_h)

        anchor = torch.from_numpy(env.x[env.idx[a_h]])
        p_v, v_v = agent.vertical(s, anchor)
        a_v = torch.multinomial(p_v, 1).item()
        logp_v.append(torch.log(p_v[a_v].clamp_min(1e-12))); V_v.append(v_v)

        _, terminated = env.step(a_h, a_v)                        # Eq.15
        r_v.append(env.r_v())                                     # Eq.5
        if terminated:
            break
    return logp_h, logp_v, r_h, r_v, V_h, V_v


def main(npz, epochs=EPOCHS, mode="faithful"):
    prlvs.MODE = mode
    feats = load(npz)
    vids = sorted(feats)
    print(f"mode={mode}; {len(vids)} videos; K={K_DEFAULT}, T={T_MAX}, "
          f"E_c={E_C}, lr={LR}, epochs={epochs}")
    rng = np.random.default_rng(0); torch.manual_seed(0)
    agent = PRLVSAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=LR, weight_decay=WD)

    batch, hist, t0 = 0, [], time.time()
    for ep in range(epochs):
        rng.shuffle(vids)
        ep_rv = []
        for v in vids:
            # Alg.1 lines 5-9: flip trained policy every E_c batches
            policy = "h" if (batch // E_C) % 2 == 0 else "v"
            lam = float((batch // E_C) % 2)                       # Eq.13
            env = PRLVSEnv(feats[v], K=K_DEFAULT, rng=rng)
            lph, lpv, rh, rv, Vh, Vv = episode(agent, env, policy, rng)
            ep_rv.append(np.mean(rv))

            Vh_t, Vv_t = torch.stack(Vh), torch.stack(Vv)
            Rh = torch.tensor(returns_from(rh, Vh_t[-1].item(), GAMMA), dtype=torch.float32)
            Rv = torch.tensor(returns_from(rv, Vv_t[-1].item(), GAMMA), dtype=torch.float32)

            actor_h  = -(torch.stack(lph) * (Rh - Vh_t.detach())).sum()   # Eq.9
            actor_v  = -(torch.stack(lpv) * (Rv - Vv_t.detach())).sum()   # Eq.10
            critic_h = ((Rh - Vh_t) ** 2).sum()                           # Eq.11
            critic_v = ((Rv - Vv_t) ** 2).sum()                           # Eq.12
            loss = lam * (actor_h + actor_v) + (1 - lam) * (critic_h + critic_v)  # Eq.14

            opt.zero_grad(); loss.backward(); opt.step()
            batch += 1

        hist.append(float(np.mean(ep_rv)))
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3}/{epochs}  mean r_v = {hist[-1]:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": agent.state_dict(), "reward_hist": hist, "mode": mode}, f"prlvs_{mode}.pt")
    json.dump(hist, open(f"train_hist_{mode}.json", "w"))
    print(f"saved prlvs_{mode}.pt")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS,
         sys.argv[3] if len(sys.argv) > 3 else "faithful")
