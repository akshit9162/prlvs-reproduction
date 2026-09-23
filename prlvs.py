"""
PRLVS — Progressive Reinforcement Learning for Video Summarization
Faithful reimplementation of Wang, Wu & Yan, Information Sciences 655 (2024) 119888.

Every component below cites the equation or section it comes from. Where the paper
is ambiguous the reading taken is flagged with  # INTERPRETATION.

Spec (§4.3): GoogLeNet 1024-d features, LSTM hidden 256, K=10, move markedly=5 /
marginally=1, E_c=10, Adam lr=1e-5 weight_decay=1e-5, batch size 1, 300 epochs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM   = 1024      # §4.3 GoogLeNet
HIDDEN     = 256       # §4.3 LSTM hidden size
K_DEFAULT  = 10        # §4.3 size of intermediate summary
MOVES      = [-1, 1, -5, 5]   # §4.3 marginally=1, markedly=5
GAMMA      = 0.99      # not stated in the paper; standard A2C default  # INTERPRETATION
T_MAX      = 40        # max time step; not stated in the paper          # INTERPRETATION
DIV_TAU    = 2         # Eq.3: r^v_d set to 1 if |g(j)-g(i)| > 2

# --- Two readings of the paper -----------------------------------------------
# "faithful": equations exactly as printed. Two terms then turn out degenerate:
#   * Eq.3 masks a pair to 1 when |g(j)-g(i)| > 2. Reading g() as the FRAME index
#     makes essentially every pair masked for realistic K and N, so r^v_d == 1.0
#     and carries no gradient.
#   * Eq.6 I_t = prod of K softmax probabilities ~ 1e-11 at K=10, so
#     r^h = I_t - I_{t-1} ~ 1e-15: below any usable dynamic range.
# "repaired": the charitable reading -- g() is the frame's POSITION in the summary
#   (1..K), and the information reward is taken in log space, which preserves the
#   ordering Eq.7 intends while restoring dynamic range.
MODE = "faithful"


# ----------------------------------------------------------------------------- model
class PRLVSAgent(nn.Module):
    """A2C with two separate policies and two separate value functions (§3.2).

    State s_t is the LSTM encoding of the K currently-selected frame features in
    temporal order.                                                # INTERPRETATION
    """
    def __init__(self, feat_dim=FEAT_DIM, hidden=HIDDEN, K=K_DEFAULT):
        super().__init__()
        self.K = K
        self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
        self.pi_h = nn.Linear(hidden, K)                 # pi^h(a^h | s)      Eq.1
        self.v_h  = nn.Linear(hidden, 1)                 # V^h(s)             Eq.9/11
        self.pi_v = nn.Linear(hidden + feat_dim, 4)      # pi^v(a^v | s, a^h) Eq.2
        self.v_v  = nn.Linear(hidden + feat_dim, 1)      # V^v(s, a^h)        Eq.10/12

    def encode(self, sel_feats):                         # (K, D) -> (hidden,)
        out, _ = self.lstm(sel_feats.unsqueeze(0))
        return out[0, -1]

    def horizontal(self, s):
        return F.softmax(self.pi_h(s), dim=-1), self.v_h(s).squeeze(-1)

    def vertical(self, s, anchor_feat):
        x = torch.cat([s, anchor_feat], dim=-1)
        return F.softmax(self.pi_v(x), dim=-1), self.v_v(x).squeeze(-1)


# ------------------------------------------------------------------------ environment
class PRLVSEnv:
    """Intermediate summary of K frame indices, edited by ±1 / ±5 (§3.1)."""

    def __init__(self, feats, K=K_DEFAULT, rng=None):
        self.x = feats                       # (N, D) L2-normalised outside
        self.N = len(feats)
        self.K = K
        self.rng = rng or np.random.default_rng()
        # Eq.4 needs, at every step, the distance from every frame to its nearest
        # selected frame. Features are L2-normalised, so ||a-b|| = sqrt(2-2*cos);
        # precomputing the full cosine matrix once per video turns an O(N*K*D)
        # recomputation per step into an O(N*K) lookup.
        self._sim = feats @ feats.T

    def reset(self):
        # §3.2 "the agent randomly selects K frames as the initial intermediate summary"
        self.idx = np.sort(self.rng.choice(self.N, self.K, replace=False))
        return self.idx

    def step(self, a_h, a_v):
        """Returns (idx, terminated). Termination is Eq.15 'keep-order'."""
        before = self.idx.copy()
        self.idx[a_h] = int(np.clip(self.idx[a_h] + MOVES[a_v], 0, self.N - 1))
        # Eq.15: terminate if the temporal order of selected frames was broken
        terminated = bool(np.any(np.diff(self.idx) < 0))
        if terminated:
            self.idx = before          # do not keep an out-of-order state
        return self.idx, terminated

    # ---- Eq.3  diversity
    def r_v_d(self):
        sel = self.x[self.idx]
        sim = sel @ sel.T                                   # cosine (features L2-normed)
        d = 1.0 - sim
        # "r^v_d is set to 1 if |g(j) - g(i)| > 2"
        if MODE == "faithful":
            far = np.abs(self.idx[:, None] - self.idx[None, :]) > DIV_TAU
        else:
            pos = np.arange(self.K)
            far = np.abs(pos[:, None] - pos[None, :]) > DIV_TAU
        d = np.where(far, 1.0, d)
        K = self.K
        iu = ~np.eye(K, dtype=bool)
        return float(d[iu].sum() / (K * (K - 1)))

    # ---- Eq.4  representativeness
    def r_v_r(self):
        # Paper writes min_{j<K} ||x^k_i - x^k_j||, which is 0 for i=j. Read as the
        # diversity-representativeness reward of Zhou et al. [10] that the paper says
        # it follows: mean over ALL frames of distance to the nearest SELECTED frame.
        #                                                             # INTERPRETATION
        d = np.sqrt(np.maximum(2.0 - 2.0 * self._sim[:, self.idx], 0.0))
        return float(np.exp(-d.min(axis=1).mean()))

    # ---- Eq.5
    def r_v(self):
        return 0.5 * self.r_v_d() + 0.5 * self.r_v_r()


# ---- Eq.6 / Eq.7  horizontal reward
def information(p_h):
    """Eq.6: I_t = prod_i p(a^h_t = i | s_t).

    faithful -> the raw product (~1e-11 at K=10; r^h is then numerical noise)
    repaired -> the same quantity in log space, which keeps Eq.7's ordering but
                gives r^h a usable scale."""
    logI = torch.log(p_h.clamp_min(1e-12)).sum()
    return float(logI.exp()) if MODE == "faithful" else float(logI)


def returns_from(rewards, bootstrap, gamma=GAMMA):
    """Eq.8: R_t = r_t + gamma*R_{t+1}, with R_T = r_T + gamma*V(s_T)."""
    out, R = [], bootstrap
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    return list(reversed(out))
