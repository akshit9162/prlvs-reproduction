"""
PRLVS, recreated with full provenance.

Wang, G., Wu, X., & Yan, J. (2024). "Progressive reinforcement learning for video
summarization." Information Sciences 655, 119888.

Every constant and design decision below carries a tag:

  [PAPER §x]  / [PAPER Eq.n]   stated explicitly in the paper
  [REPRO]                      NOT in the paper. Value taken from the independent
                               reproduction (github.com/Vivek5170/Progressive-
                               Reinforcement-Learning-for-Video-Summarization),
                               which reached 63.50% F1 on TVSum vs the paper's 63.0.
  [UNRESOLVED]                 stated in neither; choice made here, alternatives named.

The paper cannot be implemented without the [REPRO] and [UNRESOLVED] entries.
That is a property of the paper, not of this file. See PROVENANCE.md for the
complete inventory and for the six places where the reference reproduction
itself departs from the published equations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------- constants
FEAT_DIM    = 1024      # [PAPER §4.3] "GoogleNet ... size of the feature vector is 1024"
HIDDEN      = 256       # [PAPER §4.3] "size of the hidden feature in the LSTM is 256"
K           = 10        # [PAPER §4.3] "size of intermediate summary K is set to 10"
TAU_S       = 1         # [PAPER §4.3] "move left/right marginally to move one step"
TAU_L       = 5         # [PAPER §4.3] "move left/right markedly to move five steps"
MOVES       = [-TAU_L, -TAU_S, TAU_S, TAU_L]   # order is [UNRESOLVED]; paper lists no order
ACTION_DIM  = 4         # [PAPER §3.1] four atomic actions
E_C         = 10        # [PAPER §4.3] "the switch iteration E_c is set to 10"
LR          = 1e-5      # [PAPER §4.3] "Adam ... learning rate set to 1e-05"
WEIGHT_DECAY= 1e-5      # [PAPER §4.3] "weight decay set to 1e-05"
BATCH_SIZE  = 1         # [PAPER §4.3] "The batch size is set to 1"
EPOCHS      = 300       # [PAPER §4.3] "maximal number of learning epochs is set to 300"
W_DIV       = 0.5       # [PAPER Eq.5]  r_v = 0.5*r_d + 0.5*r_r
W_REP       = 0.5       # [PAPER Eq.5]
DIV_GAP     = 2         # [PAPER Eq.3]  "r_v_d is set to 1 if |g(j)-g(i)| > 2"

FPS         = 2         # [REPRO] paper never states a sampling rate
GAMMA       = 0.99      # [REPRO] paper takes gamma as an input, never gives a value
MAX_STEPS   = 80        # [REPRO] paper calls T "max time step", never gives a value
EPS         = 1e-8      # [REPRO]

# [UNRESOLVED] The paper's Eq.9/10 contain no entropy term, yet unsupervised RL of
# this kind normally needs one. The reference reproduction adds entropy_coef=0.01,
# value_coef=0.5 and grad_clip=5.0, none of which appear in the paper. Set to None
# here so that the paper-exact objective is the default and any deviation is opt-in.
ENTROPY_COEF = None
VALUE_COEF   = None
GRAD_CLIP    = None


# -------------------------------------------------------------------- policies
class HorizontalPolicy(nn.Module):
    """pi_h(a_h | s)  [PAPER Eq.1] and V_h(s)  [PAPER §3.2].

    [UNRESOLVED] The paper says only that an LSTM with hidden size 256 exists. It
    does not say what the LSTM consumes, nor whether actor and critic share it.
    Here the LSTM runs over the K selected frame features in temporal order and the
    final hidden state is the state s; actor and critic are separate linear heads on
    it. The reference reproduction makes the same two choices.
    """
    def __init__(self, feat_dim=FEAT_DIM, hidden=HIDDEN, k=K):
        super().__init__()
        self.encoder = nn.LSTM(feat_dim, hidden, num_layers=1, batch_first=True)
        self.actor   = nn.Linear(hidden, k)
        self.critic  = nn.Linear(hidden, 1)

    def forward(self, summary_feats):          # (K, D)
        _, (h_n, _) = self.encoder(summary_feats.unsqueeze(0))
        ctx = h_n[-1].squeeze(0)
        return F.softmax(self.actor(ctx), dim=-1), self.critic(ctx).squeeze(-1), ctx


class VerticalPolicy(nn.Module):
    """pi_v(a_v | s, a_h)  [PAPER Eq.2] and V_v(s, a_h)  [PAPER §3.2].

    [UNRESOLVED] The paper gives no architecture. This mirrors the reference
    reproduction: mean-pool the summary, project it and the anchor frame separately,
    fuse through one ReLU layer, then separate actor and critic heads.
    """
    def __init__(self, feat_dim=FEAT_DIM, hidden=HIDDEN, action_dim=ACTION_DIM):
        super().__init__()
        self.summary_proj = nn.Linear(feat_dim, hidden)
        self.anchor_proj  = nn.Linear(feat_dim, hidden)
        self.fuse   = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, summary_feats, anchor_slot):
        s_ctx = self.summary_proj(summary_feats.mean(dim=0))
        a_ctx = self.anchor_proj(summary_feats[anchor_slot])
        z = self.fuse(torch.cat([s_ctx, a_ctx], dim=-1))
        return F.softmax(self.actor(z), dim=-1), self.critic(z).squeeze(-1)


# ----------------------------------------------------------------- environment
class PRLVSEnv:
    """[PAPER §3.1] v = {v_1..v_N}; v^k = {v^k_1..v^k_K} is the intermediate summary;
    g(j) maps the j-th key frame to its original index in v."""

    def __init__(self, feats, k=K, rng=None):
        self.x   = feats                       # (N, D), L2-normalised by the caller
        self.N   = len(feats)
        self.K   = k
        self.rng = rng or np.random.default_rng()
        self._sim = feats @ feats.T            # cached cosine matrix; math unchanged

    def reset(self):
        # [PAPER §3.2] "the agent randomly selects K frames as the initial
        # intermediate summary"
        self.g = np.sort(self.rng.choice(self.N, self.K, replace=False))
        return self.g

    def step(self, a_h, a_v):
        """[PAPER Eq.1-2] a_h picks the slot, a_v edits it.
        [PAPER Eq.15] keep-order termination.

        [UNRESOLVED] Eq.15 declares the state terminal when order breaks but does not
        say whether the order-breaking move is kept or undone. Undone here, so the
        summary a terminal state reports is always order-valid.
        """
        self.g[a_h] = int(np.clip(self.g[a_h] + MOVES[a_v], 0, self.N - 1))
        terminated = bool(np.any(np.diff(self.g) < 0))   # #(g(j) > g(j+1)) > 0
        if terminated:
            self.g = np.sort(self.g)
        return self.g, terminated

    def r_v_d(self):
        """[PAPER Eq.3] mean pairwise (1 - cosine) over selected frames, with the
        pair set to 1 when |g(j) - g(i)| > 2.

        Note this makes r_v_d == 1 whenever the K selections are mutually further
        apart than 2 frames, which is the common case. The paper states this is
        intended: "the diversity reward works in the case that two frames are
        temporally close."  It is a guard rail against collapse, not a dense signal.
        """
        sel = self.x[self.g]
        dissim = 1.0 - (sel @ sel.T)
        far = np.abs(self.g[:, None] - self.g[None, :]) > DIV_GAP
        d = np.where(far, 1.0, dissim)
        off = ~np.eye(self.K, dtype=bool)
        return float(d[off].sum() / (self.K * (self.K - 1)))

    def r_v_r(self):
        """[PAPER Eq.4] representativeness.

        [UNRESOLVED] As printed, Eq.4 reads exp(-1/K sum_i min_{j<K} ||x^k_i - x^k_j||),
        in which both indices range over the *selected* frames, so the minimum is 0
        at i=j and the term is degenerate. The paper says the reward follows Zhou et
        al. [10], whose representativeness is the mean over ALL frames of the distance
        to the nearest SELECTED frame. That reading is used here.

        [REPRO] The reference reproduction additionally uses *squared* distance and
        averages over all frames rather than 1/K. The paper writes ||.||_2, i.e. not
        squared. The paper's form is used here; set squared=True to match the repro.
        """
        d2 = np.maximum(2.0 - 2.0 * self._sim[:, self.g], 0.0)   # ||a-b||^2 for unit vectors
        return float(np.exp(-np.sqrt(d2).min(axis=1).mean()))

    def r_v(self):
        """[PAPER Eq.5]"""
        return W_DIV * self.r_v_d() + W_REP * self.r_v_r()


# --------------------------------------------------------------------- rewards
def information(p_h, log_domain=False):
    """[PAPER Eq.6] I_t = prod_{i=1..K} p(a_h_t = i | s_t).

    At K=10 with near-uniform probabilities this is ~1e-10, so r_h = I_t - I_{t-1}
    [PAPER Eq.7] has a dynamic range of ~1e-15. The reference reproduction implements
    both this literal product and a log-domain variant, noting the latter is needed
    "for stability". Neither recovers the paper's rank correlation.
    """
    log_I = torch.log(p_h.clamp_min(EPS)).sum()
    return float(log_I) if log_domain else float(log_I.clamp(min=-100.0).exp())


def discounted_returns(rewards, bootstrap_value, gamma=GAMMA):
    """[PAPER Eq.8]  R_T = r_T + gamma*V(s_T);  R_t = r_t + gamma*R_{t+1}.

    [REPRO deviation] The reference reproduction initialises the recursion at 0,
    i.e. it drops the gamma*V(s_T) bootstrap that Eq.8 specifies. Eq.8 is followed
    here; pass bootstrap_value=0.0 to match the repro.
    """
    out, R = [], bootstrap_value
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    return list(reversed(out))


def actor_loss(log_probs, returns, values):
    """[PAPER Eq.9 / Eq.10]  -1/B sum_b sum_t [ log pi * (R_t - V) ]

    Sum over t, mean over the batch. [REPRO deviation] the reference reproduction
    takes the mean over both axes.
    """
    return -(log_probs * (returns - values.detach())).sum()


def critic_loss(returns, values):
    """[PAPER Eq.11 / Eq.12]  1/B sum_b sum_t (R_t - V)^2"""
    return ((returns - values) ** 2).sum()


def balance_factor(iteration, e_c=E_C):
    """[PAPER Eq.13]  lambda = floor(E_t / E_c) mod 2.

    [UNRESOLVED] Eq.13 defines E_t as "the total number of iterations", which would
    make lambda a single constant for an entire run and make Eq.14 train only actors
    or only critics, never both. Read here as the *current* iteration index, which is
    the only reading under which Alg.1's alternation works.
    """
    return float((iteration // e_c) % 2)
