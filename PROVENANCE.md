# PRLVS — provenance of every implementation decision

Wang, Wu & Yan, *Progressive reinforcement learning for video summarization*,
Information Sciences 655 (2024) 119888.

This file exists because **the paper cannot be implemented without decisions it does
not specify.** The goal here is not to eliminate those decisions — that is impossible —
but to ensure none of them is silent.

Cross-reference: the independent reproduction at
`github.com/Vivek5170/Progressive-Reinforcement-Learning-for-Video-Summarization`
reached **63.50% F1 on TVSum** against the paper's 63.0%, so its choices are treated
as evidence about what the authors plausibly did.

---

## 1. Specified by the paper

| Item | Value | Source |
|---|---|---|
| Feature extractor / dim | GoogLeNet, 1024-d | §4.3 |
| LSTM hidden size | 256 | §4.3 |
| Summary size K | 10 | §4.3 |
| Move "marginally" / "markedly" | 1 / 5 frames | §4.3 |
| Vertical action count | 4 | §3.1 |
| Switch iteration E_c | 10 | §4.3 |
| Optimiser | Adam, lr 1e-5, weight decay 1e-5 | §4.3 |
| Batch size | 1 | §4.3 |
| Epochs | 300 | §4.3 |
| Reward weights | 0.5 diversity + 0.5 representativeness | Eq.5 |
| Diversity temporal gap | 2 | Eq.3 |
| Optimisation | A2C, two policies, two value functions, alternating | §3.2, Alg.1 |
| Termination | keep-order | Eq.15 |
| Datasets | SumMe (25), TVSum (50) | §4.1 |
| Metrics | F-score; Kendall tau, Spearman rho | §4.2 |

## 2. Not in the paper — taken from the reference reproduction

| Item | Value | Why it matters |
|---|---|---|
| Frame sampling rate | 2 fps | Changes N, and so changes Eq.3's mask behaviour and episode length |
| Discount factor gamma | 0.99 | Eq.8 takes gamma as an input and never gives a value |
| Max time step T | 80 | Eq.8 references T; no value appears anywhere |
| Numerical epsilon | 1e-8 | Interacts with Eq.6's ~1e-10 magnitudes |
| Evaluation segmentation | KTS, RBF kernel | §4.2 defines F-score but not how frames become shots |
| Evaluation budget | 15% of duration | Benchmark convention, not stated |

The reproduction's own notes list "frame sampling rate" and "how frames are mapped to
shots for F-score" under **"Missing from paper (important to decide)"**.

## 3. Not in the paper, not in the reproduction — decided here

| Item | Decision | Alternatives |
|---|---|---|
| Order of the 4 move actions | `[-5, -1, +1, +5]` | Any permutation; affects nothing if learned from scratch |
| What the LSTM consumes | The K selected features in temporal order | Could be all N frames; paper says only that an LSTM exists |
| Actor/critic sharing | Separate heads on a shared encoder | Fully separate networks |
| Eq.15 on termination | Order-breaking move is undone | Could be kept, leaving a terminal state that violates the rule |

## 4. Where the paper is internally ambiguous

**Eq.4 (representativeness).** As printed, `exp(-1/K sum_i min_{j<K} ||x^k_i - x^k_j||)`
ranges both indices over the *selected* frames, so the minimum is zero at `i = j` and
the term is degenerate. The paper states the reward follows Zhou et al. [10], whose
representativeness is the mean over **all** frames of distance to the nearest
**selected** frame. That reading is used. The reproduction uses the same reading but
with *squared* distance, where the paper writes `||.||_2`.

**Eq.13 (balance factor).** `lambda = floor(E_t / E_c) mod 2` with `E_t` defined as
"the total number of iterations" makes lambda a constant for the whole run, which
would make Eq.14 train only actors or only critics and never both. Read as the
*current* iteration index — the only reading under which Alg.1's alternation works.

**Eq.6 (information reward).** `I_t = prod of K probabilities` is ~1e-10 at K=10, so
`r_h = I_t - I_{t-1}` (Eq.7) carries ~1e-15 of dynamic range. The reproduction
implements both the literal product and a log-domain variant, noting the latter is
needed "for stability". This is an observation about magnitude, **not** evidence that
the paper is wrong — reward normalisation is standard and routinely unstated.

**Eq.3 (diversity) is NOT ambiguous and NOT a defect.** It yields 1.0 whenever the K
selections are mutually more than 2 frames apart, which is the common case. The paper
states this is intended: *"the diversity reward works in the case that two frames are
temporally close."* It is a guard rail against the vertical policy collapsing
selections together, not a dense training signal.

## 5. Where the reference reproduction departs from the published equations

Worth recording, because it reached the paper's F1 while doing all of these:

1. **No bootstrap in the returns.** Eq.8 specifies `R_T = r_T + gamma*V(s_T)`; the
   repro initialises the recursion at 0.
2. **Entropy bonus** (`entropy_coef = 0.01`). Absent from Eq.9 and Eq.10.
3. **Separate value coefficient** (`value_coef = 0.5`). Eq.14 weights critic terms by
   `1 - lambda`, not by a constant.
4. **Gradient clipping** (`grad_clip = 5.0`). Not mentioned.
5. **Mean instead of sum over time** in the actor loss. Eq.9/10 are `-1/B sum_b sum_t`.
6. **Squared distance** in representativeness where Eq.4 writes `||.||_2`.

## 6. What reproduces and what does not

| Metric | Reference reproduction | Paper | Reproduces? |
|---|---:|---:|:---:|
| TVSum F1 | 63.50% +/- 1.45 | 63.0% | yes |
| TVSum Kendall tau | **-0.0105** | 0.080 | **no** |
| TVSum Spearman rho | **-0.0149** | 0.131 | **no** |
| SumMe F1 | 38.63% +/- 7.40 | 46.3% | partially |
| SumMe Kendall tau | **-0.0005** | 0.075 | **no** |

The F-score reproduces; the rank correlations do not, landing at approximately zero
and slightly negative across five folds.

This matters because of Otani et al. (CVPR 2019), who showed random summaries match
state of the art under F1 on these benchmarks, which is precisely why rank correlation
was introduced. **The method reproduces on the metric that cannot discriminate between
methods, and fails to reproduce on the one that can.**

This is a claim about reproducibility and evaluation practice. It is not a claim that
the paper is wrong, and it rests on two independent reimplementations rather than on
inference from the text.
