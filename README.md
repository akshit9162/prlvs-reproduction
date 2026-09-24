# PRLVS reproduction: Progressive RL for Video Summarization

**[→ Visual walkthrough and results](https://akshit9162.github.io/prlvs-reproduction/)**

A from-scratch PyTorch reproduction of

> Wang, G., Wu, X., & Yan, J. (2024). *Progressive reinforcement learning for video summarization.*
> Information Sciences 655, 119888. [doi:10.1016/j.ins.2023.119888](https://doi.org/10.1016/j.ins.2023.119888)

It evaluates rank correlation (Kendall τ / Spearman ρ against all 20 TVSum annotators) and the benchmark F-score, then uses controlled baselines to find *why* the result does not reproduce, and tests a fix.

## Result

TVSum, 50 videos, GoogLeNet 1024-d features at 2 fps, K = 10, 300 epochs.

| | Kendall τ | Spearman ρ |
|---|---:|---:|
| PRLVS, paper-faithful reward | −0.008 | −0.010 |
| PRLVS, repaired (log-domain) reward | −0.004 | −0.005 |
| Random scores | −0.004 | −0.006 |
| Human, leave-one-out | 0.312 | 0.400 |
| PRLVS as published | 0.080 | 0.131 |

Neither variant beats random on rank correlation. That matches the independent reproduction ([Vivek5170](https://github.com/Vivek5170/Progressive-Reinforcement-Learning-for-Video-Summarization), τ = −0.011), which reports the paper's 63% F1 but treats the τ gap as a side note.

## F1 can't tell PRLVS from random

`std_f1.py` computes the paper's F-score under the standard benchmark protocol: KTS shot segmentation (Potapov et al. 2014) on the frame features, mean score per shot, a 0/1 knapsack filling 15% of the video, and F1 against each annotator's summary built the same way. Brackets are bootstrap 95% CIs over the 50 videos.

| Selector | F1, mean over annotators | vs random (Wilcoxon) |
|---|---:|---|
| PRLVS, paper-faithful reward | 56.1% [51.8, 60.9] | p = 0.71 |
| PRLVS, repaired reward | 54.5% [49.9, 59.5] | p = 0.15 |
| **Random scores** | **54.9%** [50.2, 59.8] | — |
| Importance model (see below) | 58.1% [53.4, 63.1] | p = 0.03 |
| Humans, each annotator vs the rest | 57.0% [52.9, 61.5] | — |
| PRLVS as published | 63.0% | |

Random frame scores reach 55% F1, within 2 points of human annotators. Whole-shot selection under a length budget does most of the work, which is Otani et al.'s (CVPR 2019) point. PRLVS lands in the same band and is statistically indistinguishable from random. So matching the paper's F1 says little about the method. Rank correlation, where random scores 0 and humans 0.31, is the metric that separates them.

This protocol gives 56.1%, not 63.0%. The paper does not specify its KTS settings, the frame rate it scores at, or how F1 is reduced over annotators (taking the best annotator instead of the mean gives 82%, so the paper averaged). Those details plausibly account for the gap, so this is not a claim that 63% is wrong.

`eval_prlvs.py`'s F1 column is a stricter frame-level top-15% overlap and is not comparable to either.

## Why it fails: the reward, not the optimiser

`diagnose_eval.py` sends several selectors through the *same* frame-scoring pipeline: sample K frames, repeat R times, count, smooth, then compute τ against the mean human score. Each selector has its own random stream. R = 150; brackets are bootstrap 95% CIs over the 50 videos.

| Selector | What it tests | Kendall τ |
|---|---|---:|
| Oracle | Samples frames in proportion to human scores: the pipeline's ceiling | **0.729** [0.714, 0.744] |
| Reward-greedy, faithful | Hill-climbs the paper's reward (Eq. 5) directly, with no RL | 0.006 [−0.011, 0.022] |
| Reward-greedy, repaired | Same, with the repaired reward | 0.008 [−0.010, 0.026] |
| PRLVS, faithful | Trained agent | −0.006 [−0.025, 0.013] |
| PRLVS, repaired | Trained agent | 0.019 [0.002, 0.037] |
| PRLVS, untrained | Same network at random init | 0.001 [−0.018, 0.020] |
| Uniform | Random K-frame samples | −0.012 [−0.034, 0.012] |

- **The scoring pipeline isn't the bottleneck.** A selector that knows the human scores reaches τ = 0.73 through it.
- **The reward doesn't track human importance.** Selections that directly maximise it are no better than random, so even a perfect optimiser of this reward can't beat random on rank correlation.
- **Training changes nothing measurable.** Trained vs untrained, paired Wilcoxon: p = 0.44 (faithful), 0.15 (repaired).

## Fix attempt: an importance-aware reward

If the reward carries no human signal, add one. `importance_cv.py` trains a small frame-importance regressor on the GoogLeNet features with 5-fold cross-validation over videos, so every prediction is for a video the model never saw. `train_importance.py` then adds it to the vertical reward:

    r_v' = 0.5 · r_v (Eq. 5, as printed)  +  0.5 · mean importance of the K selected frames

Everything else (agent, Eq. 6–7, Alg. 1, optimiser, 300 epochs, faithful equations) is unchanged. Each fold's agent trains on 40 videos and is scored only on its 10 held-out videos. R = 150.

| Selector | Kendall τ, 5-fold held-out |
|---|---:|
| Oracle | 0.729 [0.714, 0.744] |
| **Importance model alone** | **0.243** [0.193, 0.291] |
| Reward-greedy, paper's reward | 0.006 [−0.011, 0.024] |
| Reward-greedy, importance-aware reward | 0.064 [0.042, 0.085] |
| PRLVS, paper's reward | 0.007 [−0.012, 0.026] |
| PRLVS, importance-aware reward | −0.007 [−0.026, 0.012] |
| Uniform | −0.012 [−0.035, 0.012] |

- **The missing signal exists in the features.** The held-out importance model reaches τ = 0.24.
- **The fixed reward carries it, but weakly.** Maximising it directly now beats random (0.064, CI excludes 0), against 0.006 for the paper's reward.
- **The RL agent still doesn't learn it.** Trained on the fixed reward, it scores τ = −0.007, no different from the agent trained on the paper's reward (paired Wilcoxon p = 0.37) or from uniform (p = 0.57). So the reward was one failure, and the agent not learning even a reward that carries signal is a second.
- **The action space caps it.** Even direct maximisation of the fixed reward stops at 0.064. The agent moves each of K frames by ±1 or ±5 positions within 40 steps from random starting points, so frames far from where it starts are out of reach. The reference reproduction's "linspace" initialisation (frames evenly spaced at the start) targets the same limitation; it reports only F1 for it, so its effect on τ is untested.

## Provenance

The paper cannot be implemented without decisions it leaves unstated. [`PROVENANCE.md`](PROVENANCE.md) records every one of them:
- what the paper specifies
- what was taken from the reference reproduction
- what was decided here
- where the paper is internally ambiguous (Eq. 4, Eq. 13)
- where the reference reproduction departs from the published equations

## Two variants

- **faithful**: the information reward is the literal product of K probabilities (Eq. 6). At K = 10 that product is about 1e-10, so the reward difference in Eq. 7 has about 1e-15 of dynamic range.
- **repaired**: Eq. 6 computed in the log domain, and Eq. 3's index `g()` read as the frame's position in the summary rather than its frame number. The second change gives the diversity term a gradient. See `PROVENANCE.md` §4.

## Files

| File | Purpose |
|---|---|
| `prlvs.py` | Agent, environment and rewards used for all results (`MODE` selects the variant) |
| `prlvs_paper.py` | Annotated version tagging each constant as `[PAPER]`, `[REPRO]` or `[UNRESOLVED]` |
| `extract_googlenet.py` | GoogLeNet pool5 features at 2 fps from TVSum videos |
| `train_prlvs.py` | Two-policy A2C training with alternating actor/critic updates (Alg. 1) |
| `eval_prlvs.py` | τ / ρ / F1 against every annotator, plus random and leave-one-out human baselines |
| `diagnose_eval.py` | Oracle, reward-greedy, untrained and uniform selectors through the same scoring, with bootstrap CIs and Wilcoxon tests |
| `diagnose_eval.json`, `diagnose.log` | Per-video τ for every selector at R = 30 and 150 |
| `std_f1.py`, `std_f1.json`, `std_f1.log` | Standard-protocol F-score (KTS shots, knapsack) for PRLVS, random, the importance model and humans |
| `importance_cv.py`, `importance_cv.json`, `importance_cv_pred.npz` | 5-fold cross-validated frame-importance regressor and its held-out predictions |
| `train_importance.py` | PRLVS with the importance-aware reward: per-fold training and held-out evaluation |
| `importance_eval.json`, `importance_eval.log`, `train_imp_fold*.log` | Per-video τ for every selector, and training logs |
| `docs/` | GitHub Pages walkthrough |
| `run_all.sh` | Full pipeline: sanity pass, then train and evaluate both variants |
| `prlvs_eval_*.json`, `train_hist_*.json`, `run_all.log` | Raw per-video results and training curves |

## Run

```bash
pip install -r requirements.txt
python extract_googlenet.py <tvsum_dir>   # writes features_tvsum.npz (~90 MB, not committed)
TVSUM_ROOT=<tvsum_dir> PY=python ./run_all.sh
python diagnose_eval.py features_tvsum.npz <tvsum_dir> 30 150              # controlled selectors
python std_f1.py features_tvsum.npz <tvsum_dir>                            # standard F1
python importance_cv.py features_tvsum.npz <tvsum_dir>                     # importance regressor
for f in 0 1 2 3 4; do python train_importance.py features_tvsum.npz <tvsum_dir> $f 300; done
python train_importance.py features_tvsum.npz <tvsum_dir> eval 150         # fix experiment
```

TVSum: <https://github.com/yalesong/tvsum>
