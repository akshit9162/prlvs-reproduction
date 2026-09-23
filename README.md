# PRLVS reproduction: Progressive RL for Video Summarization

**[→ Visual walkthrough and results](https://akshit9162.github.io/prlvs-reproduction/)**

A from-scratch PyTorch reproduction of

> Wang, G., Wu, X., & Yan, J. (2024). *Progressive reinforcement learning for video summarization.*
> Information Sciences 655, 119888. [doi:10.1016/j.ins.2023.119888](https://doi.org/10.1016/j.ins.2023.119888)

It includes a rank-correlation evaluation (Kendall τ / Spearman ρ against all 20 TVSum annotators) that the reference reproduction does not have.

## Result

TVSum, 50 videos, GoogLeNet 1024-d features at 2 fps, K = 10, 300 epochs.

| | Kendall τ | Spearman ρ |
|---|---:|---:|
| PRLVS, paper-faithful reward | −0.008 | −0.010 |
| PRLVS, repaired (log-domain) reward | −0.004 | −0.005 |
| Random scores | −0.004 | −0.006 |
| Human, leave-one-out | 0.312 | 0.400 |
| PRLVS as published | 0.080 | 0.131 |

Neither variant beats random on rank correlation. That matches the independent reproduction ([Vivek5170](https://github.com/Vivek5170/Progressive-Reinforcement-Learning-for-Video-Summarization), τ = −0.011), which *does* reproduce the paper's 63% F1. So the method reproduces on F1, which random summaries also score well on (Otani et al., CVPR 2019). It does not reproduce on the metric that can tell methods apart.

F1 in `eval_prlvs.py` is a simple top-k overlap, not the KTS-segmented benchmark F1. Do not compare it to the paper's F1.

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
| `docs/` | GitHub Pages walkthrough |
| `run_all.sh` | Full pipeline: sanity pass, then train and evaluate both variants |
| `prlvs_eval_*.json`, `train_hist_*.json`, `run_all.log` | Raw per-video results and training curves |

## Run

```bash
pip install -r requirements.txt
python extract_googlenet.py <tvsum_dir>   # writes features_tvsum.npz (~90 MB, not committed)
TVSUM_ROOT=<tvsum_dir> PY=python ./run_all.sh
```

TVSum: <https://github.com/yalesong/tvsum>
