# PRLVS reproduction: Progressive RL for Video Summarization

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

## Provenance

The paper cannot be implemented without decisions it leaves unstated. [`PROVENANCE.md`](PROVENANCE.md) records every one of them:
- what the paper specifies
- what was taken from the reference reproduction
- what was decided here
- where the paper is internally ambiguous (Eq. 4, Eq. 13)
- where the reference reproduction departs from the published equations

## Two variants

- **faithful**: the information reward is the literal product of K probabilities (Eq. 6). At K = 10 that product is about 1e-10, so the reward difference in Eq. 7 has about 1e-15 of dynamic range.
- **repaired**: the same reward computed in the log domain.

## Files

| File | Purpose |
|---|---|
| `prlvs.py` | Agent, environment and rewards used for all results (`MODE` selects the variant) |
| `prlvs_paper.py` | Annotated version tagging each constant as `[PAPER]`, `[REPRO]` or `[UNRESOLVED]` |
| `extract_googlenet.py` | GoogLeNet pool5 features at 2 fps from TVSum videos |
| `train_prlvs.py` | Two-policy A2C training with alternating actor/critic updates (Alg. 1) |
| `eval_prlvs.py` | τ / ρ / F1 against every annotator, plus random and leave-one-out human baselines |
| `run_all.sh` | Full pipeline: sanity pass, then train and evaluate both variants |
| `prlvs_eval_*.json`, `train_hist_*.json`, `run_all.log` | Raw per-video results and training curves |

## Run

```bash
pip install -r requirements.txt
python extract_googlenet.py <tvsum_dir>   # writes features_tvsum.npz (~90 MB, not committed)
TVSUM_ROOT=<tvsum_dir> PY=python ./run_all.sh
```

TVSum: <https://github.com/yalesong/tvsum>
