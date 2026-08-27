# ReplayDrift

### Learning from Its Own Past through Repulsive Generative Replay

ReplayDrift is a follow-up to [Drifting](https://github.com/lambertae/drifting)
and DualDrift. It lets a generator learn not only from real data, but also from
where its previous selves have already been: class-conditioned samples from an
earlier generator state are replayed as detached repulsive particles.

The project also collects the efficiency controls developed while making
Drifting practical with smaller generator token counts, narrower feature
encoders, fewer GPUs, and lower-cost loss construction.

## Core idea

For each current generated query, the reverse-drift target pool contains

```text
[current generated | real negative | historical generated | real positive]
```

The historical samples come from a frozen, class-conditioned generator
snapshot. At replay ratio `rho`, generated repulsion is divided as

```text
current weight = 1 - rho
history weight = rho * G / H
```

where `G` is the number of current generated particles and `H` is the replay
count. Historical particles are detached targets: they do not retain a
generator or feature-encoder backward graph.

## Preliminary ImageNet-256 result

The controlled B/4 MAE-256 comparison below starts every continuation from the
same epoch-10 checkpoint. Evaluation uses 50,000 samples at CFG 1.4.

| Method at epoch 40 | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|
| Reverse drift, no replay | 18.510 | 81.35 | 0.7431 | **0.4060** |
| ReplayDrift, `H=16`, `rho=0.5` | **11.822** | **122.53** | **0.7839** | 0.3305 |

Replay improves convergence, FID, and precision, while the lower recall exposes
an important quality–coverage trade-off. A lower replay ratio (`rho=0.35`) is a
useful balanced setting.

## Setup

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ImageNet, latent caches, pretrained MAE weights, FID statistics, checkpoints,
and generated samples are intentionally not stored in Git. Supply their paths
locally through the launch-script environment variables.

Example B/4 reverse-drift launch:

```bash
TORCHRUN_BIN="$(command -v torchrun)" \
IMAGENET_PATH=/path/to/ILSVRC2012 \
IMAGENET_CACHE_PATH=/path/to/image_latents \
MAE_CKPT=/path/to/mae_latent_256.pt \
GPU_IDS=0,1 NPROC_PER_NODE=2 \
LAUNCH_MODE=background \
bash scripts/run_B4_rev-drift_mae256.sh
```

The historical-replay causal and efficiency launchers are under `scripts/`.
Their configurations keep the baseline and replay comparisons matched in
checkpoint, seed, target count, and evaluation protocol.

## Repository policy

The following stay local and are ignored by Git:

- datasets and latent caches;
- pretrained weights and checkpoints;
- generated samples and evaluation archives;
- experiment runs, logs, PID files, and W&B state;
- Python, test, profiler, and editor caches.

## Attribution

ReplayDrift builds on the public Drifting implementation and the DualDrift code
lineage. Please preserve upstream attribution when redistributing this work.

## Status

This is research code under active development. Reproducibility scripts and
ablation configurations are included, but paths to datasets and pretrained
weights must be configured for each environment.
