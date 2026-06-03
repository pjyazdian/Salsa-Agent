# Motion Representation Learning

Learn compact motion representations from Salsa dance-pair data. This module trains autoencoders and tokenizers over fixed-length motion windows, supporting multiple encoder/decoder architectures, latent-space variants, and input representations.

**Supported models:** vanilla autoencoder (AE), VAE, VQ-VAE  
**Supported encoders/decoders:** GRU, Transformer  
**Supported representations:** HumanML3D, InterHuman (canonicalized), relationship features

For environment setup, dataset download, and dependencies, see the [main project README](../README.md). All commands below assume you are in the `Salsa-Agent` repository root with the `motionagent` conda environment activated.

---

## Quick Start

**Recommended HumanML3D tokenizer** — VQ-VAE on 263-dim HumanML3D windows (`--representation_type humanml3d`, the default). For the recommended InterHuman setup used in the Salsa pipeline, see [InterHuman Motion Tokenizer](#interhuman-motion-tokenizer).

```bash
# From Salsa-Agent/ (see ../README.md for conda env and data setup)
conda activate motionagent

python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 2048 \
    --learning_rate 1e-4 \
    --num_epochs 2000 \
    --use_vqvae \
    --nb_code 512 \
    --quantizer ema_reset \
    --vq_mu 0.95 \
    --commit_weight 0.02 \
    --loss_vel_weight 0.1 \
    --warm_up_epochs 5 \
    --representation_type humanml3d \
    --model_name VQVAE_GRU

# Monitor training
tensorboard --logdir motion_representation/checkpoints_VQVAE_GRU/logs --port 6006

# Evaluate (use the checkpoint dir that matches --model_name)
python -m motion_representation.eval \
    --checkpoint motion_representation/checkpoints_VQVAE_GRU/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --output_dir motion_representation/eval_outputs
```

Checkpoints are saved under `motion_representation/checkpoints_{model_name}/`. Re-run the same training command to resume automatically from the latest checkpoint.

---

## Representations & Data

Paired leader/follower clips from LMDB, sliced into 20-frame windows (`stride=5`) and cached on first run. Default data path: `dataset_processed_New/lmdb_Salsa_pair/lmdb_train` (`--lmdb_dir`).

- **`humanml3d`** (default) — 263-dim HumanML3D vectors, 20 frames. Cache: `{lmdb_dir}_humanml3d_20frames_cache/`.
- **`interhuman`** — 262-dim canonicalized single-dancer motion, 19 frames. Recommended for the Salsa pipeline motion tokenizer (see Training).
- **`relationship`** — 4-dim relative pose `[w, z, x, z]` between dancers, 19 frames. Shares the InterHuman cache (`{lmdb_dir}_interhuman_20frames_cache/`).

With `--use_both_roles` (default), leader and follower are separate training samples.

---

## Model & Loss

`MotionModel` encodes `(batch, seq_len, input_dim)` to a latent code and decodes back to motion.

**Training modes:**
- **Vanilla AE** — no extra flags; continuous deterministic latent.
- **VAE** — `--use_vae`; Gaussian latent with reparameterization.
- **VQ-VAE** — `--use_vqvae`; discrete codebook (overrides VAE).

**Loss terms:**
- **Reconstruction (L1)** — always on; weight `--recon_weight` (default 1.0).
- **Velocity (L1)** — frame-to-frame smoothness; on when `--loss_vel_weight > 0` (use 0.1 for VQ-VAE).
- **KL** — VAE only; weight `--kl_weight` (default 1e-4).
- **Commitment** — VQ-VAE only; pulls encoder output toward codebook; weight `--commit_weight` (default 0.02).

Default GRU: `latent_dim=512`, `hidden_dim=512`, `num_layers=2`. Transformer: 8 layers, 8 heads, `ff_size=2048`.

---

## Training

Run all training from the `Salsa-Agent/` root:

```bash
python -m motion_representation.train [options]
```

Training supports automatic resume: if checkpoints exist in the target directory and `--resume` is omitted, the latest checkpoint is loaded and epoch numbering continues.

### Vanilla GRU Autoencoder

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --num_epochs 100 \
    --loss_vel_weight 0.1 \
    --model_name Vanilla_GRU_Continuous
```

### VAE with GRU

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --use_vae \
    --model_name VAE_GRU
```

### Transformer Autoencoders

```bash
# Vanilla
python -m motion_representation.train \
    --encoder_type transformer \
    --decoder_type transformer \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 8 \
    --num_heads 8 \
    --ff_size 2048 \
    --activation gelu \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --model_name Vanilla_Transformer

# VAE
python -m motion_representation.train \
    --encoder_type transformer \
    --decoder_type transformer \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 8 \
    --num_heads 8 \
    --ff_size 2048 \
    --activation gelu \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --use_vae \
    --model_name VAE_Transformer
```

### VQ-VAE with GRU (T2M-GPT-style)

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --num_epochs 100 \
    --use_vqvae \
    --nb_code 512 \
    --quantizer ema_reset \
    --vq_mu 0.99 \
    --commit_weight 0.02 \
    --loss_vel_weight 0.1 \
    --warm_up_iter 1000 \
    --lr_scheduler multistep \
    --lr_scheduler_milestones 50 200 \
    --lr_scheduler_gamma 0.05 \
    --model_name VQVAE_GRU
```

### VQ-VAE Quantizer Variants

```bash
# EMA with reset (default)
python -m motion_representation.train --encoder_type gru --decoder_type gru \
    --use_vqvae --quantizer ema_reset --vq_mu 0.99 --model_name VQVAE_GRU_ema_reset

# Original quantizer
python -m motion_representation.train --encoder_type gru --decoder_type gru \
    --use_vqvae --quantizer orig --vq_beta 1.0 --model_name VQVAE_GRU_orig

# EMA only
python -m motion_representation.train --encoder_type gru --decoder_type gru \
    --use_vqvae --quantizer ema --vq_mu 0.99 --model_name VQVAE_GRU_ema

# Reset only
python -m motion_representation.train --encoder_type gru --decoder_type gru \
    --use_vqvae --quantizer reset --model_name VQVAE_GRU_reset
```

### InterHuman Motion Tokenizer

**Recommended InterHuman tokenizer** — same hyperparameters as Quick Start, but trains on **canonicalized 262-dim InterHuman** motion (`--representation_type interhuman`, 19 frames). Use this when downstream models consume InterHuman features rather than HumanML3D vectors.

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 2048 \
    --learning_rate 1e-4 \
    --num_epochs 2000 \
    --use_vqvae \
    --nb_code 512 \
    --quantizer ema_reset \
    --vq_mu 0.95 \
    --commit_weight 0.02 \
    --loss_vel_weight 0.1 \
    --warm_up_epochs 5 \
    --lr_scheduler_gamma 0.05 \
    --representation_type interhuman \
    --use_both_roles \
    --model_name VQVAE_GRU_InterHuman
```

Checkpoints: `motion_representation/checkpoints_VQVAE_GRU_InterHuman/`

### Relationship Tokenizer

Compact model for 4-dim leader–follower relative features:

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 32 \
    --hidden_dim 32 \
    --num_layers 2 \
    --batch_size 2048 \
    --learning_rate 1e-4 \
    --num_epochs 200 \
    --use_vqvae \
    --nb_code 512 \
    --quantizer ema_reset \
    --vq_mu 0.95 \
    --commit_weight 0.02 \
    --loss_vel_weight 0.1 \
    --warm_up_epochs 5 \
    --lr_scheduler_gamma 0.05 \
    --representation_type relationship \
    --use_both_roles \
    --model_name VQVAE_GRU_Relationship
```

Checkpoints: `motion_representation/checkpoints_VQVAE_GRU_Relationship/`

### Resuming Training

**Automatic (recommended):** re-run the same command with the same `--model_name`.

**Manual:**
```bash
python -m motion_representation.train \
    ... \
    --model_name Vanilla_GRU \
    --resume motion_representation/checkpoints_Vanilla_GRU/checkpoint_epoch_50.pth
```

`--num_epochs` is the total target epoch count. Resuming from epoch 50 with `--num_epochs 200` trains epochs 51–200.

---

## Evaluation & Visualization

Always point `--checkpoint` to the directory created by your `--model_name`.

### Batch evaluation

```bash
python -m motion_representation.eval \
    --checkpoint motion_representation/checkpoints_VQVAE_GRU/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --output_dir motion_representation/eval_outputs
```

### Reconstruction videos

```bash
python -m motion_representation.visualize_comparison \
    --checkpoint motion_representation/checkpoints_VQVAE_GRU/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --num_samples 5 \
    --output_dir motion_representation/visualizations \
    --fps 20
```

Creates separate original and reconstructed videos; filenames include reconstruction error for comparison.

### Interactive app

See [visualization/README.md](visualization/README.md) for the Gradio-based exploration tool (latent space, reconstructions, InterHuman/relationship modes).

---

## Checkpoints & Cache

`--model_name X` writes to `motion_representation/checkpoints_X/` (`best_checkpoint.pth`, `checkpoint_epoch_*.pth`, `model_config.txt`, `logs/`). Without `--model_name`, the folder is `motion_representation/checkpoints/`.

Window caches are created beside the LMDB on first run (`_humanml3d_20frames_cache` or `_interhuman_20frames_cache`). Delete the cache folder to rebuild.

---

## Configuration Reference

Run `python -m motion_representation.train --help` for the full CLI. Key defaults from `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 20 | Frames per training window |
| `stride` | 5 | Window stride when building cache |
| `latent_dim` | 512 | Latent / code dimension |
| `hidden_dim` | 512 | GRU hidden / Transformer d_model |
| `num_layers` | 2 | Encoder/decoder depth (GRU) |
| `recon_weight` | 1.0 | L1 reconstruction weight |
| `kl_weight` | 1e-4 | KL weight (VAE only) |
| `loss_vel_weight` | 0.0 | Velocity loss weight |
| `nb_code` | 512 | VQ-VAE codebook size |
| `quantizer` | `ema_reset` | VQ quantizer type |
| `commit_weight` | 0.02 | VQ commitment loss weight |
| `learning_rate` | 2e-4 | Initial learning rate |
| `warm_up_iter` | — | LR warm-up in iterations |
| `warm_up_epochs` | 0 | LR warm-up in epochs (overrides iter if > 0) |

---

## TensorBoard Monitoring

Logs are written to `{checkpoint_dir}/logs/`.

```bash
tensorboard --logdir motion_representation/checkpoints_VQVAE_GRU/logs --port 6006
```

Open `http://localhost:6006`.

For VQ-VAE runs, watch `Train/Loss`, `Train/ReconLoss`, `Train/CommitLoss`, and `Train/Perplexity`. Reconstruction should fall, commitment should stabilize, and perplexity should settle around 50–80% of `nb_code`.
