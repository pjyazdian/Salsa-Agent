# Motion Representation Visualization

Gradio web app for inspecting trained `MotionModel` checkpoints: reconstructions, latent space, VQ-VAE codebooks, and InterHuman/relationship views.

Supports vanilla AE, VAE, and VQ-VAE on HumanML3D, InterHuman, or relationship representations. For training and checkpoint naming, see the [motion representation README](../README.md). For conda environment and dataset setup, see the [main project README](../../README.md).

---

## Quick Start

```bash
# From Salsa-Agent/
conda activate motionagent
python -m motion_representation.visualization.vae_visualization_app
```

Open `http://localhost:7863` (the app binds to `0.0.0.0:7863`).

1. **Load Model** — use a path under `motion_representation/checkpoints_{model_name}/` (see [parent README](../README.md)), e.g. `checkpoints_VQVAE_GRU/best_checkpoint.pth` for HumanML3D, `checkpoints_VQVAE_GRU_InterHuman/best_checkpoint.pth` for InterHuman, or `checkpoints_VQVAE_GRU_Relationship/best_checkpoint.pth` for relationship features.
2. **Load Dataset** — default LMDB: `dataset_processed_New/lmdb_Salsa_pair/lmdb_train` (representation type comes from the checkpoint).
3. **Visualize** — reconstruction browser, t-SNE, VQ-VAE tools, and other sections in the UI.

---

## Features

**Core**

- **Model and dataset loading** — load a checkpoint; dataset representation is taken from the checkpoint config (`humanml3d`, `interhuman`, or `relationship`).
- **Reconstruction visualization** — browse by sample index; side-by-side original vs reconstructed 3D motion videos; MSE, MAE, and latent statistics (VQ-VAE metrics when applicable).
- **Latent space (t-SNE)** — encode 10–10,000 random samples; interactive t-SNE plot with density heatmap; optional k-means coloring; click a point or enter an index to inspect that sample.

**VQ-VAE only**

- **Codebook debugging** — codebook usage, diversity metrics, and t-SNE views of codebook entries vs encoder outputs.
- **Token cluster exploration** — quantize the dataset, then list and visualize samples assigned to each code index.

**Sequences and pairs**

- **Long sequence generation** — load raw LMDB, pick video/clip/role and frame range; autoregressive reconstruction over 20-frame windows.
- **Combined reconstruction (both dancers)** — leader and follower reconstructions in one view.

**InterHuman and relationship**

- **Relationship features** — time-series plots of 4D relationship vectors `[w, z, x, z]` (original vs reconstructed).
- **Combined motion + relationship** — load separate motion and relationship checkpoints and view fused ground-truth vs reconstruction.

---

## Usage Notes

Load the model before the dataset. For **InterHuman representation** 3D skeleton plots, the app uses `plot_3d_motion` and `HML_KINEMATIC_CHAIN` from [in2IN](https://github.com/pabloruizponce/in2IN) (`Download/in2IN`). HumanML3D uses `utils.motion_utils`.

---

## Requirements

- `gradio`, `torch`, `numpy`, `scikit-learn`, `plotly`
- Trained checkpoint and LMDB data (see parent README)
- [in2IN](https://github.com/pabloruizponce/in2IN) (`plot_3d_motion`, `HML_KINEMATIC_CHAIN`) for InterHuman representation visualization

Full environment setup: [main project README](../../README.md).

---

## Related: CLI comparison videos

For batch export of original vs reconstructed MP4s without the UI:

```bash
python -m motion_representation.visualize_comparison \
    --checkpoint motion_representation/checkpoints_VQVAE_GRU/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --num_samples 5 \
    --output_dir motion_representation/visualizations \
    --fps 20
```

See [motion representation README](../README.md) for evaluation and training commands.
