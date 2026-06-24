# DADASR: Adapter Guide with Boundary-Aware Dual Refinement

`DADASR_addguide_noadj_doublerefine` is a DSM super-resolution variant
for the local `ProcessedData_scale10` layout. It uses RGB plus an
`adapter_guide` channel to predict a high-resolution DSM from a low-resolution
DSM, applies boundary-aware local refinement, and then runs anisotropic
diffusion without an adjustment module.

## Configuration

| Component | Setting |
| --- | --- |
| Guide input | RGB plus one-channel `adapter_guide` |
| Feature extractor | ResNet-50 U-Net |
| Local refinement | Enabled by default; 64 channels and 4 residual blocks |
| Refinement heads | Flat-region residual head and edge-region residual head |
| Boundary gate | Learns a pixelwise gate from refinement features and a boundary prior |
| Boundary prior | Edge map derived from `SAM3`, then `label`, then the last guide channel |
| Post-refinement stage | Perona-Malik anisotropic diffusion |
| Adjustment module | Disabled |

The refinement residual is blended per pixel:

```text
residual = (1 - boundary_gate) * flat_residual + boundary_gate * edge_residual
y_refined = y_bicubic + residual
```

When a boundary prior is available, the final gate averages the learned gate
and the normalized prior. The model reports the two residuals and the gate as
auxiliary outputs for inspection.

## Project Layout

```text
DADASR_addguide_noadj_doublerefine/
|-- arguments/                 # Training and evaluation CLI definitions
|-- data/                      # Paired RGB, DSM, adapter-guide and boundary-prior loader
|-- model/
|   `-- gad_base.py            # U-Net features, dual-head refinement and diffusion model
|-- ProcessedData_scale10/     # Small example train/validation/test data layout
|-- losses.py                  # Masked L1/RMSE losses for final and refinement outputs
|-- run_train.py               # Training, validation, TensorBoard and checkpoint loop
|-- run_eval.py                # Checkpoint evaluation and error/fusion visualizations
`-- utils.py                   # Logging, argument CSV export, seeding and device helpers
```

## Dataset Layout

The required folders for each split are `DSM_HR`, `DSM_LR`, `RGB`, and
`adapter_guide`. `SAM3` and `label` are optional but recommended because they
provide the boundary prior used by the dual residual heads.

```text
ProcessedData_scale10/
|-- pos_train_DSM_HR/
|-- pos_train_DSM_LR/
|-- pos_train_RGB/
|-- pos_train_adapter_guide/
|-- pos_train_SAM3/            # Optional
|-- pos_train_label/           # Optional
|-- vai_train_*/               # Validation folders with the same suffixes
`-- test_*/                    # Test folders with the same suffixes
```

File IDs must match across folders within a split. The committed dataset files
are small examples of the expected layout; replace them with the complete
dataset for training.

## Training

Install compatible PyTorch, torchvision, `segmentation-models-pytorch`, NumPy,
Pillow, TensorBoard and tqdm packages, then run:

```bash
python run_train.py \
  --save-dir ./save_dir \
  --data-dir ProcessedData_scale10 \
  --num-epochs 200 \
  --boundary-refinement
```

`--boundary-refinement` is enabled by default. Use
`--no-boundary-refinement` only to reproduce the original single residual-head
refinement. Each experiment stores `args.csv`, `best_model.pth`, and
`last_model.pth` under `save_dir/DSM/experiment_<n>_<randn>/`.

## Evaluation

```bash
python run_eval.py \
  --checkpoint ./save_dir/DSM/experiment_<n>_<randn>/best_model.pth \
  --split test
```

Keep `args.csv` beside the checkpoint so evaluation reconstructs the dual-head
architecture. If it is unavailable, pass `--boundary-refinement` explicitly.
