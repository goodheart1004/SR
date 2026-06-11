# DSM Super-Resolution for ProcessedData_scale10

This repository is adapted from DADA for DSM super-resolution on the local `ProcessedData_scale10` dataset.

The project now has a single dataset path and no legacy benchmark dataset selection. Training logs are written with TensorBoard. Validation best checkpoints are selected by `rmse_loss`.

## Dataset

The default dataset root is `ProcessedData_scale10`. The loader expects these folders:

```text
ProcessedData_scale10
├── pos_train_DSM_HR
├── pos_train_DSM_LR
├── pos_train_RGB
├── vai_train_DSM_HR
├── vai_train_DSM_LR
├── vai_train_RGB
├── test_DSM_HR
├── test_DSM_LR
└── test_RGB
```

`SAM3`, `label`, and `adapter_guide` folders may be present in the dataset, but the current Real-GDSR-style model uses RGB only as the guide input.

## Setup

```bash
conda env create -f environment.yml
conda activate DSM-SR
```

## Training

```bash
python run_train.py --save-dir ./save_dir --num-epochs 4500 --val-every-n-epochs 1 --lr-step 100 --in-memory
```

Useful defaults:

- `--data-dir ProcessedData_scale10`
- `--scaling 10`
- `--crop-size 250`
- `--loss rmse`
- `--use-refinement-net`
- `--refinement-channels 64`
- `--refinement-blocks 4`

The model follows the Real-GDSR-style order: feature extraction from RGB plus bicubic DSM, local residual refinement, then the existing anisotropic diffusion loop without an adjustment step. Use `--refinement-only` to train only the local refinement module, or `--no-refinement-net` to run the diffusion baseline from bicubic DSM.

TensorBoard logs are saved in the experiment folder:

```bash
tensorboard --logdir ./save_dir/ProcessedData_scale10
```

## Evaluation

```bash
python run_eval.py --checkpoint ./save_dir/ProcessedData_scale10/experiment_<id>/best_model.pth
```

By default evaluation uses the full test image (`--crop-size 0`) and reports `l1_loss`, `mse_loss`, `rmse_loss`, optional `refinement_*` losses, and `optimization_loss` in the DSM value units.
