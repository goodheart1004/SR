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
├── pos_train_adapter_guide
├── vai_train_DSM_HR
├── vai_train_DSM_LR
├── vai_train_RGB
├── vai_train_adapter_guide
├── test_DSM_HR
├── test_DSM_LR
├── test_RGB
└── test_adapter_guide
```

`SAM3` and `label` folders are loaded when present, but the current DADA-style model uses RGB plus `adapter_guide` as the guide input.

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

TensorBoard logs are saved in the experiment folder:

```bash
tensorboard --logdir ./save_dir/ProcessedData_scale10
```

## Evaluation

```bash
python run_eval.py --checkpoint ./save_dir/ProcessedData_scale10/experiment_<id>/best_model.pth
```

By default evaluation uses the full test image (`--crop-size 0`) and reports `l1_loss`, `mse_loss`, `rmse_loss`, and `optimization_loss` in the DSM value units.
