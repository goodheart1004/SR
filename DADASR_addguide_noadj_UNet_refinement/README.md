# DADASR addguide noadj UNet refinement

This project is a DSM super-resolution variant built around UNet-based local
refinement. It keeps the adapter-guide and RGB guidance path, disables the
adjustment path, and uses `LocalRefinementNet` as the main residual recovery
module for refinement-only DSM prediction or for initializing the diffusion
stage.

## Main Idea

- `feature_extractor=UNet` extracts dense guide features from RGB and adapter
  inputs.
- `LocalRefinementNet` predicts a residual over the bicubic DSM instead of
  predicting the full DSM from scratch.
- Optional boundary-aware flat and edge residual heads can be enabled through
  `--boundary-refinement`.
- Optional semantic FiLM modulation can be enabled through
  `--semantic-modulation` when SAM3 labels follow the expected
  building/road/background schema.
- Optional RGB-DSM cross-modal fusion can be enabled through
  `--cross-modal-fusion`.

## Directory Layout

```text
arguments/              Training and evaluation argument parsers
data/                   Processed DSM dataset loader and semantic priors
model/                  GADBase, UNet feature path, and LocalRefinementNet
sam3/                   SAM3 helper modules
docs/code_explanation/  Code walkthrough documents
run_train.py            Training entry point
run_eval.py             Evaluation entry point
losses.py               Masked DSM losses
utils.py                Logging, seeding, and device helpers
ProcessedData_scale10/  Small processed sample data for smoke tests
```

## Example Training Command

```bash
python run_train.py \
  --save-dir results_unet_refinement \
  --data-dir ProcessedData_scale10 \
  --batch-size 8 \
  --crop-size 250 \
  --scaling 10 \
  --feature-extractor UNet \
  --use-refinement-net \
  --boundary-refinement \
  --cross-modal-fusion \
  --refinement-only \
  --refinement-channels 96 \
  --refinement-blocks 8 \
  --num-epochs 150 \
  --lr 1e-4 \
  --gradient-clip 0.1
```

## Notes

This version should be read as the UNet refinement branch of the DSM SR
experiments. Its core output path is:

```text
y_bicubic + UNet guide features -> LocalRefinementNet residual -> pred DSM
```

The included `docs/code_explanation/` files describe the dataset, model, loss,
training, and evaluation code in more detail.
