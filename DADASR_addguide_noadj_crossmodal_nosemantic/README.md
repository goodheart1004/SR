# DADASR Add-Guide No-Adjustment Cross-Modal Fusion

This project is a DSM super-resolution variant built on the DADASR-style
anisotropic diffusion baseline. It keeps the RGB plus adapter-guide input path,
uses the local residual refinement network, and adds an optional RGB--DSM
cross-modal feature fusion block before refinement. It does not include the
SAM3 semantic FiLM modulation used by `/Users/niko/Downloads/project`.

## Main Configuration

| Component | Setting |
| --- | --- |
| Guide input | RGB + `adapter_guide` (`guide_channels = 4`) |
| Adjustment path | Disabled |
| Local refinement | Enabled by default |
| Boundary refinement | Enabled by default with flat/edge residual heads |
| Cross-modal fusion | Available through `--cross-modal-fusion`, disabled by default |
| Semantic modulation | Not included |
| Diffusion | Perona-Malik anisotropic diffusion after optional refinement |

## Key Modules

- `model/gad_base.py`: core GAD/DADASR model, local refinement module, and
  RGB--DSM fusion modules.
- `data/processed_dsm.py`: `ProcessedData_scale10` dataset loader for
  DSM HR/LR, RGB, adapter guide, optional SAM3 boundary images, and labels.
- `run_train.py`: training entry point with checkpoint/log writing.
- `run_eval.py`: evaluation entry point with split selection, metrics, heatmaps,
  and auxiliary fusion/refinement visualization outputs.
- `sam3-patch.py`: helper script for generating SAM3 patch outputs when the
  required SAM3 assets are available.

## Cross-Modal Fusion

The optional fusion path is implemented with three lightweight blocks:

- `GateConv2D`: learns spatial gates for RGB and DSM feature stems.
- `EITF`: computes local RGB--DSM similarity at matching spatial positions.
- `MMAB`: applies joint-statistics channel gates to RGB and DSM features.

When enabled, `RGBDSMFeatureFusion` injects the fused RGB--DSM features into the
base guide features before `LocalRefinementNet` predicts the DSM residual.

## Data Layout

The default dataset root is `ProcessedData_scale10` and follows this structure:

```text
ProcessedData_scale10/
|-- pos_train_DSM_HR/
|-- pos_train_DSM_LR/
|-- pos_train_RGB/
|-- pos_train_adapter_guide/
|-- pos_train_SAM3/
|-- pos_train_label/
|-- vai_train_DSM_HR/
|-- vai_train_DSM_LR/
|-- vai_train_RGB/
|-- vai_train_adapter_guide/
|-- vai_train_SAM3/
|-- vai_train_label/
|-- test_DSM_HR/
|-- test_DSM_LR/
|-- test_RGB/
|-- test_adapter_guide/
|-- test_SAM3/
`-- test_label/
```

The included files are small sample tiles. Full training requires the complete
processed dataset with matching numeric sample IDs across HR DSM, LR DSM, RGB,
and adapter-guide folders.

## Environment

Install the main Python dependencies in an environment with PyTorch support:

```bash
pip install configargparse matplotlib numpy pillow rasterio segmentation-models-pytorch tifffile torch torchvision tqdm
```

Use a CUDA-enabled PyTorch build for practical training speed.

## Training

Train the default boundary-aware refinement version:

```bash
python run_train.py \
  --save-dir runs \
  --data-dir ProcessedData_scale10
```

Enable the RGB--DSM cross-modal fusion path:

```bash
python run_train.py \
  --save-dir runs \
  --data-dir ProcessedData_scale10 \
  --cross-modal-fusion
```

Useful switches:

```text
--no-boundary-refinement    use a single residual head
--refinement-only           skip diffusion and train only local refinement
--cross-modal-reduction N   channel reduction ratio inside MMAB
--num-epochs N              training epoch count
--batch-size N              training batch size
```

## Evaluation

Evaluate a checkpoint on the validation or test split:

```bash
python run_eval.py path/to/best_model.pth --split test --out-dir eval_test
```

If the checkpoint `args.csv` is incomplete, `run_eval.py` falls back to safe
defaults and allows CLI overrides such as `--cross-modal-fusion`,
`--boundary-refinement`, `--feature-extractor`, `--Npre`, and `--Ntrain`.

## Relationship to Local Variants

| Local directory | Main difference |
| --- | --- |
| `/Users/niko/Downloads/project_doublerefinemtn` | Boundary-aware dual residual refinement only. It does not include the RGB--DSM cross-modal fusion modules. |
| `/Users/niko/Downloads/project_Film相较于project没加入语义调制` | This version. It adds RGB--DSM cross-modal fusion on top of dual residual refinement, but does not include semantic FiLM modulation. |
| `/Users/niko/Downloads/project` | Adds SAM3 semantic inputs, semantic masks, semantic valid masks, semantic boundary prior, and `SemanticFiLMResidualBlock` modulation inside local refinement. |

