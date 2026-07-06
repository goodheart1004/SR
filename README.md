# DADASR DSM Super-Resolution Versions

This repository keeps six DADASR variants for DSM super-resolution on
`ProcessedData_scale10`. The previous single-version repository contents were
replaced by these versioned directories.

## Version Index

| Version | Module configuration | Description | Tags |
| --- | --- | --- | --- |
| `DADASR_addguide_noadj_addrefine` | adapter guide: on; adjustment: off; local refinement: on | Uses RGB plus `adapter_guide` as guide features, applies local residual refinement, then runs the anisotropic diffusion loop without the adjustment step. | `dsm-sr`, `adapter-guide`, `guide-on`, `no-adj`, `local-refinement`, `rmse` |
| `DADASR_addguide_noadj_doublerefine` | adapter guide: on; adjustment: off; boundary-aware dual local refinement: on | Extends the adapter-guide refinement variant with flat-region and edge-region residual heads, blended by a learned gate and a SAM3/label-derived boundary prior before anisotropic diffusion. | `dsm-sr`, `adapter-guide`, `guide-on`, `no-adj`, `dual-refinement`, `boundary-aware` |
| `DADASR_addguide_noadj_crossmodal_nosemantic` | adapter guide: on; adjustment: off; boundary-aware dual refinement: on; RGB--DSM fusion: optional; semantic modulation: off | Adds optional RGB--DSM cross-modal feature fusion with spatial gates, local similarity, and joint-statistics channel gates before the dual-head local refinement stage. It intentionally does not include SAM3 semantic FiLM modulation. | `dsm-sr`, `adapter-guide`, `guide-on`, `no-adj`, `dual-refinement`, `cross-modal-fusion`, `no-semantic-modulation` |
| `DADASR_addguide_noadj_UNet_refinement` | adapter guide: on; adjustment: off; UNet refinement: on | Uses the UNet feature extractor and `LocalRefinementNet` as the core residual recovery path for UNet refinement DSM super-resolution. | `dsm-sr`, `adapter-guide`, `guide-on`, `no-adj`, `unet-refinement`, `local-refinement` |
| `DADASR_noguide_noadj_addrefine` | adapter guide: off; adjustment: off; local refinement: on | Uses RGB and bicubic DSM without the adapter-guide branch, keeps local residual refinement, and runs diffusion without the adjustment step. | `dsm-sr`, `rgb-guide`, `no-adapter-guide`, `no-adj`, `local-refinement`, `real-gdsr-style` |
| `DADASR_nodguide_addadj_norefine` | adapter guide: off; adjustment: on; local refinement: off | Keeps the DADA-style RGB plus bicubic DSM input path, ignores adapter-guide inputs, enables the adjustment path, and does not use the local refinement module. | `dsm-sr`, `rgb-guide`, `no-adapter-guide`, `adj`, `no-refinement`, `dada-style` |

## Directory Layout

```text
.
|-- DADASR_addguide_noadj_addrefine/
|-- DADASR_addguide_noadj_doublerefine/
|-- DADASR_addguide_noadj_crossmodal_nosemantic/
|-- DADASR_addguide_noadj_UNet_refinement/
|-- DADASR_noguide_noadj_addrefine/
`-- DADASR_nodguide_addadj_norefine/
```

Each directory is self-contained and includes its own README, training entry
point, evaluation entry point, model code, data loader, and sample
`ProcessedData_scale10` files.

## Checkpoint Note

The source `DADASR_noguide_noadj_addrefine` directory contained two checkpoint
files that were not committed because they are about 373 MB each and exceed the
normal GitHub blob limit when Git LFS is not available:

- `checkpoint/withguide/best_model.pth`
- `checkpoint/withoutguide/best_model.pth`

To version these weights later, enable Git LFS for the repository and track the
specific checkpoint paths.
