# DADASR DSM Super-Resolution Versions

This repository keeps three DADASR variants for DSM super-resolution on
`ProcessedData_scale10`. The previous single-version repository contents were
replaced by these versioned directories.

## Version Index

| Version | Module configuration | Description | Tags |
| --- | --- | --- | --- |
| `DADASR_addguide_noadj_addrefine` | adapter guide: on; adjustment: off; local refinement: on | Uses RGB plus `adapter_guide` as guide features, applies local residual refinement, then runs the anisotropic diffusion loop without the adjustment step. | `dsm-sr`, `adapter-guide`, `guide-on`, `no-adj`, `local-refinement`, `rmse` |
| `DADASR_noguide_noadj_addrefine` | adapter guide: off; adjustment: off; local refinement: on | Uses RGB and bicubic DSM without the adapter-guide branch, keeps local residual refinement, and runs diffusion without the adjustment step. | `dsm-sr`, `rgb-guide`, `no-adapter-guide`, `no-adj`, `local-refinement`, `real-gdsr-style` |
| `DADASR_nodguide_addadj_norefine` | adapter guide: off; adjustment: on; local refinement: off | Keeps the DADA-style RGB plus bicubic DSM input path, ignores adapter-guide inputs, enables the adjustment path, and does not use the local refinement module. | `dsm-sr`, `rgb-guide`, `no-adapter-guide`, `adj`, `no-refinement`, `dada-style` |

## Directory Layout

```text
.
|-- DADASR_addguide_noadj_addrefine/
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
