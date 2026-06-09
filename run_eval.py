import argparse
import csv
import os
import pickle
import re
import sys
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from arguments import train_parser
try:
    from data import ProcessedDSMDataset
except ImportError:
    from data.processed_dsm import ProcessedDSMDataset
from losses import get_loss
from model import GADBase


DATA_DIR_CANDIDATES = ("ProcessedData_scale10", "ProcessedData")
SPLIT_TO_DATASET_SPLIT = {
    "vai": "val",
    "test": "test",
}

RGB_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
RGB_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained GADSR checkpoint on either the vai_train_* "
            "validation folders or the test_* folders."
        )
    )
    parser.add_argument("checkpoint_arg", nargs="?", help="Path to checkpoint, e.g. best_model.pth")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Overrides positional checkpoint if set.")
    parser.add_argument(
        "--split",
        choices=("vai", "test"),
        default=None,
        help="Evaluation split. Use 'vai' for vai_train_* folders or 'test' for test_* folders.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: <checkpoint_dir>/eval_<split>")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Evaluation device")

    parser.add_argument("--data-dir", default=None, help="Override data_dir loaded from args.csv")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size loaded from args.csv")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers loaded from args.csv")
    parser.add_argument("--crop-size", type=int, default=None, help="Override HR crop size loaded from args.csv")
    parser.add_argument("--scaling", type=int, default=None, help="Override LR-to-HR scale loaded from args.csv")
    parser.add_argument("--guide-source", choices=("rgb", "albedo"), default=None)
    parser.add_argument("--adapter-guide-dir", default=None, help="Override adapter guide folder root loaded from args.csv")
    parser.add_argument(
        "--feature-extractor",
        default=None,
        choices=("UNet", "none"),
    )
    parser.add_argument("--Npre", type=int, default=None)
    parser.add_argument("--Ntrain", type=int, default=None)
    parser.add_argument("--num-heatmaps", type=int, default=1, help="Number of evaluated samples to save as heatmaps")
    parser.add_argument(
        "--num-fusion-visuals",
        type=int,
        default=1,
        help="Number of evaluated samples to save gated-fusion visualizations. Default: --num-heatmaps.",
    )
    parser.add_argument("--error-percentile", type=float, default=99.0, help="Upper percentile for heatmap color scaling")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    args.checkpoint = args.checkpoint or args.checkpoint_arg
    if args.checkpoint is None:
        parser.error("checkpoint path is required")
    if args.split is None:
        args.split = prompt_split(parser)
    return args


def prompt_split(parser):
    if not sys.stdin.isatty():
        parser.error("--split is required in non-interactive mode. Choose 'vai' or 'test'.")

    while True:
        value = input("Evaluate which split? Enter 'vai' or 'test': ").strip().lower()
        if value in SPLIT_TO_DATASET_SPLIT:
            return value
        print("Invalid split. Please enter 'vai' or 'test'.")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parser_defaults():
    values = {}
    actions = {}
    for action in train_parser._actions:
        if action.dest in (None, "help"):
            continue
        actions[action.dest] = action
        values[action.dest] = None if action.default is argparse.SUPPRESS else action.default
    return values, actions


def coerce_value(raw_value, action, default):
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if value == "" or value.lower() == "none":
        return None
    if isinstance(default, bool):
        return str_to_bool(value)
    if action is not None and action.type is not None:
        return action.type(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def load_args_csv(args_csv_path, defaults, actions):
    loaded = {}
    with open(args_csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            key, raw_value = row[0], row[1]
            if key in defaults:
                loaded[key] = coerce_value(raw_value, actions.get(key), defaults.get(key))
    return loaded


def apply_cli_overrides(train_args, cli_args):
    override_names = (
        "data_dir",
        "batch_size",
        "num_workers",
        "crop_size",
        "scaling",
        "guide_source",
        "adapter_guide_dir",
        "feature_extractor",
        "Npre",
        "Ntrain",
        "use_refinement_net",
        "refinement_channels",
        "refinement_blocks",
        "refinement_only",
    )
    for name in override_names:
        value = getattr(cli_args, name, None)
        if value is not None:
            setattr(train_args, name, value)


def load_train_args(checkpoint_path, cli_args):
    defaults, actions = parser_defaults()
    values = dict(defaults)
    args_csv_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "args.csv")
    notes = []

    if os.path.isfile(args_csv_path):
        values.update(load_args_csv(args_csv_path, defaults, actions))
        loaded_from = args_csv_path
    else:
        loaded_from = "arguments/train.py defaults"
        notes.append(f"args.csv not found beside checkpoint: {args_csv_path}")

    if not values.get("data_dir") or not os.path.isdir(str(values.get("data_dir"))):
        for candidate in DATA_DIR_CANDIDATES:
            if os.path.isdir(candidate):
                values["data_dir"] = candidate
                notes.append(f"data_dir was not usable; auto-detected {candidate}.")
                break

    train_args = SimpleNamespace(**values)
    apply_cli_overrides(train_args, cli_args)
    return train_args, loaded_from, notes


def require_split_dirs(data_dir, split, guide_source=None):
    prefix = "vai_train" if split == "vai" else "test"
    required = (
        f"{prefix}_RGB",
        f"{prefix}_DSM_HR",
        f"{prefix}_DSM_LR",
        f"{prefix}_adapter_guide",
    )
    missing = [name for name in required if not os.path.isdir(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing required {split} folders under {data_dir}: {', '.join(missing)}"
        )


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        print(
            "[Warning] weights_only=True failed while loading the checkpoint. "
            "Retrying with weights_only=False; only use this for checkpoints you trust."
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_from_checkpoint(ckpt):
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a 'model' state_dict.")
    if all(key.startswith("module.") for key in state.keys()):
        state = {key[len("module.") :]: value for key, value in state.items()}
    # Compatibility with older checkpoints that may contain auxiliary keys.
    state.pop("logk2", None)
    state.pop("mean_guide", None)
    state.pop("std_guide", None)
    return state


def adapter_guide_enabled(train_args):
    value = getattr(train_args, "adapter_guide_dir", None)
    return value is not None and str(value).strip() != "" and str(value).strip().lower() != "none"


def to_device(sample, device):
    out = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list):
            out[key] = [v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            out[key] = value
    return out


def build_model(train_args, device):
    model = GADBase(
        feature_extractor=train_args.feature_extractor,
        Npre=train_args.Npre,
        Ntrain=train_args.Ntrain,
        guide_channels=ProcessedDSMDataset.guide_channels,
    )
    return model.to(device)


def build_loader(train_args, split, device):
    require_split_dirs(train_args.data_dir, split, getattr(train_args, "guide_source", None))
    dataset = ProcessedDSMDataset(
        train_args.data_dir,
        split=SPLIT_TO_DATASET_SPLIT[split],
        crop_size=train_args.crop_size,
        scaling=train_args.scaling,
        in_memory=getattr(train_args, "in_memory", False),
        max_rotation_angle=0.0,
        do_horizontal_flip=False,
        crop_deterministic=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    return dataset, loader


def sample_name(sample, batch_index, fallback_index):
    keys = sample.get("sample_key")
    if isinstance(keys, (list, tuple)):
        return safe_name(keys[batch_index])
    if isinstance(keys, str):
        return safe_name(keys)

    ids = sample.get("id")
    if torch.is_tensor(ids):
        return safe_name(f"sample_{int(ids[batch_index].item()):06d}")
    if isinstance(ids, (list, tuple)):
        return safe_name(f"sample_{int(ids[batch_index]):06d}" if str(ids[batch_index]).isdigit() else ids[batch_index])
    if isinstance(ids, (int, np.integer)):
        return safe_name(f"sample_{int(ids):06d}")

    return f"sample_{fallback_index:04d}"


def safe_name(name):
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "sample"


def compute_metrics_arrays(pred, gt, mask):
    valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
    if not valid.any():
        return None

    diff = pred[valid] - gt[valid]
    abs_diff = np.abs(diff)
    sq_diff = diff**2
    median_diff = np.median(diff)
    return {
        "count": int(valid.sum()),
        "mse": float(sq_diff.mean()),
        "rmse": float(np.sqrt(sq_diff.mean())),
        "mae": float(abs_diff.mean()),
        "medae": float(np.median(abs_diff)),
        "nmad": float(1.4826 * np.median(np.abs(diff - median_diff))),
        "rmae": float((abs_diff / (np.abs(gt[valid]) + 1e-6)).mean() * 100.0),
        "max_abs_error": float(abs_diff.max()),
        "p95_abs_error": float(np.percentile(abs_diff, 95.0)),
    }


def aggregate_metrics(rows):
    summary = {}
    for key, name in (
        ("rmse", "RMSE"),
        ("mse", "MSE"),
        ("mae", "MAE"),
        ("medae", "MedAE"),
        ("nmad", "NMAD"),
        ("rmae", "RMAE"),
        ("slope_rmse", "SlopeRMSE"),
        ("max_abs_error", "MaxAbsError"),
        ("p95_abs_error", "P95AbsError"),
    ):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[f"{name}_mean"] = float(values.mean()) if values.size else np.nan
        summary[f"{name}_std"] = float(values.std()) if values.size else np.nan
    return summary


def compute_slope(dsm):
    valid = np.isfinite(dsm)
    fill_value = float(np.nanmedian(dsm[valid])) if valid.any() else 0.0
    dsm = np.where(valid, dsm, fill_value).astype(np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
    dsm_t = torch.from_numpy(dsm)[None, None]
    gx = F.conv2d(dsm_t, torch.from_numpy(kx)[None, None], padding=1)
    gy = F.conv2d(dsm_t, torch.from_numpy(ky)[None, None], padding=1)
    return torch.sqrt(gx**2 + gy**2).squeeze().numpy()


def slope_rmse(pred, gt, mask):
    valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
    if not valid.any():
        return np.nan
    slope_pred = compute_slope(pred)
    slope_gt = compute_slope(gt)
    return float(np.sqrt(np.mean((slope_pred[valid] - slope_gt[valid]) ** 2)))


def masked_array(values, valid):
    arr = values.astype(np.float32).copy()
    arr[~valid] = np.nan
    return np.ma.masked_invalid(arr)


def positive_vmax(values, valid, percentile):
    vals = values[valid & np.isfinite(values)]
    if vals.size == 0:
        return 1.0
    vmax = float(np.percentile(vals, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(vals))
    return vmax if vmax > 0 else 1.0


def robust_depth_limits(arrays, valid):
    vals = []
    for arr in arrays:
        cur = arr[valid & np.isfinite(arr)]
        if cur.size > 0:
            vals.append(cur)
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    vmin = float(np.percentile(vals, 2.0))
    vmax = float(np.percentile(vals, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def add_colorbar(fig, ax, image, label):
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(label, fontsize=8)


def gate_array_from_output(output, batch_index):
    gates = output.get("fusion_gates")
    modalities = output.get("fusion_gate_modalities")
    if gates is None or modalities is None:
        return None, None

    if isinstance(gates, dict):
        gate_maps = {
            scale: value[batch_index].detach().cpu().numpy()
            for scale, value in gates.items()
        }
    else:
        gate_maps = {"fusion": gates[batch_index].detach().cpu().numpy()}
    return gate_maps, list(modalities)


def save_fusion_visualization(path, rgb, gt, pred, bicubic, mask, gate_maps, modalities, title, dpi):
    valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
    depth_vmin, depth_vmax = robust_depth_limits([gt, pred, bicubic], valid)
    scales = list(gate_maps.keys())
    n_rows = 1 + len(scales)
    n_cols = max(4, len(modalities))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=11)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("HR RGB")

    im = axes[0, 1].imshow(masked_array(gt, valid), cmap="terrain", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 1].set_title("HR DSM")
    add_colorbar(fig, axes[0, 1], im, "DSM")

    im = axes[0, 2].imshow(masked_array(pred, valid), cmap="terrain", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 2].set_title("Pred DSM")
    add_colorbar(fig, axes[0, 2], im, "DSM")

    im = axes[0, 3].imshow(masked_array(bicubic, valid), cmap="terrain", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 3].set_title("Bicubic DSM")
    add_colorbar(fig, axes[0, 3], im, "DSM")

    for col in range(4, n_cols):
        axes[0, col].axis("off")

    for row, scale in enumerate(scales, start=1):
        scale_gates = gate_maps[scale]
        for col, modality in enumerate(modalities):
            im = axes[row, col].imshow(scale_gates[col], cmap="viridis", vmin=0.0, vmax=1.0)
            axes[row, col].set_title(f"{scale} gate: {modality}")
            add_colorbar(fig, axes[row, col], im, "weight")
        for col in range(len(modalities), n_cols):
            axes[row, col].axis("off")

    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_error_heatmap(path, rgb, gt, pred, bicubic, mask, title, metrics, error_percentile, dpi):
    valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
    diff = pred - gt
    abs_error = np.abs(diff)
    pixel_mse = diff**2
    depth_vmin, depth_vmax = robust_depth_limits([gt, pred, bicubic], valid)
    abs_vmax = positive_vmax(abs_error, valid, error_percentile)
    mse_vmax = positive_vmax(pixel_mse, valid, error_percentile)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(
        f"{title} | RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.4f}",
        fontsize=11,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")

    im = axes[0, 1].imshow(masked_array(bicubic, valid), cmap="terrain", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 1].set_title("Bicubic DSM")
    add_colorbar(fig, axes[0, 1], im, "DSM")

    im = axes[0, 2].imshow(masked_array(pred, valid), cmap="terrain", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 2].set_title("Pred DSM")
    add_colorbar(fig, axes[0, 2], im, "DSM")

    im = axes[0, 3].imshow(masked_array(gt, valid), cmap="terrain", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 3].set_title("GT DSM")
    add_colorbar(fig, axes[0, 3], im, "DSM")

    im = axes[1, 0].imshow(masked_array(abs_error, valid), cmap="magma", vmin=0.0, vmax=abs_vmax)
    axes[1, 0].set_title("|Pred - GT|")
    add_colorbar(fig, axes[1, 0], im, "Absolute error")

    im = axes[1, 1].imshow(masked_array(pixel_mse, valid), cmap="inferno", vmin=0.0, vmax=mse_vmax)
    axes[1, 1].set_title("Pixel MSE")
    add_colorbar(fig, axes[1, 1], im, "Squared error")

    axes[1, 2].imshow(rgb)
    axes[1, 2].imshow(masked_array(pixel_mse, valid), cmap="inferno", vmin=0.0, vmax=mse_vmax, alpha=0.55)
    axes[1, 2].set_title("Pixel MSE overlay")

    axes[1, 3].axis("off")

    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_summary_csv(path, bic_summary, our_summary):
    header = [
        "Method",
        "RMSE_mean",
        "RMSE_std",
        "MSE_mean",
        "MSE_std",
        "MAE_mean",
        "MAE_std",
        "MedAE_mean",
        "MedAE_std",
        "NMAD_mean",
        "NMAD_std",
        "RMAE_mean",
        "RMAE_std",
        "SlopeRMSE_mean",
        "SlopeRMSE_std",
        "MaxAbsError_mean",
        "MaxAbsError_std",
        "P95AbsError_mean",
        "P95AbsError_std",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for method, summary in (("Bicubic", bic_summary), ("Ours", our_summary)):
            writer.writerow([method] + [summary[col] for col in header[1:]])


def write_per_image_csv(path, rows):
    fieldnames = [
        "index",
        "name",
        "valid_pixels",
        "ours_rmse",
        "ours_mse",
        "ours_mae",
        "ours_medae",
        "ours_nmad",
        "ours_rmae",
        "ours_slope_rmse",
        "bicubic_rmse",
        "bicubic_mse",
        "bicubic_mae",
        "bicubic_medae",
        "bicubic_nmad",
        "bicubic_rmae",
        "bicubic_slope_rmse",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def batch_rgb_to_numpy(sample):
    if "rgb" in sample:
        return sample["rgb"].detach().cpu().permute(0, 2, 3, 1).numpy().clip(0.0, 1.0)

    guide = sample["guide"][:, :3].detach().cpu().permute(0, 2, 3, 1).numpy()
    return np.clip(guide * RGB_STD_NP + RGB_MEAN_NP, 0.0, 1.0)


@torch.no_grad()
def main():
    cli_args = parse_args()
    if cli_args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not os.path.isfile(cli_args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {cli_args.checkpoint}")

    device = torch.device(cli_args.device)
    train_args, train_args_source, notes = load_train_args(cli_args.checkpoint, cli_args)
    for name, value in (
        ("guide_source", "rgb"),
        ("adapter_guide_dir", None),
        ("use_refinement_net", False),
        ("refinement_only", False),
        ("in_memory", False),
    ):
        if not hasattr(train_args, name):
            setattr(train_args, name, value)
    out_dir = cli_args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(cli_args.checkpoint)),
        f"eval_{cli_args.split}",
    )
    heatmap_dir = os.path.join(out_dir, "error_heatmaps")
    fusion_dir = os.path.join(out_dir, "fusion_visualizations")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(fusion_dir, exist_ok=True)
    num_fusion_visuals = (
        cli_args.num_heatmaps
        if cli_args.num_fusion_visuals is None
        else cli_args.num_fusion_visuals
    )

    print(f"[INFO] Checkpoint: {cli_args.checkpoint}")
    print(f"[INFO] Training args: {train_args_source}")
    for note in notes:
        print(f"[INFO] {note}")
    print(
        "[INFO] Eval config: "
        f"data_dir={train_args.data_dir}, split={cli_args.split}, "
        f"crop_size={train_args.crop_size}, scaling={train_args.scaling}, "
        f"batch_size={train_args.batch_size}, guide_source={train_args.guide_source}, "
        f"adapter_guide_dir={getattr(train_args, 'adapter_guide_dir', None)}, "
        f"feature_extractor={train_args.feature_extractor}, "
        f"Npre={train_args.Npre}, Ntrain={train_args.Ntrain} "
    )

    ckpt = safe_torch_load(cli_args.checkpoint, map_location=device)
    state = state_dict_from_checkpoint(ckpt)

    dataset, dataloader = build_loader(train_args, cli_args.split, device)
    print(f"[INFO] Samples: {len(dataset)}")

    model = build_model(train_args, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    our_rows = []
    bic_rows = []
    per_image_rows = []
    loss_stats = {}
    heatmaps_saved = 0
    fusion_visuals_saved = 0
    sample_index = 0
    since = time.time()

    for sample in tqdm(dataloader, desc="Evaluating", unit="batch", dynamic_ncols=True):
        sample_device = to_device(sample, device)
        output = model(sample_device, train=False)
        _, loss_dict = get_loss(output, sample_device)
        for key, value in loss_dict.items():
            loss_stats[key] = loss_stats.get(key, 0.0) + (
                value.detach().item() if torch.is_tensor(value) else float(value)
            )

        preds = output["y_pred"].detach().cpu().numpy()
        bics = sample["y_bicubic"].detach().cpu().numpy()
        gts = sample["y"].detach().cpu().numpy()
        masks = sample["mask_hr"].detach().cpu().numpy()
        rgbs = batch_rgb_to_numpy(sample)

        for batch_index in range(preds.shape[0]):
            pred = preds[batch_index, 0]
            bic = bics[batch_index, 0]
            gt = gts[batch_index, 0]
            mask = masks[batch_index, 0]
            rgb = rgbs[batch_index]
            image_name = sample_name(sample, batch_index, sample_index)

            our_metrics = compute_metrics_arrays(pred, gt, mask)
            bic_metrics = compute_metrics_arrays(bic, gt, mask)
            if our_metrics is None or bic_metrics is None:
                sample_index += 1
                continue

            our_metrics["slope_rmse"] = slope_rmse(pred, gt, mask)
            bic_metrics["slope_rmse"] = slope_rmse(bic, gt, mask)
            our_rows.append(our_metrics)
            bic_rows.append(bic_metrics)

            per_image_rows.append(
                {
                    "index": sample_index,
                    "name": image_name,
                    "valid_pixels": our_metrics["count"],
                    "ours_rmse": our_metrics["rmse"],
                    "ours_mse": our_metrics["mse"],
                    "ours_mae": our_metrics["mae"],
                    "ours_medae": our_metrics["medae"],
                    "ours_nmad": our_metrics["nmad"],
                    "ours_rmae": our_metrics["rmae"],
                    "ours_slope_rmse": our_metrics["slope_rmse"],
                    "bicubic_rmse": bic_metrics["rmse"],
                    "bicubic_mse": bic_metrics["mse"],
                    "bicubic_mae": bic_metrics["mae"],
                    "bicubic_medae": bic_metrics["medae"],
                    "bicubic_nmad": bic_metrics["nmad"],
                    "bicubic_rmae": bic_metrics["rmae"],
                    "bicubic_slope_rmse": bic_metrics["slope_rmse"],
                }
            )

            if heatmaps_saved < cli_args.num_heatmaps:
                heatmap_path = os.path.join(
                    heatmap_dir,
                    f"{cli_args.split}_{sample_index:04d}_{image_name}_error_heatmap.png",
                )
                save_error_heatmap(
                    heatmap_path,
                    rgb,
                    gt,
                    pred,
                    bic,
                    mask,
                    title=f"{cli_args.split}_{sample_index:04d}_{image_name}",
                    metrics=our_metrics,
                    error_percentile=cli_args.error_percentile,
                    dpi=cli_args.dpi,
                )
                heatmaps_saved += 1

            if fusion_visuals_saved < num_fusion_visuals:
                gate_maps, modalities = gate_array_from_output(output, batch_index)
                if gate_maps is not None and modalities is not None:
                    fusion_path = os.path.join(
                        fusion_dir,
                        f"{cli_args.split}_{sample_index:04d}_{image_name}_fusion.png",
                    )
                    save_fusion_visualization(
                        fusion_path,
                        rgb,
                        gt,
                        pred,
                        bic,
                        mask,
                        gate_maps,
                        modalities,
                        title=f"{cli_args.split}_{sample_index:04d}_{image_name} gated fusion",
                        dpi=cli_args.dpi,
                    )
                    fusion_visuals_saved += 1

            sample_index += 1

    elapsed = time.time() - since
    num_batches = max(1, len(dataloader))
    loss_stats = {key: value / num_batches for key, value in loss_stats.items()}
    our_summary = aggregate_metrics(our_rows)
    bic_summary = aggregate_metrics(bic_rows)

    summary_path = os.path.join(out_dir, "metrics_summary.csv")
    per_image_path = os.path.join(out_dir, "metrics_per_image.csv")
    write_summary_csv(summary_path, bic_summary, our_summary)
    write_per_image_csv(per_image_path, per_image_rows)

    print("\n================ DSM SR Evaluation ================")
    print(f"Samples evaluated: {len(our_rows)}")
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))
    if loss_stats:
        print("Training-style loss: " + ", ".join(f"{key}={value:.6f}" for key, value in loss_stats.items()))
    for method, summary in (("Bicubic", bic_summary), ("Ours", our_summary)):
        print(f"\n[ {method} ]")
        print(f"RMSE:       {summary['RMSE_mean']:.6f} +/- {summary['RMSE_std']:.6f}")
        print(f"MSE:        {summary['MSE_mean']:.8f} +/- {summary['MSE_std']:.8f}")
        print(f"MAE:        {summary['MAE_mean']:.6f} +/- {summary['MAE_std']:.6f}")
        print(f"MedAE:      {summary['MedAE_mean']:.6f} +/- {summary['MedAE_std']:.6f}")
        print(f"NMAD:       {summary['NMAD_mean']:.6f} +/- {summary['NMAD_std']:.6f}")
        print(f"RMAE (%):   {summary['RMAE_mean']:.6f} +/- {summary['RMAE_std']:.6f}")
        print(f"Slope RMSE: {summary['SlopeRMSE_mean']:.6f} +/- {summary['SlopeRMSE_std']:.6f}")
    print("==================================================")
    print(f"[DONE] Summary metrics: {summary_path}")
    print(f"[DONE] Per-image metrics: {per_image_path}")
    print(f"[DONE] Error heatmaps saved: {heatmaps_saved} -> {heatmap_dir}")
    print(f"[DONE] Fusion visualizations saved: {fusion_visuals_saved} -> {fusion_dir}")


if __name__ == "__main__":
    main()
