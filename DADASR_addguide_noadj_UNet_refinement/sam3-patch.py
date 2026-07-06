#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
SAM3 遥感 (RGB .tif) 关键词语义分割批量脚本（仅批量模式）

自动遍历:
  ProcessedData/pos_train_RGB
  ProcessedData/vai_train_RGB
  ProcessedData/test_RGB

输出目录结构:
  ProcessedData/pos_train_label/             -> label_*.tif (单波段语义类别标签)
  ProcessedData/pos_train_semantic_boundary/ -> boundary_*.tif (类别边界，0/1)
  ProcessedData/pos_train_SAM3/              -> SAM3_*.tif (SAM3 overlay RGB，可选 QA 输出)
  ProcessedData/vai_train_label/
  ProcessedData/vai_train_SAM3/
  ProcessedData/test_label/
  ProcessedData/test_SAM3/

标签编码（固定类别顺序）:
  0 = background / no prompt
  1 = building
  2 = road
  255 = overlap / ignore

主要加速策略:
  1) 每张图只调用一次 processor.set_image，多个 prompt 复用图像 state
  2) 默认使用 rasterio 读取 RGB TIFF，支持 LZW 等压缩格式，不依赖 imagecodecs
  3) 同时保存类别 label 和类别边界；SAM3 overlay 仅用于人工 QA，不作为模型语义输入
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence

import numpy as np
import torch
from PIL import Image
import rasterio
from rasterio.transform import Affine
from tifffile import imwrite as tif_imwrite


SEMANTIC_CLASSES = (
    ("building", "building, rooftop"),
    ("road", "road, paved surface, impervious surface"),
)
DEFAULT_PROMPTS = [prompt for _, prompt in SEMANTIC_CLASSES]
DEFAULT_CLASS_NAMES = [name for name, _ in SEMANTIC_CLASSES]
OVERLAP_LABEL = 255


# ----------------- 基础工具 -----------------

def _percentile_to_uint8(rgb: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """将任意 dtype 的 RGB(H,W,3) 拉伸到 uint8。"""
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    out = np.empty_like(rgb, dtype=np.uint8)

    for c in range(3):
        band = rgb[..., c].astype(np.float32)
        lo = np.percentile(band, p_low)
        hi = np.percentile(band, p_high)
        if hi <= lo:
            out[..., c] = np.clip(band, 0, 255).astype(np.uint8)
            continue
        band = (band - lo) / (hi - lo)
        band = np.clip(band, 0.0, 1.0) * 255.0
        out[..., c] = band.astype(np.uint8)
    return out


def _to_hwc_rgb(arr: np.ndarray, bands: Tuple[int, int, int] = (1, 2, 3)) -> np.ndarray:
    """Normalize TIFF arrays to RGB(H,W,3). Supports HWC and CHW layouts."""
    if arr.ndim == 2:
        raise ValueError("Expected RGB TIFF, got single-band image")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RGB TIFF array, got shape={arr.shape}")

    if arr.shape[-1] in (3, 4):
        return arr[..., [b - 1 for b in bands]]

    if arr.shape[0] in (3, 4):
        return np.transpose(arr[[b - 1 for b in bands], :, :], (1, 2, 0))

    raise ValueError(f"Cannot interpret TIFF as RGB, shape={arr.shape}")


def _rgb_to_model_uint8(rgb: np.ndarray, stretch: bool) -> np.ndarray:
    """Prepare RGB for SAM3. Fast path keeps already uint8 tiles unchanged."""
    if rgb.dtype == np.uint8 and not stretch:
        return np.ascontiguousarray(rgb)

    if stretch:
        return _percentile_to_uint8(rgb)

    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8, copy=False))


def read_rgb_tif(
    path: str,
    bands: Tuple[int, int, int] = (1, 2, 3),
    preserve_geo: bool = False,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    使用 rasterio 读取 tif / geotiff，返回 RGB(H,W,3) 和 geo 元数据。

    这里默认不再使用 tifffile.imread，因此 LZW/Deflate/JPEG 等压缩 TIFF
    不需要额外安装 imagecodecs。
    """
    with rasterio.open(path) as src:
        rgb = src.read(list(bands))
        meta = src.meta.copy()
        rgb = np.transpose(rgb, (1, 2, 0))

        if not preserve_geo:
            return rgb, None

        geo = {
            "crs": src.crs,
            "transform": src.transform,
            "meta": meta,
            "width": src.width,
            "height": src.height,
        }

    return rgb, geo


def maybe_resize(rgb_u8: np.ndarray, max_side: Optional[int]) -> Tuple[np.ndarray, float, float]:
    """可选缩放到较小尺寸以加速推理。返回 (resized, sx, sy)。"""
    if max_side is None:
        return rgb_u8, 1.0, 1.0

    h, w = rgb_u8.shape[:2]
    if max(h, w) <= max_side:
        return rgb_u8, 1.0, 1.0

    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pil = Image.fromarray(rgb_u8)
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    resized = np.array(pil)

    sx = w / float(new_w)
    sy = h / float(new_h)
    return resized, sx, sy


# ----------------- SAM3 推理接口 -----------------

@torch.inference_mode()
def segment_from_state(
    processor: Sam3Processor,
    state,
    prompt: str,
    score_thr: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """对单个 prompt 进行分割，返回 union mask (H,W) bool。"""
    out = processor.set_text_prompt(state=state, prompt=prompt)

    masks = out.get("masks", None)
    scores = out.get("scores", None)

    if masks is None or scores is None or masks.numel() == 0:
        return np.zeros(output_size, dtype=bool)

    scores = scores.detach().float().cpu().numpy()
    keep = scores >= float(score_thr)
    if keep.sum() == 0:
        return np.zeros(output_size, dtype=bool)

    m = masks.detach().cpu().numpy().astype(bool)  # (N,1,H,W)
    if m.ndim == 4:
        m = m[:, 0, :, :]
    m = m[keep]  # (K,H,W)

    union = np.any(m, axis=0)
    return union


@torch.inference_mode()
def segment_with_prompt(
    processor: Sam3Processor,
    pil_img: Image.Image,
    prompt: str,
    score_thr: float,
) -> np.ndarray:
    """兼容旧逻辑：每个 prompt 单独 set_image。仅在关闭 state 复用时使用。"""
    state = processor.set_image(pil_img)
    return segment_from_state(
        processor,
        state=state,
        prompt=prompt,
        score_thr=score_thr,
        output_size=(pil_img.height, pil_img.width),
    )


# ----------------- 可视化与 IO -----------------

def _color_map(num_classes: int) -> List[np.ndarray]:
    base_colors = [
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 128, 0],
        [128, 0, 255],
    ]
    return [
        np.array(base_colors[i % len(base_colors)], dtype=np.float32)
        for i in range(num_classes)
    ]


def overlay_vis(rgb_u8: np.ndarray,
                masks: Sequence[np.ndarray],
                alpha: float = 0.45) -> np.ndarray:
    """生成多 prompt 的 SAM3 overlay RGB 图。"""
    vis = rgb_u8.astype(np.float32).copy()

    for mask, color in zip(masks, _color_map(len(masks))):
        mask = mask.astype(bool)
        vis[mask] = (1 - alpha) * vis[mask] + alpha * color

    return np.clip(vis, 0, 255).astype(np.uint8)


def build_label(masks: Sequence[np.ndarray]) -> np.ndarray:
    """合成单通道 label：0=背景，1..N=prompt 序号，255=重叠。"""
    if len(masks) == 0:
        raise ValueError("masks 不能为空")

    h, w = masks[0].shape
    overlap_count = np.zeros((h, w), dtype=np.uint16)
    for mask in masks:
        if mask.shape != (h, w):
            raise ValueError(f"mask shape 不一致: expected {(h, w)}, got {mask.shape}")
        overlap_count += mask.astype(np.uint16)

    label = np.zeros((h, w), dtype=np.uint8)
    for class_id, mask in enumerate(masks, start=1):
        if class_id >= OVERLAP_LABEL:
            raise ValueError("prompt 数量过多，无法用 uint8 label 保存")
        label[mask & (overlap_count == 1)] = class_id
    label[overlap_count > 1] = OVERLAP_LABEL
    return label


def build_semantic_boundary(label: np.ndarray) -> np.ndarray:
    """Return a one-pixel categorical boundary map (0/1).

    Adjacent class IDs are compared for equality rather than by their numeric
    distance, so the boundary strength is independent of the arbitrary label
    encoding. Overlap pixels (255) are ignored.
    """
    if label.ndim != 2:
        raise ValueError(f"Expected a 2D label map, got shape={label.shape}")

    valid = label != OVERLAP_LABEL
    boundary = np.zeros_like(label, dtype=bool)

    horizontal_change = (
        valid[:, 1:]
        & valid[:, :-1]
        & (label[:, 1:] != label[:, :-1])
    )
    vertical_change = (
        valid[1:, :]
        & valid[:-1, :]
        & (label[1:, :] != label[:-1, :])
    )
    boundary[:, 1:] |= horizontal_change
    boundary[:, :-1] |= horizontal_change
    boundary[1:, :] |= vertical_change
    boundary[:-1, :] |= vertical_change
    return boundary.astype(np.uint8)


def validate_label_schema(label: np.ndarray) -> None:
    """Reject labels whose IDs do not match the active semantic schema."""
    max_class_id = len(SEMANTIC_CLASSES)
    valid = (label <= max_class_id) | (label == OVERLAP_LABEL)
    if valid.all():
        return
    unexpected = np.unique(label[~valid]).tolist()
    raise ValueError(
        f"Label contains IDs {unexpected} outside the active 0..{max_class_id} "
        f"semantic schema. Regenerate all labels with the current prompts before "
        "building semantic boundaries."
    )


def write_label_geotiff(
    out_path: str,
    label: np.ndarray,
    geo: Optional[Dict],
    sx: float,
    sy: float,
):
    """Output a single-band uint8 label or categorical-boundary GeoTIFF."""
    out_path = str(out_path)
    os.makedirs(str(Path(out_path).parent), exist_ok=True)

    if geo is None:
        tif_imwrite(out_path, label.astype(np.uint8, copy=False), metadata=None)
        return
    else:
        transform = geo["transform"]
        crs = geo["crs"]
        meta = geo["meta"].copy()

        if (sx != 1.0) or (sy != 1.0):
            transform = Affine(transform.a * sx, transform.b, transform.c,
                               transform.d, transform.e * sy, transform.f)

        meta.update(
            driver="GTiff",
            height=label.shape[0],
            width=label.shape[1],
            count=1,
            dtype=rasterio.uint8,
            transform=transform,
            crs=crs,
        )

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(label.astype(np.uint8), 1)


def write_rgb_geotiff(
    out_path: str,
    rgb_u8: np.ndarray,
    geo: Optional[Dict],
    sx: float,
    sy: float,
):
    """输出三波段 uint8 SAM3 overlay RGB TIFF。"""
    out_path = str(out_path)
    os.makedirs(str(Path(out_path).parent), exist_ok=True)

    if geo is None:
        tif_imwrite(out_path, rgb_u8.astype(np.uint8, copy=False), metadata=None)
        return

    transform = geo["transform"]
    crs = geo["crs"]
    meta = geo["meta"].copy()

    if (sx != 1.0) or (sy != 1.0):
        transform = Affine(transform.a * sx, transform.b, transform.c,
                           transform.d, transform.e * sy, transform.f)

    meta.update(
        driver="GTiff",
        height=rgb_u8.shape[0],
        width=rgb_u8.shape[1],
        count=3,
        dtype=rasterio.uint8,
        transform=transform,
        crs=crs,
    )
    meta.pop("nodata", None)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(np.transpose(rgb_u8.astype(np.uint8), (2, 0, 1)))


# ----------------- 批量处理 -----------------

def _find_rgb_images(rgb_dir: str) -> List[str]:
    exts = (".tif", ".tiff")
    return sorted([
        os.path.join(rgb_dir, f)
        for f in os.listdir(rgb_dir)
        if f.lower().endswith(exts)
    ])


def _find_label_images(label_dir: str) -> List[str]:
    exts = (".tif", ".tiff")
    return sorted([
        os.path.join(label_dir, f)
        for f in os.listdir(label_dir)
        if f.lower().endswith(exts)
    ])


def _ensure_dirs(root: str, split_name: str, save_overlay: bool, save_boundary: bool):
    label_dir = os.path.join(root, f"{split_name}_label")
    sam3_dir = os.path.join(root, f"{split_name}_SAM3")
    boundary_dir = os.path.join(root, f"{split_name}_semantic_boundary")
    os.makedirs(label_dir, exist_ok=True)
    if save_overlay:
        os.makedirs(sam3_dir, exist_ok=True)
    if save_boundary:
        os.makedirs(boundary_dir, exist_ok=True)
    return label_dir, sam3_dir, boundary_dir


def _output_stems(rgb_path: str) -> Tuple[str, str, str]:
    stem = Path(rgb_path).stem
    if stem.upper().startswith("RGB_"):
        suffix = stem[4:]
        return f"label_{suffix}", f"SAM3_{suffix}", f"boundary_{suffix}"
    return f"{stem}_label", f"{stem}_SAM3", f"{stem}_boundary"


def _boundary_stem_from_label(label_path: str) -> str:
    stem = Path(label_path).stem
    if stem.lower().startswith("label_"):
        return f"boundary_{stem[6:]}"
    return f"{stem}_boundary"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _state_for_prompt(state):
    """Use a shallow copy when SAM3 state is dict-like to avoid prompt metadata carryover."""
    if not isinstance(state, dict):
        return state
    cloned = state.copy()
    if isinstance(cloned.get("backbone_out"), dict):
        cloned["backbone_out"] = cloned["backbone_out"].copy()
    return cloned


def build_boundaries_from_labels(root: str, split_name: str, overwrite: bool):
    """Create categorical boundary TIFFs from existing semantic labels only."""
    label_dir = os.path.join(root, f"{split_name}_label")
    if not os.path.isdir(label_dir):
        print(f"[WARN] 标签目录不存在: {label_dir}")
        return

    boundary_dir = os.path.join(root, f"{split_name}_semantic_boundary")
    os.makedirs(boundary_dir, exist_ok=True)
    label_paths = _find_label_images(label_dir)
    if not label_paths:
        print(f"[WARN] 空标签目录: {label_dir}")
        return

    for index, label_path in enumerate(label_paths, start=1):
        out_path = os.path.join(
            boundary_dir,
            _boundary_stem_from_label(label_path) + ".tif",
        )
        if not overwrite and os.path.exists(out_path):
            print(f"[SKIP] 已存在: {Path(out_path).name}")
            continue

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.uint8, copy=False)
            geo = {
                "crs": src.crs,
                "transform": src.transform,
                "meta": src.meta.copy(),
                "width": src.width,
                "height": src.height,
            }

        validate_label_schema(label)
        boundary = build_semantic_boundary(label)
        write_label_geotiff(out_path, boundary, geo, sx=1.0, sy=1.0)
        print(f"[{index:04d}/{len(label_paths):04d}] 生成: {out_path}")


def run_dir(rgb_dir: str, root: str, split_name: str, processor: Sam3Processor, args):
    files = _find_rgb_images(rgb_dir)
    if len(files) == 0:
        print(f"[WARN] 空目录: {rgb_dir}")
        return

    label_dir, sam3_dir, boundary_dir = _ensure_dirs(
        root,
        split_name,
        args.save_overlay,
        args.save_boundary,
    )
    files = files[args.start_index:]
    if args.max_images is not None:
        files = files[:args.max_images]

    for i, rgb_path in enumerate(files):
        t0 = time.perf_counter()
        label_stem, overlay_stem, boundary_stem = _output_stems(rgb_path)
        out_label_tif = os.path.join(label_dir, label_stem + ".tif")
        out_overlay_tif = os.path.join(sam3_dir, overlay_stem + ".tif")
        out_boundary_tif = os.path.join(boundary_dir, boundary_stem + ".tif")

        required_outputs = [out_label_tif]
        if args.save_overlay:
            required_outputs.append(out_overlay_tif)
        if args.save_boundary:
            required_outputs.append(out_boundary_tif)

        if (not args.overwrite) and all(os.path.exists(p) for p in required_outputs):
            print(f"[SKIP] 已存在: {Path(rgb_path).stem}")
            continue

        # 1) 读 tif
        rgb, geo = read_rgb_tif(rgb_path, preserve_geo=args.preserve_geo)
        rgb_u8 = _rgb_to_model_uint8(rgb, stretch=args.stretch)

        # 2) resize
        rgb_u8_r, sx, sy = maybe_resize(rgb_u8, args.max_side)
        pil = Image.fromarray(rgb_u8_r)

        # 3) SAM3 推理
        masks: List[np.ndarray] = []
        if args.reuse_image_state:
            state = processor.set_image(pil)
            output_size = (pil.height, pil.width)
            for prompt in args.prompts:
                masks.append(
                    segment_from_state(
                        processor,
                        _state_for_prompt(state),
                        prompt,
                        args.score_thr,
                        output_size,
                    )
                )
        else:
            for prompt in args.prompts:
                masks.append(segment_with_prompt(processor, pil, prompt, args.score_thr))

        # 4) 合成 label
        label = build_label(masks)

        # 5) Save semantic artifacts. The boundary is derived from categorical
        # labels, never from the overlay RGB visualization.
        write_label_geotiff(out_label_tif, label, geo, sx=sx, sy=sy)
        if args.save_boundary:
            boundary = build_semantic_boundary(label)
            write_label_geotiff(out_boundary_tif, boundary, geo, sx=sx, sy=sy)

        if args.save_overlay:
            overlay = overlay_vis(rgb_u8_r, masks, alpha=args.overlay_alpha)
            write_rgb_geotiff(out_overlay_tif, overlay, geo, sx=sx, sy=sy)

        print(f"[{i+1:04d}/{len(files):04d}] "
              f"生成: {out_label_tif}"
              f"{' | ' + out_boundary_tif if args.save_boundary else ''}"
              f"{' | ' + out_overlay_tif if args.save_overlay else ''}"
              f" | {time.perf_counter() - t0:.2f}s")


def write_label_map(root: str, prompts: Sequence[str], class_names: Sequence[str]):
    """Write human- and machine-readable class schemas beside the outputs."""
    txt_path = os.path.join(root, "sam3_label_map.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("0 = background\n")
        for idx, prompt in enumerate(prompts, start=1):
            f.write(f"{idx} = {prompt}\n")
        f.write(f"{OVERLAP_LABEL} = overlap\n")

    schema = {
        "schema_version": 1,
        "background": {"id": 0, "name": "background_or_no_prompt"},
        "classes": [
            {
                "id": index,
                "name": class_names[index - 1],
                "prompt": prompt,
            }
            for index, prompt in enumerate(prompts, start=1)
        ],
        "ignore": {"id": OVERLAP_LABEL, "name": "overlap"},
        "boundary": {
            "folder_suffix": "semantic_boundary",
            "encoding": "uint8: 0=interior, 1=class boundary",
        },
    }
    json_path = os.path.join(root, "sam3_label_map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return txt_path, json_path


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="ProcessedData 根目录")
    ap.add_argument("--ckpt", default="./sam3ckpt/sam3.pt", help="本地权重路径")
    ap.add_argument("--device", default=None, help="默认自动选择 cuda -> mps -> cpu")
    ap.add_argument("--score_thr", type=float, default=0.20, help="置信度阈值")
    ap.add_argument("--max_side", type=int, default=None, help="最长边缩放")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="要处理的 split 前缀；默认处理存在的 pos_train vai_train test")
    ap.add_argument("--preserve_geo", action="store_true",
                    help="使用 rasterio 读写并保留 GeoTIFF 元数据；当前 512 切块无地理信息时不要开，tifffile 更快")
    ap.add_argument("--stretch", action="store_true",
                    help="对每张 RGB 做 2%%-98%% 百分位拉伸；uint8 正射切块默认不拉伸以减少 CPU 开销")
    ap.add_argument("--save_overlay", dest="save_overlay", action="store_true",
                    help="保存 SAM3_*.tif overlay RGB，用于人工 QA，不作为模型语义输入")
    ap.add_argument("--no_save_overlay", dest="save_overlay", action="store_false",
                    help="只保存 label，不保存 SAM3 overlay RGB")
    ap.add_argument("--save_boundary", dest="save_boundary", action="store_true",
                    help="保存从 categorical label 导出的 boundary_*.tif；默认开启")
    ap.add_argument("--no_save_boundary", dest="save_boundary", action="store_false",
                    help="不保存 semantic boundary TIFF")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有输出；默认跳过已完成文件")
    ap.add_argument("--max_images", type=int, default=None, help="每个 split 最多处理多少张，调试/测速用")
    ap.add_argument("--start_index", type=int, default=0, help="从排序后的第几张开始处理，默认 0")
    ap.add_argument("--dry_run", action="store_true", help="只打印输入和输出路径，不加载 SAM3 模型")
    ap.add_argument(
        "--build_boundaries_only",
        action="store_true",
        help="只从现有 label_*.tif 生成 boundary_*.tif；不加载 SAM3，也不改写 label",
    )
    ap.add_argument("--no_reuse_image_state", action="store_true",
                    help="关闭图像 state 复用，回退到旧逻辑；仅在发现 SAM3 多 prompt 结果异常时使用")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                    help="固定顺序: building road；顺序决定 label ID")
    ap.add_argument(
        "--class_names",
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="与 --prompts 一一对应的稳定类别名；默认: building road",
    )
    ap.add_argument("--overlay_alpha", type=float, default=0.45, help="SAM3 overlay 透明度")
    ap.add_argument("--road_prompt", default=None, help="兼容旧参数；传入后与 --building_prompt 组成两类 prompt")
    ap.add_argument("--building_prompt", default=None, help="兼容旧参数；传入后与 --road_prompt 组成两类 prompt")
    ap.set_defaults(save_overlay=True, save_boundary=True)
    args = ap.parse_args()
    args.device = args.device or _auto_device()
    args.reuse_image_state = not args.no_reuse_image_state
    if args.road_prompt is not None or args.building_prompt is not None:
        args.prompts = [
            args.building_prompt or "building, rooftop",
            args.road_prompt or "road, paved surface, impervious surface",
        ]
    if len(args.prompts) != len(SEMANTIC_CLASSES):
        ap.error(
            "Semantic-modulation labels require exactly two prompts in this order: "
            "building road."
        )
    if len(args.class_names) != len(args.prompts):
        ap.error("--class_names must contain exactly one stable name for each --prompts entry")

    root = args.data_root
    split_names = args.splits or ["pos_train", "vai_train", "test"]
    splits = [(name, os.path.join(root, f"{name}_RGB")) for name in split_names]
    splits = [(name, path) for name, path in splits if os.path.isdir(path)]

    if not splits and not args.build_boundaries_only:
        raise FileNotFoundError(
            f"找不到输入目录: {', '.join(os.path.join(root, f'{name}_RGB') for name in split_names)}"
        )

    if args.build_boundaries_only:
        print("=== build_boundaries_only | no SAM3 model will be loaded ===")
        for split_name in split_names:
            build_boundaries_from_labels(root, split_name, overwrite=args.overwrite)
        print("=== DONE (boundaries only) ===")
        return

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.dry_run:
        print(
            f"=== dry_run | root={root} | classes={args.class_names} | prompts={args.prompts} "
            f"| save_boundary={args.save_boundary} ==="
        )
        for split_name, rgb_dir in splits:
            label_dir = os.path.join(root, f"{split_name}_label")
            sam3_dir = os.path.join(root, f"{split_name}_SAM3")
            boundary_dir = os.path.join(root, f"{split_name}_semantic_boundary")
            files = _find_rgb_images(rgb_dir)[args.start_index:]
            if args.max_images is not None:
                files = files[:args.max_images]
            print(f"=== {split_name}: {len(files)} files ===")
            for rgb_path in files[:10]:
                label_stem, overlay_stem, boundary_stem = _output_stems(rgb_path)
                print(
                    f"{rgb_path} -> "
                    f"{os.path.join(label_dir, label_stem + '.tif')} | "
                    f"{os.path.join(boundary_dir, boundary_stem + '.tif')} | "
                    f"{os.path.join(sam3_dir, overlay_stem + '.tif')}"
                )
            if len(files) > 10:
                print(f"... {len(files) - 10} more")
        print("=== DONE (dry_run) ===")
        return

    # 你的离线 builder（你之前修改过的版本）
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # 加载一次模型
    model = build_sam3_image_model(
        checkpoint_path=args.ckpt,
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.score_thr)

    print(f"=== device={args.device} | reuse_image_state={args.reuse_image_state} "
          f"| stretch={args.stretch} | save_overlay={args.save_overlay} "
          f"| save_boundary={args.save_boundary} "
          f"| preserve_geo={args.preserve_geo} ===")
    print(f"=== classes={args.class_names} | prompts={args.prompts} ===")
    print(f"=== splits={[name for name, _ in splits]} ===")

    for split_name, rgb_dir in splits:
        print(f"=== 处理 {Path(rgb_dir).name} ===")
        run_dir(rgb_dir, root, split_name, processor, args)

    label_map_txt, label_map_json = write_label_map(root, args.prompts, args.class_names)
    print(f"=== label maps: {label_map_txt} | {label_map_json} ===")
    print("=== DONE (batch) ===")


if __name__ == "__main__":
    main()
