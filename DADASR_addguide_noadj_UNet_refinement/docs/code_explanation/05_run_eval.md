# `run_eval.py` 代码说明书

源码位置：[run_eval.py](/Users/niko/Downloads/project/run_eval.py)

这个文件是评估入口。它负责：

1. 读取 checkpoint。
2. 找到训练时的参数。
3. 构建和训练时一致的 `GADBase`。
4. 在 `vai` 或 `test` split 上预测。
5. 计算每张图和整体指标。
6. 保存误差热力图、fusion gate 可视化和 CSV 结果。

通俗地说：

```text
run_train.py 负责训练模型。
run_eval.py 负责拿训练好的模型去考试，并把分数和错题图保存下来。
```

## 顶部导入和常量

```python
import argparse
import csv
import os
import pickle
import re
import sys
import time
from types import SimpleNamespace
```

技术解释：

- `argparse` 解析评估命令行参数。
- `csv` 读写 `args.csv` 和评估结果。
- `os` 处理路径。
- `pickle` 捕获 checkpoint 安全加载失败。
- `re` 清理文件名。
- `sys` 判断是否交互式终端。
- `time` 统计评估耗时。
- `SimpleNamespace` 把参数字典变成点号访问对象。

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

技术解释：

- `Agg` 是无界面后端。
- 服务器或无显示器环境也能保存 PNG。

通俗解释：

```text
评估脚本不需要弹窗显示图，只要把图保存成文件。
```

```python
DATA_DIR_CANDIDATES = ("ProcessedData_scale10", "ProcessedData")
SPLIT_TO_DATASET_SPLIT = {
    "vai": "val",
    "test": "test",
}
```

技术解释：

- 如果 checkpoint 里的 `data_dir` 不可用，尝试自动找这些目录。
- 命令行用 `--split vai`，数据集内部用 `split='val'`。

```python
RGB_MEAN_NP = np.array([...]).reshape(1, 1, 3)
RGB_STD_NP = np.array([...]).reshape(1, 1, 3)
```

技术解释：用于把标准化后的 RGB 还原到 0-1，方便可视化。

## `parse_args`

源码位置：[run_eval.py:40](/Users/niko/Downloads/project/run_eval.py:40)

这个函数定义并解析评估脚本参数。

核心参数：

```python
parser.add_argument("checkpoint_arg", nargs="?", ...)
parser.add_argument("--checkpoint", default=None, ...)
```

技术解释：

- checkpoint 可以作为位置参数，也可以用 `--checkpoint`。
- `--checkpoint` 优先级更高。

```python
parser.add_argument("--split", choices=("vai", "test"), default=None)
```

技术解释：评估验证集或测试集。

```python
parser.add_argument("--out-dir", default=None)
parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
```

技术解释：

- `out-dir` 不传时默认保存到 checkpoint 同目录下的 `eval_<split>`。
- device 控制评估在 CPU 还是 GPU。

后面的参数大量是 override：

```python
--data-dir
--batch-size
--crop-size
--feature-extractor
--use-refinement-net
--boundary-refinement
--semantic-modulation
--refinement-only
--cross-modal-fusion
...
```

技术解释：

- 默认从 checkpoint 旁边的 `args.csv` 恢复训练配置。
- 命令行传入这些参数时，可以覆盖 `args.csv`。

通俗解释：

```text
评估时原则上应该使用训练时的配置。
但如果路径变了、batch size 想改、或者要手动覆盖某个开关，可以用命令行参数覆盖。
```

```python
args.checkpoint = args.checkpoint or args.checkpoint_arg
```

技术解释：优先使用 `--checkpoint`，否则使用位置参数。

```python
if args.checkpoint is None:
    parser.error("checkpoint path is required")
```

技术解释：没有 checkpoint 就不能评估。

```python
if args.split is None:
    args.split = prompt_split(parser)
```

技术解释：如果没传 split，在交互式终端里询问；非交互模式会报错。

## `prompt_split`

源码位置：[run_eval.py:180](/Users/niko/Downloads/project/run_eval.py:180)

```python
if not sys.stdin.isatty():
    parser.error("--split is required in non-interactive mode. Choose 'vai' or 'test'.")
```

技术解释：

- 如果脚本不是在交互式终端运行，例如批处理或 CI，就不能等待用户输入。
- 这时必须显式传 `--split`。

```python
while True:
    value = input(...).strip().lower()
    if value in SPLIT_TO_DATASET_SPLIT:
        return value
```

技术解释：交互式输入直到用户输入合法的 `vai` 或 `test`。

通俗解释：

```text
手动运行时可以问你要评估哪个 split。
脚本自动运行时必须把参数写清楚。
```

## `str_to_bool`

源码位置：[run_eval.py:191](/Users/niko/Downloads/project/run_eval.py:191)

```python
if isinstance(value, bool):
    return value
return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
```

技术解释：

- 把 CSV 里的字符串转成 bool。
- `"true"`、`"yes"`、`"1"` 都认为是 True。

通俗解释：

```text
args.csv 里布尔值可能是字符串，评估脚本要把它还原成真正的 True/False。
```

## `parser_defaults`

源码位置：[run_eval.py:197](/Users/niko/Downloads/project/run_eval.py:197)

```python
for action in train_parser._actions:
```

技术解释：遍历训练脚本定义过的所有参数。

```python
actions[action.dest] = action
values[action.dest] = None if action.default is argparse.SUPPRESS else action.default
```

技术解释：

- `actions` 保存参数定义本身，用于之后知道类型。
- `values` 保存训练参数默认值。

通俗解释：

```text
评估脚本要知道训练脚本有哪些参数、默认值是什么。
否则只读 args.csv 时，缺字段就不知道怎么补。
```

## `coerce_value`

源码位置：[run_eval.py:208](/Users/niko/Downloads/project/run_eval.py:208)

这个函数把 CSV 中的字符串值转回正确类型。

```python
if raw_value is None:
    return None
value = str(raw_value).strip()
if value == "" or value.lower() == "none":
    return None
```

技术解释：空字符串或 `"none"` 转成 None。

```python
if isinstance(default, bool):
    return str_to_bool(value)
```

技术解释：默认值是 bool 时，用 bool 转换逻辑。

```python
if action is not None and action.type is not None:
    return action.type(value)
```

技术解释：如果 argparse 定义过类型，例如 `int/float`，就用对应类型转换。

通俗解释：

```text
args.csv 里所有东西都是文本。
这个函数负责把 "8" 变回 int，把 "0.0001" 变回 float，把 "True" 变回 bool。
```

## `load_args_csv`

源码位置：[run_eval.py:225](/Users/niko/Downloads/project/run_eval.py:225)

```python
with open(args_csv_path, newline="") as f:
    reader = csv.reader(f)
    next(reader, None)
```

技术解释：打开训练时保存的 `args.csv`，跳过表头。

```python
for row in reader:
    if len(row) < 2:
        continue
    key, raw_value = row[0], row[1]
    if key in defaults:
        loaded[key] = coerce_value(...)
```

技术解释：

- 每行应该是 `key,value`。
- 只读取训练 parser 认识的 key。
- 值转换成正确类型。

## `apply_cli_overrides`

源码位置：[run_eval.py:239](/Users/niko/Downloads/project/run_eval.py:239)

```python
override_names = (...)
```

技术解释：列出允许命令行覆盖的训练参数。

```python
for name in override_names:
    value = getattr(cli_args, name, None)
    if value is not None:
        setattr(train_args, name, value)
```

技术解释：

- 如果评估命令行显式传了某个参数，就覆盖训练参数。
- 没传的不覆盖。

通俗解释：

```text
默认尊重 checkpoint 的训练配置；
你明确在 eval 命令里写的参数才会改掉它。
```

## `load_train_args`

源码位置：[run_eval.py:267](/Users/niko/Downloads/project/run_eval.py:267)

这个函数恢复训练配置。

```python
args_csv_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "args.csv")
```

技术解释：默认认为 `args.csv` 和 checkpoint 在同一个实验目录。

```python
if os.path.isfile(args_csv_path):
    loaded = load_args_csv(...)
    values.update(loaded)
```

技术解释：如果找到 args.csv，就用里面的训练参数覆盖默认值。

兼容旧 checkpoint：

```python
if "use_refinement_net" not in loaded:
    values["use_refinement_net"] = False
...
```

技术解释：

- 老实验可能没有这些新参数。
- 为了加载老 checkpoint，缺失字段时给安全默认值。

通俗解释：

```text
以前训练的模型没有 boundary_refinement/semantic_modulation 参数。
评估脚本不能因为 args.csv 少字段就崩掉，所以补默认值。
```

自动检测数据目录：

```python
if not values.get("data_dir") or not os.path.isdir(str(values.get("data_dir"))):
    for candidate in DATA_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            values["data_dir"] = candidate
```

技术解释：如果训练记录里的数据路径在当前机器不可用，尝试找本地常见目录。

```python
train_args = SimpleNamespace(**values)
apply_cli_overrides(train_args, cli_args)
return train_args, loaded_from, notes
```

技术解释：返回可点号访问的参数对象、参数来源、提示信息。

## `require_split_dirs`

源码位置：[run_eval.py:306](/Users/niko/Downloads/project/run_eval.py:306)

```python
prefix = "vai_train" if split == "vai" else "test"
required = (
    f"{prefix}_RGB",
    f"{prefix}_DSM_HR",
    f"{prefix}_DSM_LR",
    f"{prefix}_adapter_guide",
)
```

技术解释：评估一个 split 前，先检查必需目录是否存在。

```python
missing = [name for name in required if not os.path.isdir(...)]
if missing:
    raise FileNotFoundError(...)
```

通俗解释：

```text
如果 test_DSM_HR 或 test_RGB 缺了，评估结果没有意义，所以提前报错。
```

## `safe_torch_load`

源码位置：[run_eval.py:321](/Users/niko/Downloads/project/run_eval.py:321)

```python
try:
    return torch.load(path, map_location=map_location, weights_only=True)
```

技术解释：优先用 PyTorch 的安全权重加载模式。

```python
except TypeError:
    return torch.load(path, map_location=map_location)
```

技术解释：兼容老版本 PyTorch，老版本没有 `weights_only` 参数。

```python
except pickle.UnpicklingError:
    print("[Warning] ...")
    return torch.load(path, map_location=map_location, weights_only=False)
```

技术解释：

- 如果安全加载失败，退回普通加载。
- 但会警告只应该加载可信 checkpoint。

通俗解释：

```text
优先安全加载。
如果 checkpoint 格式不兼容，再退回老方式。
```

## `state_dict_from_checkpoint`

源码位置：[run_eval.py:334](/Users/niko/Downloads/project/run_eval.py:334)

```python
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
```

技术解释：

- 训练脚本保存的是包含 `model/epoch/optimizer` 的大字典。
- 也兼容直接保存 state_dict 的 checkpoint。

```python
if all(key.startswith("module.") for key in state.keys()):
    state = {key[len("module.") :]: value for key, value in state.items()}
```

技术解释：

- DataParallel 训练的权重 key 可能带 `module.` 前缀。
- 单卡模型加载时需要去掉。

```python
state.pop("logk2", None)
state.pop("mean_guide", None)
state.pop("std_guide", None)
```

技术解释：删除老 checkpoint 可能包含的兼容性废弃键。

## `adapter_guide_enabled`

源码位置：[run_eval.py:347](/Users/niko/Downloads/project/run_eval.py:347)

这个函数判断 adapter guide 路径是否有效。当前评估主流程里没有明显使用它，属于保留的兼容辅助函数。

## `to_device`

源码位置：[run_eval.py:352](/Users/niko/Downloads/project/run_eval.py:352)

```python
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        out[key] = value.to(device, non_blocking=True)
```

技术解释：把 sample 中所有 tensor 移动到评估 device。

```python
elif isinstance(value, list):
    out[key] = [v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for v in value]
```

技术解释：如果字段是 list，也处理 list 里的 tensor。

通俗解释：

```text
模型在哪个设备上，输入也必须在哪个设备上。
```

## `build_model`

源码位置：[run_eval.py:364](/Users/niko/Downloads/project/run_eval.py:364)

```python
model = GADBase(
    feature_extractor=train_args.feature_extractor,
    Npre=train_args.Npre,
    ...
)
return model.to(device)
```

技术解释：

- 用恢复出来的训练参数构建模型。
- 这必须和训练时结构一致，否则 `load_state_dict(strict=True)` 会失败。

通俗解释：

```text
checkpoint 只保存权重，不自动保存模型代码结构。
评估时必须先用同样参数搭一个空模型，再把权重塞进去。
```

## `build_loader`

源码位置：[run_eval.py:383](/Users/niko/Downloads/project/run_eval.py:383)

```python
require_split_dirs(train_args.data_dir, split, ...)
```

技术解释：先检查数据目录。

```python
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
```

技术解释：

- 评估不做旋转，不做翻转。
- 使用固定裁剪。
- `vai` 映射到 dataset 的 `val`。

通俗解释：

```text
评估必须稳定，不能每次随机裁剪或随机增强。
```

## `sample_name`

源码位置：[run_eval.py:406](/Users/niko/Downloads/project/run_eval.py:406)

这个函数给每张输出图生成名字。

处理优先级：

1. 如果 sample 有 `sample_key`，用它。
2. 如果有 `id` tensor/list/int，用 `sample_000001`。
3. 都没有，用 fallback index。

通俗解释：

```text
保存 heatmap 时不能每张都叫 result.png。
这个函数负责生成稳定、合法的文件名。
```

## `safe_name`

源码位置：[run_eval.py:424](/Users/niko/Downloads/project/run_eval.py:424)

```python
stem = os.path.splitext(os.path.basename(str(name)))[0]
return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "sample"
```

技术解释：

- 去掉路径和后缀。
- 把不适合文件名的字符替换成 `_`。
- 如果最终为空，返回 `"sample"`。

## `compute_metrics_arrays`

源码位置：[run_eval.py:429](/Users/niko/Downloads/project/run_eval.py:429)

```python
valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
if not valid.any():
    return None
```

技术解释：只在 mask 有效且 pred/gt 都有限的像素上评估。

```python
diff = pred[valid] - gt[valid]
abs_diff = np.abs(diff)
sq_diff = diff**2
median_diff = np.median(diff)
```

技术解释：

- `diff` 是预测误差。
- `abs_diff` 是绝对误差。
- `sq_diff` 是平方误差。
- `median_diff` 用于 NMAD。

返回指标：

```text
count: 有效像素数
mse: 平均平方误差
rmse: 均方根误差
mae: 平均绝对误差
medae: 绝对误差中位数
nmad: normalized median absolute deviation
rmae: 相对绝对误差百分比
max_abs_error: 最大绝对误差
p95_abs_error: 95 分位绝对误差
```

通俗解释：

```text
RMSE 看整体误差，特别惩罚大错。
MAE 看平均错多少。
MedAE/NMAD 对少量极端错误更稳。
P95 看大多数区域里的高误差水平。
```

## `aggregate_metrics`

源码位置：[run_eval.py:451](/Users/niko/Downloads/project/run_eval.py:451)

```python
for key, name in (...):
    values = np.array([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    summary[f"{name}_mean"] = float(values.mean()) if values.size else np.nan
    summary[f"{name}_std"] = float(values.std()) if values.size else np.nan
```

技术解释：

- 对每张图的指标求 mean/std。
- 忽略 NaN。

通俗解释：

```text
每张图都有分数。
这个函数把所有图的分数汇总成整体平均和波动。
```

## `compute_slope`

源码位置：[run_eval.py:471](/Users/niko/Downloads/project/run_eval.py:471)

```python
valid = np.isfinite(dsm)
fill_value = float(np.nanmedian(dsm[valid])) if valid.any() else 0.0
dsm = np.where(valid, dsm, fill_value).astype(np.float32)
```

技术解释：

- 无效 DSM 位置用中位数填充。
- 避免卷积计算 slope 时 NaN 扩散。

```python
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
```

技术解释：Sobel 算子，分别计算 x/y 方向梯度。

```python
gx = F.conv2d(dsm_t, torch.from_numpy(kx)[None, None], padding=1)
gy = F.conv2d(dsm_t, torch.from_numpy(ky)[None, None], padding=1)
return torch.sqrt(gx**2 + gy**2).squeeze().numpy()
```

技术解释：计算坡度强度。

通俗解释：

```text
slope 不是看高度本身，而是看高度变化有多陡。
DSM 超分不仅要高度对，也要边缘和坡度结构对。
```

## `slope_rmse`

源码位置：[run_eval.py:483](/Users/niko/Downloads/project/run_eval.py:483)

```python
slope_pred = compute_slope(pred)
slope_gt = compute_slope(gt)
return float(np.sqrt(np.mean((slope_pred[valid] - slope_gt[valid]) ** 2)))
```

技术解释：计算预测坡度和真值坡度之间的 RMSE。

## `masked_array`

源码位置：[run_eval.py:492](/Users/niko/Downloads/project/run_eval.py:492)

```python
arr = values.astype(np.float32).copy()
arr[~valid] = np.nan
return np.ma.masked_invalid(arr)
```

技术解释：

- 把无效位置设为 NaN。
- 返回 matplotlib 可识别的 masked array。

通俗解释：

```text
画图时无效区域不要乱显示颜色。
```

## `positive_vmax`

源码位置：[run_eval.py:498](/Users/niko/Downloads/project/run_eval.py:498)

这个函数用于 heatmap 色条上限。

```python
vmax = float(np.percentile(vals, percentile))
```

技术解释：用分位数而不是最大值，避免极端错误把色条拉得太大。

通俗解释：

```text
如果某个像素错得特别离谱，直接用最大值会让整张误差图都变暗。
用 99 分位更容易看清大多数区域。
```

## `robust_depth_limits`

源码位置：[run_eval.py:508](/Users/niko/Downloads/project/run_eval.py:508)

```python
vmin = float(np.percentile(vals, 2.0))
vmax = float(np.percentile(vals, 98.0))
```

技术解释：DSM 可视化用 2/98 分位作为显示范围，减少异常值影响。

## `add_colorbar`

源码位置：[run_eval.py:526](/Users/niko/Downloads/project/run_eval.py:526)

给 matplotlib 子图添加 colorbar，并设置字号。

## `gate_array_from_output`

源码位置：[run_eval.py:532](/Users/niko/Downloads/project/run_eval.py:532)

```python
gates = output.get("fusion_gates")
modalities = output.get("fusion_gate_modalities")
if gates is None or modalities is None:
    return None, None
```

技术解释：只有模型输出 gate 时才保存 fusion 可视化。

```python
if isinstance(gates, dict):
    gate_maps = {scale: value[batch_index].detach().cpu().numpy() ...}
else:
    gate_maps = {"fusion": gates[batch_index].detach().cpu().numpy()}
```

技术解释：

- 兼容多尺度 gate 字典。
- 当前 boundary refinement 通常是一个 tensor，包装成 `{"fusion": ...}`。

通俗解释：

```text
这个函数把模型内部的 gate tensor 拿出来，转成 numpy，方便画图。
```

## `save_fusion_visualization`

源码位置：[run_eval.py:548](/Users/niko/Downloads/project/run_eval.py:548)

这个函数保存 gate 可视化图。

第一行：

```python
valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(gt)
```

技术解释：只在有效区域显示 DSM。

```python
depth_vmin, depth_vmax = robust_depth_limits([gt, pred, bicubic], valid)
```

技术解释：GT、预测、bicubic 用同一个深度色条范围，便于比较。

```python
n_rows = 1 + len(scales)
n_cols = max(4, len(modalities))
```

技术解释：

- 第一行显示 RGB/GT/Pred/Bicubic。
- 后面的行显示不同 scale 的 gate。

通俗解释：

```text
一张图里先看预测效果，再看模型在 flat/edge 等分支之间怎么分配权重。
```

绘图部分：

```python
axes[0, 0].imshow(rgb)
...
axes[row, col].imshow(scale_gates[col], cmap="viridis", vmin=0.0, vmax=1.0)
```

技术解释：

- 第一行画 RGB、HR DSM、Pred DSM、Bicubic DSM。
- 后续行画 gate map。

```python
fig.savefig(path, dpi=dpi)
plt.close(fig)
```

技术解释：保存 PNG 并关闭 figure，防止内存积累。

## `save_error_heatmap`

源码位置：[run_eval.py:593](/Users/niko/Downloads/project/run_eval.py:593)

这个函数保存每张图的误差分析图。

```python
diff = pred - gt
abs_error = np.abs(diff)
pixel_mse = diff**2
```

技术解释：

- `diff` 是有符号误差。
- `abs_error` 是绝对误差。
- `pixel_mse` 是每像素平方误差。

```python
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
```

技术解释：创建 2x4 可视化布局。

子图内容：

```text
第一行:
  RGB
  Bicubic DSM
  Pred DSM
  GT DSM

第二行:
  |Pred - GT|
  Pixel MSE
  Pixel MSE overlay on RGB
  空白
```

通俗解释：

```text
这张图既能看模型预测是否接近真值，也能看错误集中在哪里。
overlay 图可以帮助判断错误是否出现在建筑、道路或边界处。
```

## `write_summary_csv`

源码位置：[run_eval.py:645](/Users/niko/Downloads/project/run_eval.py:645)

```python
header = [...]
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for method, summary in (("Bicubic", bic_summary), ("Ours", our_summary)):
        writer.writerow([method] + [summary[col] for col in header[1:]])
```

技术解释：

- 写整体指标 CSV。
- 比较 Bicubic baseline 和模型结果。

## `write_per_image_csv`

源码位置：[run_eval.py:674](/Users/niko/Downloads/project/run_eval.py:674)

这个函数保存每张图的指标，字段包括：

```text
ours_rmse, ours_mse, ours_mae, ...
bicubic_rmse, bicubic_mse, bicubic_mae, ...
```

通俗解释：

```text
summary 看整体平均；
per_image 看具体哪张图表现好、哪张图表现差。
```

## `batch_rgb_to_numpy`

源码位置：[run_eval.py:700](/Users/niko/Downloads/project/run_eval.py:700)

```python
if "rgb" in sample:
    return sample["rgb"].detach().cpu().permute(0, 2, 3, 1).numpy().clip(0.0, 1.0)
```

技术解释：如果 sample 单独提供 RGB，就直接转成 `[B,H,W,3]`。

```python
guide = sample["guide"][:, :3].detach().cpu().permute(0, 2, 3, 1).numpy()
return np.clip(guide * RGB_STD_NP + RGB_MEAN_NP, 0.0, 1.0)
```

技术解释：

- 否则从 guide 的前三通道取 RGB。
- 由于 dataset 对 RGB 做过标准化，这里乘 std 加 mean 还原。

通俗解释：

```text
模型看到的是标准化 RGB。
人看图需要正常 RGB，所以评估脚本要反标准化。
```

## `main`

源码位置：[run_eval.py:709](/Users/niko/Downloads/project/run_eval.py:709)

这是评估脚本主流程。

### 参数和环境检查

```python
cli_args = parse_args()
if cli_args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError(...)
if not os.path.isfile(cli_args.checkpoint):
    raise FileNotFoundError(...)
```

技术解释：

- 解析命令行参数。
- 检查 CUDA 是否可用。
- 检查 checkpoint 是否存在。

### 恢复训练参数

```python
device = torch.device(cli_args.device)
train_args, train_args_source, notes = load_train_args(cli_args.checkpoint, cli_args)
```

技术解释：读取训练配置并应用命令行覆盖。

```python
for name, value in (...):
    if not hasattr(train_args, name):
        setattr(train_args, name, value)
```

技术解释：给可能缺失的旧参数补默认值。

### 创建输出目录

```python
out_dir = cli_args.out_dir or os.path.join(
    os.path.dirname(os.path.abspath(cli_args.checkpoint)),
    f"eval_{cli_args.split}",
)
heatmap_dir = os.path.join(out_dir, "error_heatmaps")
fusion_dir = os.path.join(out_dir, "fusion_visualizations")
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(fusion_dir, exist_ok=True)
```

技术解释：

- 默认输出到 checkpoint 同目录。
- 单独创建误差热力图和 fusion 可视化目录。

### 加载 checkpoint、数据和模型

```python
ckpt = safe_torch_load(cli_args.checkpoint, map_location=device)
state = state_dict_from_checkpoint(ckpt)
```

技术解释：安全加载 checkpoint 并提取 state_dict。

```python
dataset, dataloader = build_loader(train_args, cli_args.split, device)
model = build_model(train_args, device)
model.load_state_dict(state, strict=True)
model.eval()
```

技术解释：

- 构建数据集和 DataLoader。
- 构建模型。
- 严格加载权重。
- 切换评估模式。

通俗解释：

```text
strict=True 很重要：
如果模型结构和 checkpoint 不一致，会立刻报错。
这样可以防止用错结构评估出假的结果。
```

### 主评估循环

```python
for sample in tqdm(dataloader, desc="Evaluating", unit="batch", dynamic_ncols=True):
    sample_device = to_device(sample, device)
    output = model(sample_device, train=False)
```

技术解释：

- 遍历 batch。
- 把 sample 移到 device。
- 模型前向预测。

```python
_, loss_dict = get_loss(output, sample_device, getattr(train_args, "loss", "rmse"))
```

技术解释：计算训练风格 loss，用于和训练日志一致。

```python
preds = output["y_pred"].detach().cpu().numpy()
bics = sample["y_bicubic"].detach().cpu().numpy()
gts = sample["y"].detach().cpu().numpy()
masks = sample["mask_hr"].detach().cpu().numpy()
rgbs = batch_rgb_to_numpy(sample)
```

技术解释：

- 把预测、bicubic、GT、mask、RGB 转成 numpy。
- 后续指标和 matplotlib 都用 numpy。

### 单张图指标

```python
for batch_index in range(preds.shape[0]):
```

技术解释：batch 内逐张图计算指标和保存图片。

```python
our_metrics = compute_metrics_arrays(pred, gt, mask)
bic_metrics = compute_metrics_arrays(bic, gt, mask)
```

技术解释：

- 模型结果和 GT 比。
- Bicubic baseline 和 GT 比。

```python
our_metrics["slope_rmse"] = slope_rmse(pred, gt, mask)
bic_metrics["slope_rmse"] = slope_rmse(bic, gt, mask)
```

技术解释：额外计算坡度误差。

```python
per_image_rows.append({...})
```

技术解释：记录每张图的所有指标，最后写 CSV。

### 保存误差热力图

```python
if heatmaps_saved < cli_args.num_heatmaps:
    save_error_heatmap(...)
    heatmaps_saved += 1
```

技术解释：只保存前 `num_heatmaps` 张，避免输出太多图片。

### 保存 fusion 可视化

```python
if fusion_visuals_saved < num_fusion_visuals:
    gate_maps, modalities = gate_array_from_output(output, batch_index)
    if gate_maps is not None and modalities is not None:
        save_fusion_visualization(...)
```

技术解释：

- 如果模型输出 gate，就保存 gate 图。
- 如果没有 gate，跳过。

通俗解释：

```text
不是所有模型都有 boundary/fusion gate。
有就画，没有就不画。
```

### 汇总和写文件

```python
loss_stats = {key: value / num_batches for key, value in loss_stats.items()}
our_summary = aggregate_metrics(our_rows)
bic_summary = aggregate_metrics(bic_rows)
```

技术解释：

- loss 按 batch 平均。
- 指标按图像汇总。

```python
write_summary_csv(summary_path, bic_summary, our_summary)
write_per_image_csv(per_image_path, per_image_rows)
```

技术解释：保存整体和逐图指标。

### 打印结果

```python
print("\n================ DSM SR Evaluation ================")
...
print(f"[DONE] Summary metrics: {summary_path}")
```

技术解释：在终端输出评估摘要和文件路径。

## 主入口

源码位置：[run_eval.py:916](/Users/niko/Downloads/project/run_eval.py:916)

```python
if __name__ == "__main__":
    main()
```

技术解释：直接运行 `python run_eval.py ...` 时执行 `main()`。

## 评估输出文件

默认输出目录：

```text
<checkpoint_dir>/eval_vai
或
<checkpoint_dir>/eval_test
```

里面包含：

```text
metrics_summary.csv
metrics_per_image.csv
error_heatmaps/*.png
fusion_visualizations/*.png
```

## 常用评估命令示例

```bash
python run_eval.py results/DSM/experiment_xxx/best_model.pth \
  --split vai \
  --device cuda
```

如果需要覆盖 batch size：

```bash
python run_eval.py results/DSM/experiment_xxx/best_model.pth \
  --split test \
  --batch-size 4
```

如果 checkpoint 旁边的 `args.csv` 数据路径失效：

```bash
python run_eval.py results/DSM/experiment_xxx/best_model.pth \
  --split vai \
  --data-dir ProcessedData_scale10
```

## 和训练代码的关系

`run_eval.py` 不重新训练模型。它做的是：

```text
按训练参数重建模型结构
-> 加载 checkpoint 权重
-> 固定裁剪评估数据
-> 输出指标和可视化
```

因此如果你修改了模型结构，比如 `LocalRefinementNet` 从旧结构改成 U-Net residual decoder，那么旧 checkpoint 可能无法 `strict=True` 加载。这是正常的，因为权重结构已经变了，需要重新训练新 checkpoint。

