# `data/processed_dsm.py` 代码说明书

源码位置：[data/processed_dsm.py](/Users/niko/Downloads/project/data/processed_dsm.py)

这个文件定义了项目的主数据集类 `ProcessedDSMDataset`。它的职责不是训练模型，而是把硬盘上的 `tif` 数据整理成模型能直接吃的 `sample` 字典。

通俗地说，它做了五件事：

1. 找到当前 split 对应的文件夹。
2. 按样本编号匹配 HR DSM、LR DSM、RGB、adapter guide。
3. 读取 tif，并做随机裁剪、翻转、旋转。
4. 把低分辨率 DSM bicubic 到高分辨率，得到 `y_bicubic`。
5. 生成训练需要的 mask、边界先验、语义 mask。

## 顶部导入和常量

```python
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomRotation
import torchvision.transforms.functional as TF
```

技术解释：

- `re` 用于从文件名中提取数字编号，例如 `DSM_000001.tif` 中的 `000001`。
- `Path` 用于更清晰地拼接路径。
- `numpy` 用于把 PIL 图像转成数组。
- `torch` 和 `F` 用于张量处理、插值、pooling。
- `Image` 用于读取 tif。
- `Dataset` 是 PyTorch 数据集基类。
- `RandomRotation` 和 `TF.rotate` 用于数据增强。

通俗解释：

```text
这些 import 是数据工厂需要的工具箱：
读文件、找编号、转成 tensor、裁剪旋转、最后喂给 DataLoader。
```

```python
SPLIT_PREFIX = {
    'train': 'pos_train',
    'val': 'vai_train',
    'test': 'test',
}
```

技术解释：

- 代码内部使用 `train/val/test`。
- 硬盘文件夹使用 `pos_train_*`、`vai_train_*`、`test_*`。
- 这个字典负责把逻辑 split 映射成真实文件夹前缀。

通俗解释：

```text
你说我要 train，代码就去找 pos_train_DSM_HR、pos_train_RGB 等目录。
你说我要 val，代码就去找 vai_train_DSM_HR、vai_train_RGB 等目录。
```

```python
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
```

技术解释：

- 这是 ImageNet 常用的 RGB 均值和标准差。
- `.view(3, 1, 1)` 把它变成 `[3,1,1]`，方便和 `[3,H,W]` 的 RGB tensor 广播计算。

通俗解释：

```text
RGB 读进来以后不是直接用 0 到 1 的值，而是做标准化。
这让后面的 UNet/ResNet 风格特征提取器更容易适应。
```

```python
SEMANTIC_CLASS_IDS = (1, 2)  # building, road
SEMANTIC_IGNORE_LABEL = 255
```

技术解释：

- SAM3 语义标签中，`1` 表示 building，`2` 表示 road。
- `255` 表示 ignore，不参与语义调制。

通俗解释：

```text
语义图里只明确关心建筑和道路。
255 是“这个像素不要信，也不要用它影响模型”。
```

## `ProcessedDSMDataset`

```python
class ProcessedDSMDataset(Dataset):
    guide_channels = 4
```

技术解释：

- 继承 PyTorch `Dataset`，因此可以被 `DataLoader` 使用。
- `guide_channels=4` 表示 `guide` 由 4 个通道组成：RGB 3 通道 + adapter guide 1 通道。

通俗解释：

```text
这个类就是项目的数据入口。
每次 DataLoader 问它要一个样本，它就返回一个整理好的 sample 字典。
```

## `__init__`

源码位置：[data/processed_dsm.py:31](/Users/niko/Downloads/project/data/processed_dsm.py:31)

```python
def __init__(
        self,
        data_dir: str,
        split='train',
        crop_size=250,
        scaling=10,
        do_horizontal_flip=True,
        max_rotation_angle=0.,
        in_memory=False,
        crop_deterministic=False,
        **kwargs
):
```

参数解释：

- `data_dir`：数据根目录，例如 `ProcessedData_scale10`。
- `split`：`train`、`val` 或 `test`。
- `crop_size`：HR crop 大小，默认 250。
- `scaling`：LR 到 HR 的比例，默认 10。
- `do_horizontal_flip`：训练时是否随机水平翻转。
- `max_rotation_angle`：训练时最大随机旋转角度。
- `in_memory`：是否把所有样本一次性加载进内存。
- `crop_deterministic`：是否使用固定裁剪网格，验证/测试时通常为 True。
- `**kwargs`：允许外部传入额外参数但不报错，保持接口宽松。

逐段解释：

```python
if split not in SPLIT_PREFIX:
    raise ValueError(f'Unsupported split {split}')
```

技术解释：只允许 `train/val/test`，防止拼错 split 后默默读错目录。

通俗解释：如果你传了 `valid` 这种代码不认识的名字，它会马上报错。

```python
if scaling <= 0:
    raise ValueError('scaling must be positive')
```

技术解释：缩放倍数必须大于 0。

通俗解释：LR 到 HR 的放大倍数不可能是 0 或负数。

```python
self.root = self._resolve_root(data_dir)
self.split = split
self.prefix = SPLIT_PREFIX[split]
self.scaling = int(scaling)
self.crop_size = None if crop_size is None or int(crop_size) <= 0 else int(crop_size)
```

技术解释：

- `_resolve_root` 找到真正的数据根目录。
- `self.prefix` 把 `train` 映射成 `pos_train`。
- 如果 `crop_size` 是 None 或小于等于 0，就不裁剪。

通俗解释：

```text
这几行是在记录“我接下来要读哪个目录、按多大 crop 读、LR/HR 差几倍”。
```

```python
if self.crop_size is not None and self.crop_size % self.scaling != 0:
    raise ValueError(f'crop_size ({self.crop_size}) must be divisible by scaling ({self.scaling})')
```

技术解释：

- HR crop 必须能被 `scaling` 整除。
- 比如 `crop_size=250`、`scaling=10`，LR crop 就是 `25`。

通俗解释：

```text
HR 裁 250，对应 LR 要裁 25。
如果不能整除，LR 和 HR 的区域就对不上。
```

```python
self.records = self._build_records()
self.cache = [self._load_record(record) for record in self.records] if in_memory else None
self.deterministic_map = self._build_deterministic_map() if crop_deterministic and self.crop_size else None
```

技术解释：

- `_build_records()` 建立样本列表，每个元素记录一组文件路径。
- `in_memory=True` 时提前把所有 tif 读入内存。
- `crop_deterministic=True` 时建立固定裁剪索引，通常用于验证和测试。

通俗解释：

```text
records 是“数据清单”。
cache 是“要不要先把所有图片搬进内存”。
deterministic_map 是“验证时不要随机裁，而是按固定网格一块块裁”。
```

## `_resolve_root`

源码位置：[data/processed_dsm.py:65](/Users/niko/Downloads/project/data/processed_dsm.py:65)

```python
root = Path(data_dir)
if (root / 'pos_train_DSM_HR').is_dir():
    return root
nested = root / 'ProcessedData_scale10'
if nested.is_dir():
    return nested
raise FileNotFoundError(...)
```

技术解释：

- 如果 `data_dir` 本身就是数据根目录，则直接返回。
- 如果你传的是上一级目录，而里面有 `ProcessedData_scale10`，就自动进入。
- 两种都找不到就报错。

通俗解释：

```text
这段是容错逻辑：
你可以传 ProcessedData_scale10，也可以传它的父目录。
只要代码能找到 pos_train_DSM_HR，就认为找到了数据。
```

## `_sample_id`

源码位置：[data/processed_dsm.py:75](/Users/niko/Downloads/project/data/processed_dsm.py:75)

```python
matches = re.findall(r'\d+', path.stem)
```

技术解释：

- 从文件名不含后缀的部分提取所有数字片段。
- 例如 `DSM_000001.tif` 的 `path.stem` 是 `DSM_000001`，提取结果是 `['000001']`。

```python
return int(matches[-1])
```

技术解释：

- 使用最后一个数字作为样本 ID。
- 这样 `adapter_1.tif` 和 `DSM_000001.tif` 都会得到 ID `1`。

通俗解释：

```text
这就是“给每张图找身份证号”。
后面 RGB、DSM_HR、DSM_LR 都靠这个编号配对。
```

## `_index_folder`

源码位置：[data/processed_dsm.py:81](/Users/niko/Downloads/project/data/processed_dsm.py:81)

```python
folder = self.root / f'{self.prefix}_{suffix}'
```

技术解释：根据 split 前缀和后缀拼出目录，例如 `pos_train_DSM_HR`。

```python
if not folder.is_dir():
    raise FileNotFoundError(...)
```

技术解释：这是必需目录，缺了就不能训练。

```python
return {self._sample_id(path): path for path in sorted(folder.glob('*.tif'))}
```

技术解释：

- 找到目录下所有 `.tif`。
- 用样本编号作为 key，文件路径作为 value。

通俗解释：

```text
把一个目录变成字典：
1 -> DSM_000001.tif
2 -> DSM_000002.tif
```

## `_build_records`

源码位置：[data/processed_dsm.py:87](/Users/niko/Downloads/project/data/processed_dsm.py:87)

这个函数把多个目录中的同编号文件配成一个样本。

```python
dsm_hr = self._index_folder('DSM_HR')
dsm_lr = self._index_folder('DSM_LR')
rgb = self._index_folder('RGB')
adapter = self._index_folder('adapter_guide')
```

技术解释：四类文件是必需的。

通俗解释：

```text
没有 HR DSM 就没监督；
没有 LR DSM 就没输入；
没有 RGB/adapter guide 就没引导信息。
```

```python
ids = sorted(set(dsm_hr) & set(dsm_lr) & set(rgb) & set(adapter))
```

技术解释：

- 取四个字典 key 的交集。
- 只有同时存在 HR、LR、RGB、adapter 的编号才会成为有效样本。

通俗解释：

```text
如果某个编号只有 RGB 没有 DSM，它不会被使用。
这行就是在找“完整样本”。
```

```python
sam3 = self._optional_index_folder('SAM3')
label = self._optional_index_folder('label')
```

技术解释：SAM3 overlay 和 label 是可选的。

通俗解释：没有语义标注也能走旧训练，只是不能启用 semantic modulation。

```python
record = {
    'id': sample_id,
    'dsm_hr': dsm_hr[sample_id],
    'dsm_lr': dsm_lr[sample_id],
    'rgb': rgb[sample_id],
    'adapter': adapter[sample_id],
}
```

技术解释：一个 `record` 保存一个样本所有文件路径。

通俗解释：`record` 不是图像本身，只是“这一个样本对应哪些文件”。

## `_optional_index_folder`

源码位置：[data/processed_dsm.py:115](/Users/niko/Downloads/project/data/processed_dsm.py:115)

这个函数和 `_index_folder` 类似，但目录不存在时返回空字典，不报错。

技术上用于：

```text
SAM3
label
```

通俗解释：

```text
语义文件是增强功能。
有就用，没有也不影响普通 boundary/refinement 训练。
```

## `_build_deterministic_map`

源码位置：[data/processed_dsm.py:121](/Users/niko/Downloads/project/data/processed_dsm.py:121)

这个函数用于验证/测试时固定裁剪。

```python
with Image.open(record['dsm_hr']) as image:
    width, height = image.size
```

技术解释：读取 HR DSM 的尺寸。

```python
if self.crop_size > height or self.crop_size > width:
    raise ValueError(...)
```

技术解释：crop 不能比原图还大。

```python
num_h = height // self.crop_size
num_w = width // self.crop_size
```

技术解释：计算竖向和横向能切出多少个完整 crop。

```python
deterministic_map.extend(
    (record_index, crop_h, crop_w)
    for crop_h in range(num_h)
    for crop_w in range(num_w)
)
```

技术解释：保存每个 crop 对应的原始样本编号和网格位置。

通俗解释：

```text
训练时可以随机裁。
验证/测试时要稳定，所以把整张图按 250x250 网格切开。
每次验证都切同样的位置，结果才可比较。
```

## `__getitem__`

源码位置：[data/processed_dsm.py:137](/Users/niko/Downloads/project/data/processed_dsm.py:137)

这是 PyTorch DataLoader 真正调用的函数。

```python
if self.deterministic_map is None:
    record_index = index
    crop_index = None
else:
    record_index, crop_h, crop_w = self.deterministic_map[index]
    crop_index = (crop_h, crop_w)
```

技术解释：

- 训练时通常没有 deterministic map，`index` 就是样本编号。
- 验证时 `index` 代表某个固定 crop，先查出它属于哪张图、哪个网格。

通俗解释：

```text
随机训练：第 index 个样本随便裁一块。
固定验证：第 index 个条目其实是“某张图的第几块 crop”。
```

```python
sample = self.cache[record_index] if self.cache is not None else self._load_record(self.records[record_index])
```

技术解释：

- 如果开启缓存，从内存拿。
- 否则从硬盘读 tif。

```python
sample = {key: value.clone() if torch.is_tensor(value) else value for key, value in sample.items()}
```

技术解释：

- tensor 要 clone，防止后面的 crop/augment 修改缓存中的原始数据。

通俗解释：

```text
如果数据已经缓存进内存，每次取出来都要复制一份。
否则随机裁剪会把原始缓存改坏。
```

```python
sample = self._crop(sample, crop_index)
sample = self._augment(sample)
sample = self._finalize(sample)
return sample
```

技术解释：一个样本会依次经过裁剪、增强、最终字段生成。

通俗解释：

```text
先切出一小块，再做翻转/旋转，最后补齐模型需要的 y_bicubic、mask、边界和语义字段。
```

## `__len__`

源码位置：[data/processed_dsm.py:152](/Users/niko/Downloads/project/data/processed_dsm.py:152)

```python
return len(self.deterministic_map) if self.deterministic_map is not None else len(self.records)
```

技术解释：

- 固定裁剪时，数据集长度是 crop 数量。
- 随机训练时，数据集长度是原图样本数量。

通俗解释：

```text
验证时一张 500x500 图可以切成多个 250x250 crop，所以长度可能比原图数量大。
```

## `_load_record`

源码位置：[data/processed_dsm.py:155](/Users/niko/Downloads/project/data/processed_dsm.py:155)

```python
rgb = self._read_rgb(record['rgb'])
adapter = self._read_single(record['adapter'])
guide = torch.cat([rgb, adapter], dim=0)
```

技术解释：

- RGB 读成 `[3,H,W]`。
- adapter guide 读成 `[1,H,W]`。
- 拼接成 `[4,H,W]`。

通俗解释：

```text
模型的 guide 不是只有 RGB，而是 RGB 三个通道再加一个辅助引导通道。
```

```python
sample = {
    'id': record['id'],
    'guide': guide,
    'y': self._read_single(record['dsm_hr']),
    'source': self._read_single(record['dsm_lr']),
}
```

技术解释：

- `y` 是 HR DSM，训练目标。
- `source` 是 LR DSM，模型输入的低分辨率高程。

```python
if 'sam3' in record:
    sample['sam3'] = self._read_rgb(record['sam3'], normalize=False)
if 'label' in record:
    sample['label'] = self._read_single(record['label']).long()
```

技术解释：

- SAM3 overlay 不做 ImageNet normalization。
- label 读入后转成 long，因为它是类别编号，不是连续图像。

通俗解释：

```text
DSM 是连续高程值。
label 是类别 ID，所以必须当整数类别处理。
```

## `_read_rgb`

源码位置：[data/processed_dsm.py:173](/Users/niko/Downloads/project/data/processed_dsm.py:173)

```python
with Image.open(path) as image:
    array = np.array(image.convert('RGB'), dtype=np.float32)
```

技术解释：

- 用 PIL 打开图像。
- 强制转成 RGB。
- 转成 float32 numpy 数组。

```python
tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
```

技术解释：

- 原始 numpy 是 `[H,W,3]`。
- PyTorch 通常用 `[C,H,W]`，所以 `permute(2,0,1)`。
- 除以 255，把 0-255 变成 0-1。

```python
return (tensor - RGB_MEAN) / RGB_STD if normalize else tensor
```

技术解释：

- 默认对 RGB 做标准化。
- 如果 `normalize=False`，返回 0-1 的原始 RGB。

通俗解释：

```text
训练用 RGB 要标准化；
SAM3 overlay 只是可视化或边界来源，不一定要标准化。
```

## `_read_single`

源码位置：[data/processed_dsm.py:180](/Users/niko/Downloads/project/data/processed_dsm.py:180)

```python
array = np.array(image, dtype=np.float32)
if array.ndim == 3:
    array = array[..., 0]
return torch.from_numpy(array).unsqueeze(0)
```

技术解释：

- 单通道 tif 读成二维 `[H,W]`。
- 如果某些 tif 被读成三维，则取第一个通道。
- `unsqueeze(0)` 变成 `[1,H,W]`。

通俗解释：

```text
DSM、adapter、label 这些都按单通道处理。
即使文件意外有多个通道，这里也只拿第一层。
```

## `_crop`

源码位置：[data/processed_dsm.py:187](/Users/niko/Downloads/project/data/processed_dsm.py:187)

这个函数同时裁 HR 和 LR，关键是要保证它们对应同一块地理区域。

```python
_, hr_h, hr_w = sample['y'].shape
_, lr_h, lr_w = sample['source'].shape
lr_crop = self.crop_size // self.scaling
```

技术解释：

- HR crop 是 `self.crop_size`。
- LR crop 是 HR crop 除以 scale。
- 默认 `250 // 10 = 25`。

```python
if crop_index is None:
    lr_top = torch.randint(0, lr_h - lr_crop + 1, (1,)).item()
    lr_left = torch.randint(0, lr_w - lr_crop + 1, (1,)).item()
else:
    crop_h, crop_w = crop_index
    lr_top = crop_h * lr_crop
    lr_left = crop_w * lr_crop
```

技术解释：

- 训练时随机选择 LR 起点。
- 验证时使用固定网格位置。

通俗解释：

```text
先在 LR 图上决定裁哪里。
因为 LR 和 HR 有 10 倍关系，所以 LR 的一个像素对应 HR 的 10 个像素。
```

```python
hr_top = lr_top * self.scaling
hr_left = lr_left * self.scaling
```

技术解释：把 LR 坐标换算成 HR 坐标。

通俗解释：LR 从第 5 个像素开始，HR 就从第 50 个像素开始。

```python
sample['guide'] = sample['guide'][:, hr_top:hr_bottom, hr_left:hr_right]
sample['y'] = sample['y'][:, hr_top:hr_bottom, hr_left:hr_right]
sample['source'] = sample['source'][:, lr_top:lr_bottom, lr_left:lr_right]
```

技术解释：

- `guide` 和 `y` 是 HR 尺寸，所以用 HR 坐标裁。
- `source` 是 LR 尺寸，所以用 LR 坐标裁。

通俗解释：

```text
RGB/adapter/HR DSM 裁大块 250x250。
LR DSM 裁对应小块 25x25。
```

`sam3` 和 `label` 也按 HR 坐标裁，因为它们和 RGB/HR DSM 同分辨率。

## `_augment`

源码位置：[data/processed_dsm.py:222](/Users/niko/Downloads/project/data/processed_dsm.py:222)

```python
if self.do_horizontal_flip and self.split == 'train' and torch.rand(()) < 0.5:
```

技术解释：

- 只有训练集做随机水平翻转。
- 验证/测试不做随机增强，保证结果稳定。

```python
for key in ('guide', 'y', 'source', 'sam3', 'label'):
    if key in sample:
        sample[key] = sample[key].flip(-1)
```

技术解释：对所有空间字段同步翻转。

通俗解释：

```text
不能只翻 RGB 不翻 DSM。
所有图必须一起翻，地理位置才对得上。
```

旋转部分：

```python
angle = RandomRotation.get_params([-self.max_rotation_angle, self.max_rotation_angle])
```

技术解释：随机采样一个旋转角度。

```python
sample['guide'] = TF.rotate(sample['guide'], angle, InterpolationMode.BILINEAR, fill=0)
sample['y'] = TF.rotate(sample['y'], angle, InterpolationMode.BILINEAR, fill=0)
sample['source'] = TF.rotate(sample['source'], angle, InterpolationMode.BILINEAR, fill=0)
```

技术解释：

- 连续值图像使用 bilinear 插值。
- 空出来的区域填 0。

```python
sample['label'] = TF.rotate(sample['label'].float(), angle, InterpolationMode.NEAREST, fill=0).long()
```

技术解释：

- label 是类别，不能用 bilinear。
- nearest 保证类别 ID 不会被插值成小数。

通俗解释：

```text
DSM 可以平滑插值。
类别图不能平滑插值，否则 building=1 和 road=2 会变成 1.37 这种没有意义的类别。
```

## `_finalize`

源码位置：[data/processed_dsm.py:239](/Users/niko/Downloads/project/data/processed_dsm.py:239)

这个函数把原始字段整理成训练真正需要的字段。

```python
mask_hr = torch.isfinite(y) & (y > 0)
mask_lr = torch.isfinite(source) & (source > 0)
```

技术解释：

- DSM 中无效值可能是 NaN、inf 或非正值。
- 有效像素要求有限且大于 0。

通俗解释：

```text
只在真实有效的 DSM 高程区域算 loss。
坏值区域不能参与训练。
```

```python
y = torch.where(mask_hr, y, torch.zeros_like(y))
source = torch.where(mask_lr, source, torch.zeros_like(source))
```

技术解释：无效位置置 0，避免后续插值或模型计算中传播 NaN。

```python
y_bicubic = F.interpolate(
    source.unsqueeze(0),
    size=y.shape[-2:],
    mode='bicubic',
    align_corners=False
).squeeze(0)
```

技术解释：

- `source` 是 `[1,lr_h,lr_w]`。
- `F.interpolate` 需要 batch 维，所以先 `unsqueeze(0)` 变成 `[1,1,lr_h,lr_w]`。
- 上采样到 HR DSM 的空间尺寸。
- 最后 `squeeze(0)` 去掉临时 batch 维。

通俗解释：

```text
这一步就是把低分辨率 DSM 先用传统 bicubic 放大。
后面的网络不是从零生成 DSM，而是在这个粗结果上做修正。
```

```python
sample['boundary_prior'] = self._make_boundary_prior(sample)
```

技术解释：生成旧 boundary refinement 使用的一通道边界先验。

```python
(
    sample['semantic_masks'],
    sample['semantic_valid_mask'],
    sample['semantic_schema_valid'],
    sample['semantic_boundary_prior'],
) = self._make_semantic_inputs(sample)
```

技术解释：生成 semantic modulation 使用的 building/road masks、有效区域、schema 检查结果和语义边界。

通俗解释：

```text
finalize 是数据出厂前的最后加工：
补 bicubic 初始 DSM，补有效 mask，补边界和语义提示。
```

## `_make_boundary_prior`

源码位置：[data/processed_dsm.py:270](/Users/niko/Downloads/project/data/processed_dsm.py:270)

```python
if 'sam3' in sample:
    source = sample['sam3']
elif 'label' in sample:
    source = sample['label'].float()
else:
    source = sample['guide'][-1:].float()
```

技术解释：

- 优先从 SAM3 overlay 生成边界。
- 没有 SAM3 则从 label 生成边界。
- 都没有则从 adapter guide 最后一通道生成边界。

通俗解释：

```text
边界先验需要一个“哪里变化明显”的图。
有 SAM3 就用 SAM3，没有就用 label，再没有就退回 adapter guide。
```

```python
boundary = ProcessedDSMDataset._edge_map(source)
```

技术解释：调用 `_edge_map` 计算边缘强度图。

```python
if boundary.shape[-2:] != target_size:
    boundary = F.interpolate(...)
```

技术解释：确保边界图尺寸和 HR DSM 一致。

通俗解释：

```text
LocalRefinementNet 的 boundary gate 要和 DSM residual 同尺寸，所以 boundary_prior 必须 resize 到 HR 大小。
```

## `_make_semantic_inputs`

源码位置：[data/processed_dsm.py:290](/Users/niko/Downloads/project/data/processed_dsm.py:290)

这个函数专门为 `semantic_modulation=True` 准备输入。

如果没有 label：

```python
zeros = sample['y'].new_zeros(1, *target_size)
return (
    sample['y'].new_zeros(len(SEMANTIC_CLASS_IDS), *target_size),
    zeros,
    torch.tensor(False, dtype=torch.bool),
    zeros,
)
```

技术解释：

- 返回全 0 的 semantic masks。
- `semantic_schema_valid=False`。
- 如果之后模型开启 semantic modulation，`GADBase` 会检查并报错。

通俗解释：

```text
没有 label 时不马上报错，因为普通训练可以不用语义。
但如果你硬要开 semantic_modulation，它会阻止你用无效语义训练。
```

label 尺寸和类型检查：

```python
if label.ndim == 2:
    label = label.unsqueeze(0)
if label.ndim != 3 or label.shape[0] != 1:
    raise ValueError(...)
```

技术解释：label 必须是 `[1,H,W]`。

```python
if label.shape[-2:] != target_size:
    label = F.interpolate(..., mode='nearest').squeeze(0).long()
else:
    label = label.long()
```

技术解释：

- 如果 label 尺寸不等于 HR DSM，就最近邻 resize。
- 最后确保是整数类别。

schema 检查：

```python
schema_valid = (
    (label == 0)
    | (label == SEMANTIC_CLASS_IDS[0])
    | (label == SEMANTIC_CLASS_IDS[1])
    | (label == SEMANTIC_IGNORE_LABEL)
).all()
```

技术解释：

- 只允许 0、1、2、255。
- 出现其他值说明 label schema 不符合当前 semantic modulation 设定。

通俗解释：

```text
这相当于检查标签字典是否正确。
如果图里出现 class=3，但模型只知道 building 和 road，就不能继续。
```

生成 semantic masks：

```python
semantic_masks = torch.cat(
    [(label == class_id).float() for class_id in SEMANTIC_CLASS_IDS],
    dim=0,
)
```

技术解释：

- 对 building 和 road 分别生成二值 mask。
- 输出形状 `[2,H,W]`。

通俗解释：

```text
第一张 mask 告诉模型哪里是建筑。
第二张 mask 告诉模型哪里是道路。
```

```python
semantic_valid_mask = (label != SEMANTIC_IGNORE_LABEL).float()
```

技术解释：不是 255 的地方都算语义有效。

```python
semantic_boundary_prior = ProcessedDSMDataset._semantic_boundary_map(label, semantic_valid_mask)
```

技术解释：基于类别变化生成语义边界。

## `_semantic_boundary_map`

源码位置：[data/processed_dsm.py:344](/Users/niko/Downloads/project/data/processed_dsm.py:344)

这个函数检测类别边界。

```python
boundary = valid_mask.new_zeros(valid_mask.shape)
valid = valid_mask.bool()
```

技术解释：初始化全 0 边界图，只在 valid 区域内计算边界。

水平边界：

```python
horizontal = (label[:, :, 1:] != label[:, :, :-1]) & (
    valid[:, :, 1:] & valid[:, :, :-1]
)
```

技术解释：

- 比较左右相邻像素类别是否不同。
- 两个像素都必须 valid，ignore 区域不产生边界。

通俗解释：

```text
如果一个像素是 building，旁边是 road 或 background，就认为中间有语义边界。
但如果旁边是 ignore，就不算边界，因为 ignore 不可信。
```

垂直边界：

```python
vertical = (label[:, 1:, :] != label[:, :-1, :]) & (
    valid[:, 1:, :] & valid[:, :-1, :]
)
```

技术解释：同理，比较上下相邻像素类别是否不同。

```python
return F.max_pool2d(boundary.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
```

技术解释：

- 用 max pooling 把边界膨胀一圈。
- `unsqueeze(0)` 是临时 batch 维。

通俗解释：

```text
边界通常很细，只有 1 个像素。
膨胀后变粗一点，模型更容易感知到边界区域。
```

## `_edge_map`

源码位置：[data/processed_dsm.py:369](/Users/niko/Downloads/project/data/processed_dsm.py:369)

这个函数从任意单通道或多通道图中生成边缘强度图。

```python
tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
```

技术解释：把 NaN 和 inf 替换成 0，避免边缘计算产生异常。

```python
if tensor.ndim == 2:
    tensor = tensor.unsqueeze(0)
```

技术解释：确保输入是 `[C,H,W]`。

```python
dx[:, :, 1:] = (tensor[:, :, 1:] - tensor[:, :, :-1]).abs().mean(dim=0, keepdim=True)
dy[:, 1:, :] = (tensor[:, 1:, :] - tensor[:, :-1, :]).abs().mean(dim=0, keepdim=True)
```

技术解释：

- `dx` 是横向相邻像素差。
- `dy` 是纵向相邻像素差。
- 如果输入有多通道，就对通道求平均。

通俗解释：

```text
看左右、上下相邻像素差多少。
差得越大，越可能是边界。
```

```python
edge = torch.maximum(dx, dy)
edge = F.max_pool2d(edge.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
return edge / edge.amax().clamp_min(1e-6)
```

技术解释：

- 取横向和纵向边缘的最大值。
- max pooling 膨胀边缘。
- 最后归一化到 0-1。
- `clamp_min(1e-6)` 防止除以 0。

通俗解释：

```text
把“变化明显的地方”变成一张 0 到 1 的边界图。
0 表示不像边界，1 表示非常像边界。
```

## 本文件的关键输出字段

最终 `ProcessedDSMDataset.__getitem__()` 返回的 `sample` 通常包含：

```text
id
guide
y
source
mask_hr
mask_lr
y_bicubic
boundary_prior
semantic_masks
semantic_valid_mask
semantic_schema_valid
semantic_boundary_prior
```

如果原始 record 有 SAM3/label，还会包含：

```text
sam3
label
```

## 和模型的关系

`GADBase.forward()` 会使用：

```text
guide
y_bicubic
source
boundary_prior 或 semantic_boundary_prior
semantic_masks
semantic_valid_mask
semantic_schema_valid
```

`losses.get_loss()` 会使用：

```text
y
mask_hr
```

因此这个数据集类是整个项目的数据合同：它决定了模型和 loss 能拿到哪些字段。

