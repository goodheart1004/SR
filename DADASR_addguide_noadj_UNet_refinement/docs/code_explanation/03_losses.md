# `losses.py` 代码说明书

源码位置：[losses.py](/Users/niko/Downloads/project/losses.py)

这个文件负责训练和验证时的损失计算。它的核心思想是：

```text
只在有效 DSM 像素上计算误差。
```

DSM 数据中可能存在无效值，比如 0、NaN、inf。`ProcessedDSMDataset` 已经生成了 `mask_hr`，这个文件会用 `mask_hr` 排除无效区域。

## 顶部导入

```python
import torch.nn.functional as F
```

技术解释：

- 只导入 PyTorch 的函数式接口。
- 本文件使用 `F.mse_loss` 和 `F.l1_loss`。

通俗解释：

```text
losses.py 不定义网络层，只需要用 PyTorch 自带的损失函数。
```

## `get_loss`

源码位置：[losses.py:4](/Users/niko/Downloads/project/losses.py:4)

```python
def get_loss(output, sample, loss_name='rmse'):
```

参数解释：

- `output`：模型输出字典，至少包含 `y_pred`。
- `sample`：数据集返回的 sample 字典，至少包含 `y` 和 `mask_hr`。
- `loss_name`：优化目标，支持 `rmse` 或 `l1`。

```python
y_pred = output['y_pred']
y, mask_hr = (sample[k] for k in ('y', 'mask_hr'))
```

技术解释：

- 从模型输出中取预测 DSM。
- 从样本中取 GT DSM 和有效区域 mask。

通俗解释：

```text
模型说“我预测的是 y_pred”；
数据说“真值是 y，但只有 mask_hr 为 1 的地方可信”。
```

```python
l1_loss, mse_loss, rmse_loss = masked_loss_triplet(y_pred, y, mask_hr)
```

技术解释：

- 一次性计算 L1、MSE、RMSE 三种指标。
- 它们都只在 `mask_hr==1` 的像素上计算。

```python
loss = select_loss(loss_name, l1_loss, rmse_loss)
optimization_loss = loss
```

技术解释：

- 根据 `loss_name` 选择真正用于反向传播的 loss。
- `optimization_loss` 初始等于主预测 loss。

通俗解释：

```text
虽然日志里会记录 L1/MSE/RMSE，但真正训练优化哪个，由 --loss 决定。
```

```python
loss_dict = {
    'l1_loss': l1_loss.detach(),
    'mse_loss': mse_loss.detach(),
    'rmse_loss': rmse_loss.detach(),
}
```

技术解释：

- 构造日志字典。
- `.detach()` 表示这些值只用于记录，不参与梯度传播。

通俗解释：

```text
训练只需要一个 loss 反传。
其他指标只是给 TensorBoard/进度条看的，所以从计算图里摘出来。
```

### `y_refined` 分支

```python
if 'y_refined' in output:
```

技术解释：

- 如果模型不是 `refinement_only`，`GADBase` 在 diffusion 后会保留 refinement 前的 `y_refined`。
- 这里会给 `y_refined` 额外算一个 loss。

通俗解释：

```text
如果模型先做 refinement 再做 diffusion，
代码不仅监督最终 y_pred，也监督 refinement 阶段的中间结果。
```

```python
ref_l1, ref_mse, ref_rmse = masked_loss_triplet(output['y_refined'], y, mask_hr)
refinement_loss = select_loss(loss_name, ref_l1, ref_rmse)
optimization_loss = optimization_loss + refinement_loss
```

技术解释：

- 对 `y_refined` 也计算 masked L1/MSE/RMSE。
- 根据同一个 `loss_name` 选择 refinement loss。
- 总优化 loss = 最终输出 loss + refinement 中间监督 loss。

通俗解释：

```text
最终答案要对，refinement 中间答案也要对。
这能让 refinement 网络不要把所有责任都丢给后面的 diffusion。
```

```python
loss_dict.update({
    'refinement_l1_loss': ref_l1.detach(),
    'refinement_mse_loss': ref_mse.detach(),
    'refinement_rmse_loss': ref_rmse.detach(),
    'refinement_optimization_loss': refinement_loss.detach(),
})
```

技术解释：把 refinement 阶段的指标也加入日志。

```python
loss_dict['optimization_loss'] = optimization_loss.detach()
return optimization_loss, loss_dict
```

技术解释：

- 返回用于反传的 `optimization_loss`。
- 返回用于记录的 `loss_dict`。

通俗解释：

```text
第一个返回值是训练真正拿来 backward 的。
第二个返回值是给进度条、TensorBoard、评估日志看的。
```

## `masked_loss_triplet`

源码位置：[losses.py:32](/Users/niko/Downloads/project/losses.py:32)

```python
def masked_loss_triplet(pred, gt, mask):
    mse_loss = mse_loss_func(pred, gt, mask)
    return l1_loss_func(pred, gt, mask), mse_loss, mse_loss.sqrt()
```

技术解释：

- 先算 MSE。
- 返回 L1、MSE、RMSE。
- RMSE 直接用 `sqrt(MSE)`。

通俗解释：

```text
这个函数一次打包返回三种误差指标，避免上层重复写代码。
```

为什么 RMSE 从 MSE 开方：

```text
MSE 是平方误差平均。
RMSE 是 MSE 的平方根，单位和 DSM 高度单位一致，更直观。
```

## `select_loss`

源码位置：[losses.py:37](/Users/niko/Downloads/project/losses.py:37)

```python
if loss_name == 'l1':
    return l1_loss
if loss_name == 'rmse':
    return rmse_loss
raise ValueError(f'Unsupported loss {loss_name}')
```

技术解释：

- 支持 `l1` 和 `rmse`。
- 不支持的名字直接报错。

通俗解释：

```text
你在命令行写 --loss rmse，就用 RMSE 训练。
你写 --loss l1，就用 L1 训练。
写别的就不让跑，防止拼错参数。
```

## `mse_loss_func`

源码位置：[losses.py:45](/Users/niko/Downloads/project/losses.py:45)

```python
valid = mask == 1.
```

技术解释：

- 找出有效像素。
- `mask_hr` 是 float，值通常是 0 或 1，所以这里用 `== 1.`。

```python
if not valid.any().item():
    return pred.sum() * 0.
```

技术解释：

- 如果这个 batch 没有任何有效像素，返回一个值为 0 的 loss。
- `pred.sum() * 0.` 仍然和计算图连接，不会破坏反向传播。

通俗解释：

```text
如果全是无效区域，不能拿空数组算 MSE。
但也不能直接返回 Python 的 0，因为训练图需要一个 tensor。
所以用 pred.sum()*0 得到“可反传但梯度为 0”的 loss。
```

```python
return F.mse_loss(pred[valid], gt[valid])
```

技术解释：

- 只取 valid 位置上的预测和真值。
- 计算 MSE。

通俗解释：

```text
模型只为有效 DSM 像素负责。
无效区域不奖励也不惩罚。
```

## `rmse_loss_func`

源码位置：[losses.py:52](/Users/niko/Downloads/project/losses.py:52)

```python
return mse_loss_func(pred, gt, mask).sqrt()
```

技术解释：RMSE 就是 MSE 的平方根。

通俗解释：

```text
MSE 的单位是高度平方。
开根号后变回高度单位，更符合 DSM 误差直觉。
```

## `l1_loss_func`

源码位置：[losses.py:56](/Users/niko/Downloads/project/losses.py:56)

```python
valid = mask == 1.
if not valid.any().item():
    return pred.sum() * 0.
return F.l1_loss(pred[valid], gt[valid])
```

技术解释：

- 和 MSE 一样先筛有效像素。
- 没有有效像素时返回 0 tensor。
- 有有效像素时计算平均绝对误差。

通俗解释：

```text
L1 看的是“平均差多少”，不平方。
它通常比 MSE/RMSE 对特别大的异常误差更不敏感。
```

## 与训练脚本的关系

`run_train.py` 中：

```python
output = self.model(sample, train=True)
loss, loss_dict = get_loss(output, sample, self.args.loss)
loss.backward()
```

含义是：

```text
模型先预测 y_pred。
get_loss 根据 y_pred、y、mask_hr 计算优化目标。
loss.backward() 根据这个目标更新模型参数。
```

## 与 refinement-only 的关系

如果使用：

```bash
--refinement-only
```

`GADBase.forward()` 直接返回：

```python
{'y_pred': y_init, ...}
```

此时 `output` 通常没有 `y_refined`，所以 loss 只监督最终 `y_pred`。

如果不使用 refinement-only，模型会 diffusion，最终输出 `y_pred`，并额外带上 `y_refined`。这时 loss 会同时监督两个结果。

