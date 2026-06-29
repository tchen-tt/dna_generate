# diffusion_generate

DNA-Diffusion 模型的重新实现，修复了原始代码中的已知 bug，去除 Hydra 依赖，使用标准 argparse，结构更清晰。

## 文件结构

```
diffusion_generate/
├── layers.py       # 神经网络基础模块（Block、ResnetBlock、Attention 等）
├── unet.py         # UNet 架构
├── diffusion.py    # DDPM 扩散模型包装器
├── dataloader.py   # 数据加载与预处理
├── utils.py        # 工具函数
├── train.py        # 训练入口
└── infer.py        # 推理入口
```

## 快速开始

### 训练

```bash
cd diffusion_generate

# 使用默认参数训练
python train.py

# 使用缓存的预处理数据（推荐）
python train.py --load-saved-data

# 调试模式（单条序列，快速验证逻辑）
python train.py --debug --num-epochs 10 --min-epochs 0

# 自定义超参数
python train.py \
  --batch-size 64 \
  --lr 5e-5 \
  --num-epochs 3000 \
  --min-epochs 1000 \
  --precision bf16 \
  --use-wandb
```

### 推理

```bash
# 生成所有细胞类型的序列
python infer.py --checkpoint checkpoints/model_epoch999_step12000_val0.0123.pt

# 只生成指定细胞类型
python infer.py \
  --checkpoint checkpoints/model.pt \
  --cell-types K562 HepG2

# 调整引导强度和数量
python infer.py \
  --checkpoint checkpoints/model.pt \
  --cond-weight 7.0 \
  --num-samples 1000 \
  --sample-bs 20

# 保存交叉注意力图
python infer.py \
  --checkpoint checkpoints/model.pt \
  --save-attention
```

输出文件写入 `../data/outputs/{cell_type}.txt`，FASTA 格式。

## 相比原始实现的修复

| 位置 | 问题 | 修复 |
|------|------|------|
| `train.py` | `rank_0 == 0`（bool 与 int 比较），单卡时 W&B 日志和周期采样永不触发 | 改为 `if is_main` |
| `train.py` | 验证循环未调用 `model.eval()` | 验证前 `model.eval()`，训练前 `model.train()` |
| `train_util.py` | `precision=bf16` 时输入 `x` 仍被强制转为 `float32` | `x.to(device, dtype=dtype)` 正确遵循 precision |
| `utils.py` | `extract()` 中 `result.to(device)` 是空操作（返回值未赋值） | `result = result.to(device)` |
| `diffusion.py` | `classes=None` 时 `p_sample_loop` 尝试解包单值，引发 `ValueError` | `_p_sample_uncond` 始终返回 `(tensor, None)` 元组 |
| `unet.py` | `channels` 构造参数被硬编码 `channels = 1` 覆盖 | 移除覆盖，使用传入值 |
| `unet.py` | `label_emb(classes)` 被调用 4 次（相同计算重复执行） | 计算一次，结果加到共享 `t` |
| `layers.py` | `ResnetBlock` 的 FiLM 调制只作用于第一个子 Block | 两个子 Block 都应用 `scale_shift` |
| `dataloader.py` | `T.ToTensor()` 对 float 数组行为未定义（设计用于 uint8） | 直接用 `torch.from_numpy()` + `unsqueeze(1)` |
