# BITWISE_VAE - 运动编码解码器

**文件路径**: `app/modules/bitwise_vae.py`
**代码行数**: 348 行
**核心职责**: 运动参数的编码、量化和解码

## 模型概述

`BITWISE_VAE` 是一个**变分自编码器 (VAE)**，负责将 106 维运动参数压缩为 32 维二值码，并支持反向解码。

## 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                        BITWISE_VAE                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  编码路径:                                                    │
│  motion [B, T, 106]                                          │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────┐                                    │
│  │ TransformerEncoder  │  (depth=6, heads=8)                │
│  │  106 → 512 → 32     │                                    │
│  └─────────────────────┘                                    │
│      │ code [B, T, 32]                                      │
│      ▼                                                       │
│  ┌─────────────────────┐                                    │
│  │ MultiScaleBSQ       │  (5个尺度残差量化)                 │
│  │  连续码 → 二值码     │                                    │
│  └─────────────────────┘                                    │
│      │ bits [B, T*sum(pn), 32]                              │
│      │                                                       │
│  解码路径:                                                    │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────┐                                    │
│  │ vqidx_to_feat       │  (多尺度重建)                       │
│  │  bits → code        │                                    │
│  └─────────────────────┘                                    │
│      │ code [B, T, 32]                                      │
│      ▼                                                       │
│  ┌─────────────────────┐                                    │
│  │ TransformerDecoder  │  (depth=6, heads=8)                │
│  │  32 → 512 → 106     │                                    │
│  └─────────────────────┘                                    │
│      │                                                       │
│      ▼                                                       │
│  motion [B, T, 106]                                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. TransformerEncoder

**代码详解** (bitwise_vae.py:128-157):

```python
class TransformerEncoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, code_dim, depth=6, n_heads=8):
        super().__init__()
        # 输入映射: 106 → 512
        self.inp_mapping = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.LeakyReLU(0.2, True)
        )

        # 输出映射: 512 → 32
        self.code_mapping = nn.Linear(hidden_dim, code_dim)

        # Transformer blocks
        blocks = []
        for i in range(depth):
            blocks += [
                SimpleSelfAttention(hidden_dim, n_heads=n_heads),
                nn.Sequential(
                    nn.Linear(hidden_dim, int(1.5 * hidden_dim)),
                    nn.GELU(approximate='tanh'),
                    nn.Linear(int(1.5 * hidden_dim), hidden_dim)
                )
            ]
        self.encoder_transformer = nn.ModuleList(blocks)

    def forward(self, inp_BLC, attn_mask=None):
        feat = self.inp_mapping(inp_BLC)  # [B, L, 512]

        # Transformer处理
        for block in self.encoder_transformer:
            if isinstance(block, SimpleSelfAttention):
                feat = feat + block(feat, attn_mask)
            else:
                feat = feat + block(feat)

        out = self.code_mapping(feat)  # [B, L, 32]
        return out
```

### 2. MultiScaleBSQ (二值标量量化)

**核心思想**: 多尺度残差量化

```
原始信号 f
  │
  ├─ 尺度0 (1个patch):   f₀ → q₀,  残差 r₀ = f - q₀
  │
  ├─ 尺度1 (5个patch):   r₀ → q₁, 残差 r₁ = r₀ - q₁
  │
  ├─ 尺度2 (25个patch):  r₁ → q₂, 残差 r₂ = r₁ - q₂
  │
  └─ ...

最终重建: f̂ = q₀ + q₁ + q₂ + q₃ + q₄
```

**代码详解** (bitwise_vae.py:218-242):

```python
class MultiScaleBSQ(nn.Module):
    def forward(self, f_BTC):
        B, T, C = f_BTC.size()
        quantized_out, residual = 0., f_BTC
        all_losses, all_bit_indices = [], []

        # 逐尺度量化
        for lvl_idx, pt in enumerate(self.scale_schedule):  # (1,5,25,50,100)
            # 插值到当前尺度
            interpolate_residual = F.interpolate(
                residual.permute(0, 2, 1), size=(pt), mode='area'
            ).permute(0, 2, 1).contiguous() if pt != T else residual

            # BSQ量化
            quantized, bit_indices, loss = self.bsq_quant(interpolate_residual)

            # 插值回原始分辨率
            quantized = F.interpolate(
                quantized.permute(0, 2, 1), size=(T), mode='linear'
            ).permute(0, 2, 1).contiguous() if pt != T else quantized

            # 更新残差
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_bit_indices.append(bit_indices)
            all_losses.append(loss)

        all_bit_indices = torch.cat(all_bit_indices, dim=1)  # [B, 181*32]
        return quantized_out, all_bit_indices, all_losses
```

### 3. BSQ (Binary Scalar Quantization)

**原理**: 将L2归一化后的特征量化为 `{-1, +1}`

**代码详解** (bitwise_vae.py:308-334):

```python
class BSQ(nn.Module):
    def forward(self, f_BTC):
        # L2归一化
        f_BTC = F.normalize(f_BTC, dim=-1)

        # 量化
        quantized = self.quantize(f_BTC)  # [B, T, 32]

        # 损失计算
        persample_entropy, cb_entropy = self.soft_entropy_loss(f_BTC)
        entropy_penalty = (persample_entropy - cb_entropy) / self.inv_temperature
        commit_loss = torch.mean(((quantized.detach() - f_BTC) ** 2).sum(dim=-1))

        aux_loss = entropy_penalty * 0.1 + commit_loss * 0.2

        # 转为二值索引
        bit_indices = (quantized > 0).int()  # {0, 1}
        return quantized, bit_indices, aux_loss

    def quantize(self, z):
        q_scale = 1. / (self.codebook_dim ** 0.5)  # 1/sqrt(32)
        zhat = torch.where(z > 0, torch.tensor(1).type_as(z), torch.tensor(-1).type_as(z))
        zhat = q_scale * zhat  # 缩放到单位球面
        return z + (zhat - z).detach()  # Straight-through estimator
```

**二值化公式**:

```
z_normalized ∈ ℝ³²  (L2归一化)
      ↓
z_binary = sign(z_normalized) ∈ {-1, +1}³²
      ↓
z_scaled = z_binary / √32
```

## 核心函数

### quant_to_vqidx() - 编码

```python
@torch.no_grad()
def quant_to_vqidx(self, prev_motion, this_motion=None):
    # 拼接prev和this
    all_motion = torch.cat([prev_motion, this_motion], dim=1)  # [B, 2T, 106]

    # 归一化
    enc_in = self.norm_with_stats(all_motion)

    # Transformer编码
    enc_out = self.encoder(enc_in + self.enc_pos_embed, attn_mask=self.attn_mask)

    # 多尺度量化
    _, code_idx, _ = self.quantize(enc_out)

    # 分离prev和this
    prev_code_idx = code_idx[:, :seq_len]
    this_code_idx = code_idx[:, seq_len:]

    return prev_code_idx, this_code_idx
```

### vqidx_to_motion() - 解码

```python
@torch.no_grad()
def vqidx_to_motion(self, prev_code_idx, this_code_idx):
    # 反量化
    prev_vq_out = self.quantize.vqidx_to_feat(prev_code_idx, multi_scale=False)
    this_vq_out = self.quantize.vqidx_to_feat(this_code_idx, multi_scale=False)

    # 拼接
    vq_out = torch.cat([prev_vq_out, this_vq_out], dim=1)

    # Transformer解码
    dec_out = self.decoder(vq_out + self.dec_pos_embed, attn_mask=self.attn_mask)

    # 反归一化
    motion_code = self.unnorm_with_stats(dec_out)

    return motion_code[:, :seq_len], motion_code[:, seq_len:]
```

## 关键技术点

### 1. 双帧设计

输入固定为**2帧** (prev + this):

```
attn_mask:
       prev  this
prev   [0    -∞ ]
this   [0     0 ]
```

- prev不能看到this (causal)
- this可以看到prev (context)

### 2. 数据归一化

使用全局统计量归一化:

```python
motion_mean = [0.123, ..., 0.456]  # [106]
motion_std  = [0.789, ..., 0.321]  # [106]

normed = (motion - mean) / std
```

### 3. 位置编码

可学习的绝对位置编码:

```python
enc_pos_embed: [1, 2T, 106]
dec_pos_embed: [1, 2T, 32]
```

## 总结

BITWISE_VAE 通过：
1. **Transformer编码**: 捕捉时序依赖
2. **多尺度BSQ**: 高效二值量化
3. **Transformer解码**: 重建运动参数

实现了运动的压缩表示，为自回归生成提供离散化基础。
