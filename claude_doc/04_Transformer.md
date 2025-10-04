# Transformer - 注意力机制

**文件路径**: `app/transformer.py`
**代码行数**: 119 行
**核心职责**: 提供条件化的自注意力模块

## 核心类: AdaLNSelfAttn

### 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                     AdaLNSelfAttn                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  输入:                                                        │
│  ├── feat: [B, L, C] 当前token                              │
│  ├── prev_feat: [B, prev_L, C] 历史token                    │
│  ├── cond_BD: [B, L, D] 条件 (音频特征)                     │
│  └── attn_bias: [B, 1, L, prev_L+L] Attention mask          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. AdaLN调制                                            │ │
│  │    cond → (gamma, scale, shift) × 2                    │ │
│  └────────────────────────────────────────────────────────┘ │
│      │                                                       │
│  ┌───▼────────────────────────────────────────────────────┐ │
│  │ 2. Self-Attention分支                                   │ │
│  │    LN(feat) * (1+scale1) + shift1                       │ │
│  │       ↓                                                 │ │
│  │    ModifiedSelfAttention(feat, prev_feat, attn_bias)    │ │
│  │       ↓                                                 │ │
│  │    output * gamma1                                      │ │
│  └────────────────────────────────────────────────────────┘ │
│      │                                                       │
│  feat = feat + DropPath(attn_output)                         │
│      │                                                       │
│  ┌───▼────────────────────────────────────────────────────┐ │
│  │ 3. FFN分支                                              │ │
│  │    LN(feat) * (1+scale2) + shift2                       │ │
│  │       ↓                                                 │ │
│  │    Linear → GELU → Linear                               │ │
│  │       ↓                                                 │ │
│  │    output * gamma2                                      │ │
│  └────────────────────────────────────────────────────────┘ │
│      │                                                       │
│  feat = feat + DropPath(ffn_output)                          │
│                                                              │
│  输出: feat [B, L, C]                                         │
└──────────────────────────────────────────────────────────────┘
```

### 代码详解 (transformer.py:12-43)

```python
class AdaLNSelfAttn(nn.Module):
    def __init__(self, embed_dim, cond_dim, num_heads, mlp_ratio=4., drop_path=0., attn_l2_norm=True):
        super().__init__()
        self.C, self.D = embed_dim, cond_dim  # 768, 1024
        hidden_features = round(embed_dim * mlp_ratio)  # 768*4 = 3072

        # DropPath正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Self-Attention
        self.attn = ModifiedSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_l2_norm=attn_l2_norm)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_features, embed_dim)
        )

        # AdaLN参数预测
        self.ada_lin = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(cond_dim, 6*embed_dim)  # 6参数: gamma1,gamma2,scale1,scale2,shift1,shift2
        )

        # LayerNorm (无可学习参数)
        self.ln_wo_grad = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, feat, prev_feat, cond_BD, attn_bias=None):
        batch_size, cond_len = feat.shape[0], cond_BD.shape[1]

        # 解析AdaLN参数
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(batch_size, cond_len, 6, -1).unbind(2)

        # Attention分支
        feat = feat + self.drop_path(
            self.attn(
                self.ln_wo_grad(feat).mul(scale1.add(1)).add_(shift1),  # AdaLN
                prev_feat,
                attn_bias
            ).mul_(gamma1)  # 缩放
        )

        # FFN分支
        feat = feat + self.drop_path(
            self.ffn(
                self.ln_wo_grad(feat).mul(scale2.add(1)).add_(shift2)  # AdaLN
            ).mul(gamma2)  # 缩放
        )

        return feat
```

## ModifiedSelfAttention

### 特点

1. **跨上下文注意力**: 同时attend to `prev_feat` 和 `feat`
2. **L2归一化注意力** (可选): 提升训练稳定性

### 代码详解 (transformer.py:46-79)

```python
class ModifiedSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_l2_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 768/12 = 64
        self.attn_l2_norm = attn_l2_norm

        if attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full((1, num_heads, 1, 1), 4.0).log(), requires_grad=True)
            self.max_scale_mul = math.log(100)
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)  # 1/(4*sqrt(64))

        self.query = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, feat, prev_feat, attn_bias):
        B, L, C = feat.shape
        _, prev_L, C = prev_feat.shape

        # Query: 仅来自当前feat
        q = self.query(feat).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, c]

        # Key/Value: 来自prev_feat + feat拼接
        k = self.key(torch.cat([prev_feat, feat], dim=1)).view(B, prev_L+L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(torch.cat([prev_feat, feat], dim=1)).view(B, prev_L+L, self.num_heads, self.head_dim).transpose(1, 2)

        # (可选) L2归一化
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        # Scaled Dot-Product Attention
        output = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=0.0
        ).transpose(1, 2).reshape(B, L, C)

        output = self.proj(output)
        return output
```

### 注意力矩阵

```
Query: [B, L, C]  ← 当前token
Key:   [B, prev_L+L, C]  ← 历史+当前
Value: [B, prev_L+L, C]

Attention:
         prev_0 ... prev_n | curr_0 ... curr_L
curr_0    attn     attn         attn     -∞
curr_1    attn     attn         attn    attn
...
curr_L    attn     attn         attn    attn
```

## DropPath 正则化

**原理**: Stochastic Depth，随机丢弃整个残差分支

```python
output = x + DropPath(F(x))
       = x + mask * F(x) / keep_prob
```

**代码详解** (transformer.py:82-96):

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, 1, ...]
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)  # 期望不变

        return x * random_tensor
```

## AdaLN vs LayerNorm

### LayerNorm

```
LN(x) = (x - μ) / σ * γ + β
```

γ, β 是可学习参数，全局共享

### AdaLN (Adaptive Layer Normalization)

```
AdaLN(x, c) = (x - μ) / σ * (1 + scale(c)) + shift(c)
```

scale, shift 根据条件 `c` 动态生成

**优势**: 更强的条件控制

## 总结

`transformer.py` 提供了：
1. **AdaLN条件化**: 通过音频特征调制token
2. **跨上下文注意力**: 融合历史和当前
3. **L2归一化**: 提升训练稳定性
4. **DropPath**: 正则化防止过拟合
