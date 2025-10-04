# BitwiseARModel - 自回归生成模型

**文件路径**: `app/models.py`
**代码行数**: 165 行
**核心职责**: 多尺度自回归Transformer，从音频和风格生成运动序列

## 模型概述

`BitwiseARModel` 是 ARTalk 的**核心生成模型**，采用**多尺度自回归 (Multi-Scale Autoregressive)** 架构，结合：

- **条件**: 音频特征 + 风格特征
- **生成**: 运动参数的二值量化码
- **策略**: 从粗到细，逐尺度生成 (1→5→25→50→100帧)

## 模型架构图

```
┌──────────────────────────────────────────────────────────────┐
│                      BitwiseARModel                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  输入:                                                        │
│  ├── audio: [B, audio_len] 16kHz 音频                       │
│  └── style_motion: [B, 50, 106] 风格参考                    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. 音频编码 (audio_encoder)                            │ │
│  │     ├── wav2vec2: [B, L, 1024]                         │ │
│  │     └── mimi: [B, L, 512]                              │ │
│  └────────────────────────────────────────────────────────┘ │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────────┐ │
│  │  2. 风格编码 (style_encoder)                            │ │
│  │     style_motion → [B, 1, 128] → [B, 1, 768]          │ │
│  └────────────────────────────────────────────────────────┘ │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────────┐ │
│  │  3. 自回归生成 (Transformer + VAE)                      │ │
│  │                                                         │ │
│  │  for pidx in [0,1,2,3,4]:  # 5个尺度                   │ │
│  │      pn = patch_nums[pidx]  # (1,5,25,50,100)          │ │
│  │                                                         │ │
│  │      ┌────────────────────────────────────────┐        │ │
│  │      │ 3a. 音频插值到当前尺度                  │        │ │
│  │      │     audio_feat → [B, pn, 1024]         │        │ │
│  │      └────────────────────────────────────────┘        │ │
│  │                │                                        │ │
│  │      ┌─────────▼──────────────────────────────┐        │ │
│  │      │ 3b. Transformer (depth=16)              │        │ │
│  │      │     - Causal attention                  │        │ │
│  │      │     - AdaLN conditioning                │        │ │
│  │      │     - Cross-attention with prev_motion  │        │ │
│  │      └────────────────────────────────────────┘        │ │
│  │                │                                        │ │
│  │      ┌─────────▼──────────────────────────────┐        │ │
│  │      │ 3c. 预测二值码                          │        │ │
│  │      │     logits → bits [B, pn, 32]          │        │ │
│  │      └────────────────────────────────────────┘        │ │
│  │                │                                        │ │
│  │      ┌─────────▼──────────────────────────────┐        │ │
│  │      │ 3d. VAE解码下一尺度特征                 │        │ │
│  │      │     vqidx_to_ar_vqfeat()                │        │ │
│  │      └────────────────────────────────────────┘        │ │
│  │                                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────────┐ │
│  │  4. 最终解码 (basic_vae.vqidx_to_motion)               │ │
│  │     bits → motion [B, T, 106]                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  输出: pred_motions [B, T, 106]                             │
└──────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 初始化 (`__init__`)

**代码详解** (app/models.py:13-56):

```python
class BitwiseARModel(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()

        # ========== VAE编码解码器 ==========
        self.basic_vae = BITWISE_VAE(model_cfg=model_cfg["VAE_CONFIG"])
        self.patch_nums = self.basic_vae.patch_nums  # (1,5,25,50,100)

        # VQ特征映射到Transformer空间
        self.vqfeat_embed = nn.Linear(self.basic_vae.code_dim, 768)

        # ========== 风格编码器 ==========
        self.style_encoder = StyleEncoder()  # 106 → 128
        self.style_cond_embed = nn.Linear(128, 768)

        # ========== 音频编码器 ==========
        if model_cfg["AR_CONFIG"]['AUDIO_ENCODER'] == 'wav2vec':
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
            self.audio_encoder = Wav2Vec2Model(config)
            self.audio_feature_dim = 1024
        elif model_cfg["AR_CONFIG"]['AUDIO_ENCODER'] == 'mimi':
            self.audio_encoder = MimiModelWrapper()
            self.audio_feature_dim = 512
        else:
            raise ValueError("Invalid audio encoder")

        # ========== Transformer blocks ==========
        self.attn_depth = model_cfg["AR_CONFIG"]['T_DEPTH']  # 16
        dpr = [x.item() for x in torch.linspace(0, 0.1 * self.attn_depth/24, self.attn_depth)]

        self.attn_blocks = nn.ModuleList([
            AdaLNSelfAttn(
                embed_dim=768,
                cond_dim=self.audio_feature_dim,
                num_heads=model_cfg["AR_CONFIG"]['T_NUM_HEADS'],  # 12
                drop_path=dpr[depth_idx]
            )
            for depth_idx in range(self.attn_depth)
        ])

        # ========== 输出头 ==========
        self.cond_logits_head = AdaLNBeforeHead(embed_dim=768, cond_dim=self.audio_feature_dim)
        self.logits_head = nn.Linear(768, self.basic_vae.code_dim * 2)  # 32*2=64 (二分类)

        # ========== 无条件风格嵌入 (CFG) ==========
        self.null_style_cond = nn.Parameter(torch.randn(1, 1, 768) * 0.5)

        # ========== 位置编码 ==========
        self.prev_ratio = model_cfg["AR_CONFIG"]['PREV_RATIO']  # 2

        # 当前token位置编码
        pos_embed = torch.empty(1, sum(self.patch_nums), 768)  # [1, 181, 768]
        nn.init.trunc_normal_(pos_embed, mean=0, std=math.sqrt(1 / 768 / 3))
        self.pos_embed = nn.Parameter(pos_embed)

        # 历史token位置编码
        prev_pos_embed = torch.empty(1, sum(self.patch_nums) * self.prev_ratio, 768)
        nn.init.trunc_normal_(prev_pos_embed, mean=0, std=math.sqrt(1 / 768 / 3))
        self.prev_pos_embed = nn.Parameter(prev_pos_embed)

        # 尺度级别嵌入
        self.lvl_embed = nn.Embedding(len(self.patch_nums), 768)  # 5个级别
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=math.sqrt(1 / 768 / 3))

        # ========== Causal attention mask ==========
        attn_bias_for_masking, lvl_idx = self.build_attn_mask(self.patch_nums)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking)
        self.register_buffer('lvl_idx', lvl_idx)
```

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `patch_nums` | `(1,5,25,50,100)` | 多尺度patch数量 |
| `embed_dim` | 768 | Transformer隐藏层维度 |
| `audio_feature_dim` | 1024/512 | 音频特征维度 |
| `attn_depth` | 16 | Transformer层数 |
| `num_heads` | 12 | 注意力头数 |
| `prev_ratio` | 2 | 历史上下文窗口比例 |

### 2. 核心推理函数 (`inference`)

**函数签名**:
```python
@torch.no_grad()
def inference(self, batch, with_gtmotion=False)
```

**输入**:
```python
batch = {
    'audio': Tensor [B, audio_len],      # 16kHz音频
    'style_motion': Tensor [B, 50, 106]  # 风格参考 (可选)
}
```

**输出**:
```python
pred_motions: Tensor [B, T, 106]  # 生成的运动序列
```

#### 2.1 风格编码

**代码详解** (app/models.py:66-73):

```python
# 计算序列长度 (25 fps)
seq_length = math.ceil(batch["audio"].shape[-1] / 16000 * 25.0)

# 风格条件编码
if 'style_motion' in batch.keys() and batch['style_motion'] is not None:
    # 编码风格motion: [B, 50, 106] → [B, 128]
    motion_style = self.style_encoder(batch["style_motion"]).detach()

    # 映射到768维: [B, 128] → [B, 1, 768]
    motion_style_cond = self.style_cond_embed(motion_style)[:, None]

    # Classifier-Free Guidance (CFG)
    # guidance_scale = 1.1
    motion_style_cond = motion_style_cond * 1.1 - self.null_style_cond * 0.1
else:
    # 无风格: 使用null条件
    print("No style motion provided, use default style condition.")
    motion_style_cond = self.null_style_cond  # [1, 1, 768]
```

**CFG公式**:
```
cond_final = cond_style * 1.1 - cond_null * 0.1
           = cond_style + 0.1 * (cond_style - cond_null)
```

作用: 增强风格影响，远离无条件生成

#### 2.2 位置和级别嵌入

**代码详解** (app/models.py:74-75):

```python
# 当前token: 级别嵌入 + 位置嵌入
lvl_pos_embed = self.lvl_embed(self.lvl_idx) + self.pos_embed  # [1, 181, 768]

# 历史token: 重复级别嵌入 + 扩展位置嵌入
prev_lvl_pos_embed = self.lvl_embed(self.lvl_idx).repeat(1, self.prev_ratio, 1) + self.prev_pos_embed
```

**尺度索引示意**:

```
lvl_idx: [0,0,0,0,0, 1,1,1,1,1, ..., 4,4,...,4]
         └─── 5个0 ──┘ └─── 25个2 ──┘      └─ 100个4 ─┘
         (1个patch)    (5个patch)         (100个patch)
```

#### 2.3 音频和Motion padding

**代码详解** (app/models.py:77-88):

```python
# 填充到patch_nums[-1]=100的整数倍
padded_frame_length = math.ceil(seq_length / self.patch_nums[-1]) * self.patch_nums[-1]
padded_audio_length = int(padded_frame_length / 25.0 * 16000)
patch_audio_length = int(self.patch_nums[-1] / 25.0 * 16000)  # 100帧对应音频长度

# 音频分块
audio_chunks = batch["audio"]
audio_chunks = torch.cat([
    audio_chunks,
    audio_chunks.new_zeros(batch_size, padded_audio_length - audio_chunks.shape[1])
], dim=-1).split(patch_audio_length, dim=-1)

# 初始化prev_motion为全零
prev_motion = batch["audio"].new_zeros(batch_size, self.patch_nums[-1], self.basic_vae.motion_dim)

# 编码为VQ code
prev_code_bits, _ = self.basic_vae.quant_to_vqidx(prev_motion, this_motion=None)

# 解码为VQ特征
prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits)

# 拼接风格条件
prev_attn_feat = torch.cat([motion_style_cond, self.vqfeat_embed(prev_vqfeat)], dim=1) \
                     .repeat(1, self.prev_ratio, 1)  # [B, 362, 768]
```

**Padding示意**:

```
原始音频: |████████████████|  (比如375帧)
Padding:  |████████████████░░░░|  (填充到400帧 = 4*100)

分块:     |█████| |█████| |█████| |█████|
          chunk0  chunk1  chunk2  chunk3
          (各100帧)
```

#### 2.4 多尺度自回归生成 (核心循环)

**代码详解** (app/models.py:91-114):

```python
all_pred_motions = []

# 遍历每个音频块
for idx in range(len(audio_chunks)):
    # ========== 音频编码 ==========
    split_audio_feat = self.audio_encoder(audio_chunks[idx]).permute(0, 2, 1)  # [B, C, L]

    # ========== 多尺度插值 ==========
    split_audio_feats = [
        F.interpolate(split_audio_feat, size=(pn), mode='area').permute(0, 2, 1)
        for pn in self.patch_nums  # (1,5,25,50,100)
    ]
    split_audio_cond = torch.cat(split_audio_feats, dim=1).detach()  # [B, 181, 1024]

    # ========== 逐尺度自回归 ==========
    next_ar_vqfeat = motion_style_cond  # 初始: [B, 1, 768]

    for pidx, pn in enumerate(self.patch_nums):
        # 当前尺度的音频条件
        patch_audio_cond = split_audio_cond[:, :sum(self.patch_nums[:pidx+1])]

        # 当前尺度的attention mask
        patch_attn_bias = self.attn_bias_for_masking[
            :, :,
            :sum(self.patch_nums[:pidx+1]),
            :sum(self.patch_nums[:pidx+1]) + sum(self.patch_nums)*self.prev_ratio
        ]

        # 当前token特征 = VQ特征 + 位置编码
        attn_feat = next_ar_vqfeat + lvl_pos_embed[:, :next_ar_vqfeat.shape[1]]

        # Transformer blocks
        for bidx in range(self.attn_depth):
            attn_feat = self.attn_blocks[bidx](
                attn_feat,                          # 当前token
                prev_attn_feat + prev_lvl_pos_embed,  # 历史token
                patch_audio_cond,                   # 音频条件
                attn_bias=patch_attn_bias           # causal mask
            )

        # 输出头: 预测logits
        pred_motion_logits = self.logits_head(
            self.cond_logits_head(attn_feat, patch_audio_cond)
        )

        # 转为二值码
        pred_motion_bits = pred_motion_logits.view(
            pred_motion_logits.shape[0],
            pred_motion_logits.shape[1],
            -1, 2  # [B, L, 32, 2]
        ).argmax(dim=-1)  # [B, L, 32]

        # (非最后一层) 解码为下一尺度的输入
        if pidx < len(self.patch_nums) - 1:
            next_ar_vqfeat = self.basic_vae.vqidx_to_ar_vqfeat(pidx, pred_motion_bits)
            next_ar_vqfeat = torch.cat([motion_style_cond, self.vqfeat_embed(next_ar_vqfeat)], dim=1)

    # ========== 解码motion ==========
    split_prev_motion, split_pred_motion = self.basic_vae.vqidx_to_motion(
        prev_code_bits, pred_motion_bits
    )
    all_pred_motions.append(split_pred_motion)

    # ========== 更新历史 ==========
    prev_code_bits, _ = self.basic_vae.quant_to_vqidx(split_pred_motion, this_motion=None)
    prev_vqfeat = self.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits).detach()

    this_prev_attn_feat = torch.cat([motion_style_cond, self.vqfeat_embed(prev_vqfeat)], dim=1)

    # 滑动窗口: 丢弃最旧的，追加最新的
    prev_attn_feat = torch.cat([
        prev_attn_feat[:, this_prev_attn_feat.shape[1]:],  # 保留后面的
        this_prev_attn_feat                                # 追加新的
    ], dim=1)

# 拼接所有块
pred_motions = torch.cat(all_pred_motions, dim=1)[:, :seq_length]
return pred_motions
```

**逐尺度生成示意**:

```
尺度0 (pidx=0, pn=1):
  输入: motion_style_cond [B, 1, 768]
  输出: pred_bits [B, 1, 32]
  解码: next_ar_vqfeat [B, 1, 768]

尺度1 (pidx=1, pn=5):
  输入: prev_output + new_feat [B, 6, 768]
        (1个style + 5个新token)
  输出: pred_bits [B, 6, 32]
  解码: next_ar_vqfeat [B, 6, 768]

...

尺度4 (pidx=4, pn=100):
  输入: [B, 181, 768]
        (1+5+25+50+100)
  输出: pred_bits [B, 181, 32]
  解码: motion [B, 100, 106]  ← 最终输出
```

### 3. Attention Mask构建 (`build_attn_mask`)

**代码详解** (app/models.py:123-135):

```python
@torch.no_grad()
def build_attn_mask(self, patch_nums):
    L = sum(patch_nums)  # 181

    # 构建级别矩阵
    d = torch.cat([
        torch.full((pn,), i)
        for i, pn in enumerate(patch_nums)
    ]).view(1, L, 1)
    # d = [0,0,0,0,0, 1,1,1,1,1, ..., 4,...]

    dT = d.transpose(1, 2)  # [1, 1, L]
    lvl_idx = dT[:, 0].contiguous()  # [1, L]

    # Causal mask: 只能看到同级别或更粗级别
    attn_bias_for_masking = torch.where(
        d >= dT,       # 当前级别 >= 目标级别
        0.,            # 可见
        -torch.inf     # 不可见
    ).reshape(1, 1, L, L).contiguous()

    # 添加prev_motion维度 (始终可见)
    zero_attn_bias_for_masking = attn_bias_for_masking.new_zeros(
        attn_bias_for_masking.shape
    ).repeat(1, 1, 1, self.prev_ratio)

    attn_bias_for_masking = torch.cat([
        zero_attn_bias_for_masking,  # prev可见
        attn_bias_for_masking        # causal mask
    ], dim=-1)

    return attn_bias_for_masking, lvl_idx
```

**Mask矩阵示意** (简化为3尺度: 1,2,3):

```
         prev (可见)  |  lvl0  lvl1  lvl2
                      |  (1个) (2个) (3个)
         ─────────────┼───────────────────
         p p p p p p | 0  1  1  2  2  2   ← d^T (列级别)
         ─────────────┼───────────────────
lvl0(1个) 0  0 0 0 0 0 | 0  -∞  -∞  -∞  -∞  -∞
         ─────────────┼───────────────────
lvl1(2个) 1  0 0 0 0 0 | 0  0  -∞  -∞  -∞  -∞
         1  0 0 0 0 0 | 0  0  0  -∞  -∞  -∞
         ─────────────┼───────────────────
lvl2(3个) 2  0 0 0 0 0 | 0  0  0  0  -∞  -∞
         2  0 0 0 0 0 | 0  0  0  0  0  -∞
         2  0 0 0 0 0 | 0  0  0  0  0  0
         ─────────────┼───────────────────
          ↑ (d行级别)
```

**解读**:
- `d >= dT`: 当前级别 >= 目标级别时可见
- 粗尺度(lvl0)只能看到自己
- 细尺度(lvl2)可以看到所有更粗的级别
- 历史motion始终可见

## 辅助类: AdaLNBeforeHead

**代码详解** (app/models.py:138-148):

```python
class AdaLNBeforeHead(nn.Module):
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.C, self.D = embed_dim, cond_dim

        # LayerNorm (无参数)
        self.ln_wo_grad = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        # Adaptive affine变换
        self.ada_lin = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(cond_dim, 2*embed_dim)  # 预测scale和shift
        )

    def forward(self, feat, cond_BD):
        batch_size, cond_len = feat.shape[0], cond_BD.shape[1]

        # 解析scale和shift
        scale, shift = self.ada_lin(cond_BD).view(batch_size, cond_len, 2, -1).unbind(2)

        # AdaLN: LN(x) * (1+scale) + shift
        return self.ln_wo_grad(feat).mul(scale.add(1)).add_(shift)
```

**AdaLN公式**:
```
AdaLN(x, c) = LayerNorm(x) * (1 + scale(c)) + shift(c)
```

作用: 用条件 `c` (音频特征) 动态调制token表示

## 关键技术点

### 1. 多尺度自回归

**为什么需要多尺度?**

- **问题**: 直接生成长序列(100帧)困难
- **解决**: 从粗到细，逐步细化

```
粗尺度 (1帧):   全局语义 (说话/不说话)
  ↓
中尺度 (5,25帧): 音节级别运动
  ↓
细尺度 (50,100帧): 精细lip sync
```

### 2. 二值量化 (Bitwise)

**原理**: 将连续特征转为32维二值码 `{0,1}^32`

**优点**:
- 降低码本大小: `2^32` vs 传统VQ的 `8192`
- 更平滑的特征空间

### 3. Classifier-Free Guidance

**公式**:
```python
cond = cond_style * 1.1 - cond_null * 0.1
```

**效果**: 增强风格控制，避免生成过于平淡

### 4. 历史上下文窗口

**prev_ratio = 2**: 保留前2个chunk的motion

```
生成chunk3时:
  ├── prev_motion: chunk1 + chunk2 (200帧)
  └── curr_motion: chunk3 (100帧)
```

作用: 保持时序连贯性

## 数据流总结

```
输入: audio [B, 64000] (4秒音频)

1. 分块: 4个chunk，各1.6秒 (40000 samples → 40帧)
   → 填充到100帧 (4秒 → 100帧)

2. 每个chunk:
   audio → wav2vec → [B, L, 1024]
   ↓
   5个尺度插值 → [B, 1, 1024], [B, 5, 1024], ..., [B, 100, 1024]
   ↓
   逐尺度Transformer → bits [B, 181, 32]
   ↓
   VAE解码 → motion [B, 100, 106]

3. 拼接: 4个chunk → [B, 400, 106]
4. 截断: → [B, 100, 106] (取前100帧)

输出: pred_motions [B, 100, 106]
```

## 性能优化

1. **音频缓存**: `split_audio_cond.detach()` 减少梯度内存
2. **VAE特征缓存**: 每个尺度复用上一尺度结果
3. **批量推理**: 虽然代码限制 `batch_size=1`，但架构支持批处理

## 与VAR模型对比

ARTalk基于 [VAR (Visual Autoregressive)](https://github.com/FoundationVision/VAR) 架构，但有关键改进:

| 特性 | VAR (图像生成) | BitwiseARModel (运动生成) |
|------|----------------|---------------------------|
| 条件 | Class label | Audio + Style |
| 尺度 | (1,2,3,4,...) | (1,5,25,50,100) |
| 输出 | Image tokens | Motion codes |
| 调制 | AdaLN | AdaLN + Cross-attention |
| 历史 | 无 | 2×prev_ratio 滑动窗口 |

## 总结

`BitwiseARModel` 是 ARTalk 的**智能大脑**，通过：

1. **多尺度自回归**: 从粗到细生成
2. **双重条件**: 音频(内容) + 风格(manner)
3. **时序建模**: 历史上下文窗口
4. **高效量化**: 二值标量量化

实现了从音频到运动的高质量生成。
