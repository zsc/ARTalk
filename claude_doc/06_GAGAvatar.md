# GAGAvatar - 高斯溅射渲染器

**文件路径**: `app/GAGAvatar/models.py`
**代码行数**: 331 行
**核心职责**: 基于3D Gaussian Splatting的真实感头像渲染

## 模型概述

GAGAvatar 使用**3D高斯溅射 (Gaussian Splatting)** 技术渲染高质量头像，比传统网格渲染更真实。

## 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                        GAGAvatar                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  初始化 (一次性):                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. DINOBase特征提取                                     │ │
│  │    参考图像 [518,518,3] → feature maps                  │ │
│  │    ├── f_feature0: [256, H/4, W/4]                     │ │
│  │    └── f_feature1: [1024]                              │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│  ┌──────▼─────────────────────────────────────────────────┐ │
│  │ 2. 生成高斯参数                                          │ │
│  │   ┌──────────────────────────────────────────────────┐ │ │
│  │   │ 全局部分 (5023个高斯)                             │ │ │
│  │   │   head_base + f_feature1 → LinearGSGenerator      │ │ │
│  │   │   ├── xyz: FLAME顶点位置                          │ │ │
│  │   │   ├── colors: [5023, 32] SH系数                  │ │ │
│  │   │   ├── opacities: [5023, 1]                       │ │ │
│  │   │   ├── scales: [5023, 3]                          │ │ │
│  │   │   └── rotations: [5023, 4] 四元数                │ │ │
│  │   └──────────────────────────────────────────────────┘ │ │
│  │   ┌──────────────────────────────────────────────────┐ │ │
│  │   │ 局部部分 (2×296²个高斯)                           │ │ │
│  │   │   f_feature0 → ConvGSGenerator × 2               │ │ │
│  │   │   ├── plane_points: 平面点云                      │ │ │
│  │   │   ├── xyz: points + offset * dirs                │ │ │
│  │   │   └── colors, opacities, scales, rotations       │ │ │
│  │   └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│         │ (缓存为self._gs_params)                           │
│                                                              │
│  每帧渲染:                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 3. 更新高斯位置                                          │ │
│  │    motion [106] → FLAME → vertices [5023, 3]           │ │
│  │    gs_params['xyz'][:5023] = vertices                  │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│  ┌──────▼─────────────────────────────────────────────────┐ │
│  │ 4. 高斯渲染                                              │ │
│  │    render_gaussian(gs_params, cam_matrix, cam_params)  │ │
│  │    → [512, 512, 32] (低分辨率渲染)                      │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│  ┌──────▼─────────────────────────────────────────────────┐ │
│  │ 5. StyleUNet超分辨率                                     │ │
│  │    [512,512,32] → [512,512,3] RGB                       │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│  ┌──────▼─────────────────────────────────────────────────┐ │
│  │ 6. 添加水印                                              │ │
│  │    blend GAGAvatar logo                                │ │
│  └────────────────────────────────────────────────────────┘ │
│         │                                                    │
│  输出: RGB image [512, 512, 3]                               │
└──────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. LinearGSGenerator - 全局高斯生成

**代码详解** (GAGAvatar/models.py:141-193):

```python
class LinearGSGenerator(nn.Module):
    def __init__(self, in_dim=1024, dir_dim=27):
        super().__init__()

        # 特征提取
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//4, bias=True),  # 1024 → 256
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
        )

        layer_in_dim = in_dim//4 + dir_dim  # 256+27=283

        # 颜色预测 (SH系数)
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 32, bias=True),  # 32通道SH
        )

        # 不透明度
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True),
        )

        # 尺度
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True)
        )

        # 旋转 (四元数)
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
        )

    def forward(self, input_features, plane_direnc):
        # input_features: [B, 5023, 1024]
        # plane_direnc: [B, 27]

        input_features = self.feature_layers(input_features)  # [B, 5023, 256]

        # 扩展direnc到每个高斯
        plane_direnc = plane_direnc[:, None].expand(-1, input_features.shape[1], -1)

        # 拼接
        input_features = torch.cat([input_features, plane_direnc], dim=-1)

        # 预测高斯参数
        colors = self.color_layers(input_features)
        colors[..., :3] = torch.sigmoid(colors[..., :3])  # RGB归一化

        opacities = torch.sigmoid(self.opacity_layers(input_features))
        scales = torch.sigmoid(self.scale_layers(input_features)) * 0.05
        rotations = nn.functional.normalize(self.rotation_layers(input_features))

        return {
            'colors': colors,       # [B, 5023, 32]
            'opacities': opacities, # [B, 5023, 1]
            'scales': scales,       # [B, 5023, 3]
            'rotations': rotations  # [B, 5023, 4]
        }
```

### 2. ConvGSGenerator - 局部高斯生成

**代码详解** (GAGAvatar/models.py:196-233):

```python
class ConvGSGenerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27):
        super().__init__()
        out_dim = 32 + 1 + 3 + 4 + 1  # color+opacity+scale+rotation+position = 41

        self.gaussian_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input_features, plane_direnc):
        # input_features: [B, 256, H, W]
        # plane_direnc: [B, 27]

        # 扩展direnc到空间维度
        plane_direnc = plane_direnc[:, :, None, None].expand(-1, -1, input_features.shape[2], input_features.shape[3])

        input_features = torch.cat([input_features, plane_direnc], dim=1)

        # 卷积预测
        gaussian_params = self.gaussian_conv(input_features)  # [B, 41, H, W]

        # 解析通道
        colors = gaussian_params[:, :32]
        colors[..., :3] = torch.sigmoid(colors[..., :3])

        opacities = torch.sigmoid(gaussian_params[:, 32:33])
        scales = torch.sigmoid(gaussian_params[:, 33:36]) * 0.05
        rotations = nn.functional.normalize(gaussian_params[:, 36:40])
        positions = torch.sigmoid(gaussian_params[:, 40:41])  # 深度偏移

        # 重排为点云格式
        results = {
            'colors': colors, 'opacities': opacities,
            'scales': scales, 'rotations': rotations, 'positions': positions
        }
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).reshape(
                results[key].shape[0], -1, results[key].shape[1]
            )  # [B, H*W, C]

        return results
```

### 3. build_forward_batch() - 构建渲染batch

**代码详解** (GAGAvatar/models.py:98-128):

```python
@torch.no_grad()
def build_forward_batch(self, motion_code, flame_model):
    """
    从motion生成FLAME顶点并构建渲染batch
    """
    if not hasattr(self, '_tracked_id'):
        self.set_avatar_id('11.jpg')  # 默认avatar

    device = motion_code.device

    # 复用缓存的特征
    if not hasattr(self, 'feature_batch'):
        feature_batch = {}
        # 参考图像
        feature_batch['f_image'] = torchvision.transforms.functional.resize(
            self._tracked_id['image'], (518, 518), antialias=True
        )[None].to(device)

        # 平面点云
        feature_batch['f_planes'] = build_points_planes(296, self._tracked_id['transform_matrix'])
        feature_batch['f_planes']['plane_points'] = feature_batch['f_planes']['plane_points'][None].to(device)
        feature_batch['f_planes']['plane_dirs'] = feature_batch['f_planes']['plane_dirs'][None].to(device)

        # 目标图像元数据
        feature_batch['t_image'] = torchvision.transforms.functional.resize(
            self._tracked_id['image'], (512, 512), antialias=True
        )[None].to(device)
        feature_batch['t_transform'] = self._tracked_id['transform_matrix'][None].to(device)

        self.feature_batch = feature_batch
        self.shapecode = self._tracked_id['shapecode'][None].to(device)

    feature_batch = copy.deepcopy(self.feature_batch)

    # ========== FLAME生成顶点 ==========
    exp_code = motion_code[:, :100]
    pose_code = torch.cat([
        motion_code.new_zeros(1, 3),  # 全局旋转置零
        motion_code[:, 103:]          # 下颌旋转
    ], dim=-1)

    t_points = flame_model(
        shape_params=self.shapecode,
        pose_params=pose_code,
        expression_params=exp_code,
        eye_pose_params=pose_code.new_zeros(1, 6)
    ).float()

    # ========== 平滑额头 (稳定头顶) ==========
    if not hasattr(self, 'upper_points'):
        self.upper_points = t_points[:, forehead_indices]
    else:
        current_points = t_points[:, forehead_indices]
        self.upper_points = 0.98 * self.upper_points + 0.02 * current_points
        t_points[:, forehead_indices] = self.upper_points

    feature_batch['t_points'] = t_points

    # 更新全局旋转
    feature_batch['t_transform'][:, :3, :3] = transform_emoca_to_p3d(motion_code[:, 100:103])[:, :3, :3]

    return feature_batch
```

### 4. forward_expression() - 渲染

**代码详解** (GAGAvatar/models.py:63-95):

```python
@torch.no_grad()
def forward_expression(self, batch):
    """
    从motion渲染图像
    """
    # ========== 初始化高斯参数 (仅首次) ==========
    if not hasattr(self, '_gs_params'):
        batch_size = batch['f_image'].shape[0]
        f_image, f_planes = batch['f_image'], batch['f_planes']

        # 特征提取
        f_feature0, f_feature1 = self.base_model(f_image)

        # 方向编码
        plane_direnc = self.harmo_encoder(f_planes['plane_dirs'])

        # 全局高斯
        gs_params_g = self.gs_generator_g(
            torch.cat([
                self.head_base[None].expand(batch_size, -1, -1),  # [B, 5023, 256]
                f_feature1[:, None].expand(-1, 5023, -1),         # [B, 5023, 1024]
            ], dim=-1),
            plane_direnc
        )
        gs_params_g['xyz'] = batch['f_image'].new_zeros((batch_size, 5023, 3))

        # 局部高斯 (双侧)
        gs_params_l0 = self.gs_generator_l0(f_feature0, plane_direnc)
        gs_params_l1 = self.gs_generator_l1(f_feature0, plane_direnc)

        gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]
        gs_params_l1['xyz'] = f_planes['plane_points'] + -1 * gs_params_l1['positions'] * f_planes['plane_dirs'][:, None]

        # 拼接
        gs_params = {
            k: torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1)
            for k in gs_params_g.keys()
        }
        self._gs_params = gs_params

    gs_params = self._gs_params

    # ========== 更新高斯位置 ==========
    t_image, t_points, t_transform = batch['t_image'], batch['t_points'], batch['t_transform']
    gs_params['xyz'][:, :5023] = t_points  # 只更新全局高斯位置

    # ========== 渲染 ==========
    gen_images = render_gaussian(
        gs_params=gs_params,
        cam_matrix=t_transform,
        cam_params=self.cam_params
    )['images']  # [B, 32, 512, 512]

    # ========== 超分辨率 ==========
    sr_gen_images = self.upsampler(gen_images)  # [B, 3, 512, 512]

    # ========== 添加水印 ==========
    return self.add_water_mark(sr_gen_images.clamp(0, 1))
```

## 高斯Splatting原理

### 3D高斯

每个高斯定义为:

```
G(x) = exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

其中:
- μ: 中心位置 (xyz)
- Σ: 协方差矩阵 = R * S * Sᵀ * Rᵀ
  - R: 旋转 (四元数)
  - S: 尺度 (3D各向异性)
```

### 渲染公式

```
C(pixel) = Σ(α_i * c_i * T_i)

其中:
- α_i: 高斯i在像素的不透明度
- c_i: 高斯i的颜色 (球谐函数)
- T_i: 传输率 = Π(1-α_j), j<i
```

### 双侧平面高斯

```
人脸中心平面
    │
    ├── 前侧平面: 296² 个高斯 (面部细节)
    │
    └── 后侧平面: 296² 个高斯 (头发、耳朵)
```

## 总结

GAGAvatar 通过：
1. **3D高斯**: 比网格更灵活的表示
2. **混合高斯**: 全局(FLAME) + 局部(特征图)
3. **神经渲染**: 可微分光栅化
4. **超分辨率**: 提升最终质量

实现了照片级真实感的头像渲染。
