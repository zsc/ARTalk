# FLAME - 3D人脸参数化模型

**文件路径**: `app/flame_model/FLAME.py`
**代码行数**: 204 行
**核心职责**: FLAME (Faces Learned with an Articulated Model and Expressions) 人脸模型

## 模型概述

FLAME 是一个**可学习的 3D 人脸模型**，通过少量参数控制整个人脸网格：

```
参数 → FLAME → 3D Mesh (5023个顶点)
```

## 参数结构

```
┌────────────────────────────────────────────────────────┐
│                  FLAME Parameters                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  shape_params [N, 300]:    身份特征 (PCA系数)          │
│    └─ 控制: 脸型、五官位置、整体大小                    │
│                                                        │
│  expression_params [N, 100]: 表情参数                  │
│    └─ 控制: 嘴型、眉毛、眼睛等blend shapes             │
│                                                        │
│  pose_params [N, 6]:       姿态参数                    │
│    ├─ [0:3] 全局旋转 (global rotation)                │
│    └─ [3:6] 下颌旋转 (jaw rotation)                   │
│                                                        │
│  eye_pose_params [N, 6]:   眼球姿态                    │
│    ├─ [0:3] 左眼                                      │
│    └─ [3:6] 右眼                                      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## FLAME生成流程

```
1. 基础模板
   v_template [5023, 3]  平均人脸顶点

2. 形状混合 (Shape Blendshapes)
   v_shaped = v_template + Σ(shape_i * shapedir_i)

3. 姿态混合 (Pose Blendshapes)
   v_posed = v_shaped + pose_correctives(pose)

4. 线性混合蒙皮 (LBS)
   v_final = Σ(w_i * T_i * v_posed)

5. (可选) 提取landmarks
   lmks = barycentric_interpolation(v_final)
```

## 核心代码详解

### __init__() - 加载FLAME数据

**代码详解** (FLAME.py:21-66):

```python
def __init__(self, n_shape, n_exp, scale=1.0, no_lmks=False, lmks_type='lmks70'):
    super().__init__()
    self.scale = scale  # 缩放因子
    self.no_lmks, self.lmks_type = no_lmks, lmks_type

    # 加载FLAME模型数据
    _abs_path = os.path.dirname(os.path.abspath(__file__))
    self.flame_ckpt = torch.load(
        os.path.join(_abs_path, '../../assets', 'FLAME_with_eye.pt'),
        map_location='cpu', weights_only=True
    )

    flame_model = self.flame_ckpt['flame_model']
    flame_lmk = self.flame_ckpt['lmk_embeddings']

    # ========== 注册缓冲区 ==========

    # 三角面片
    self.register_buffer('faces_tensor', flame_model['f'])  # [9976, 3]

    # 基础模板顶点
    self.register_buffer('v_template', flame_model['v_template'])  # [5023, 3]

    # Shape和Expression混合形状
    shapedirs = flame_model['shapedirs']  # [5023, 3, 400]
    self.register_buffer('shapedirs',
        torch.cat([
            shapedirs[:, :, :n_shape],      # 前300个: shape
            shapedirs[:, :, 300:300+n_exp]  # 后100个: expression
        ], 2)
    )

    # Pose混合形状 (姿态校正)
    num_pose_basis = flame_model['posedirs'].shape[-1]
    self.register_buffer('posedirs',
        flame_model['posedirs'].reshape(-1, num_pose_basis).T
    )

    # 骨骼回归矩阵
    self.register_buffer('J_regressor', flame_model['J_regressor'])  # [5, 5023]

    # 骨骼层级 (kinematic tree)
    parents = flame_model['kintree_table'][0]
    parents[0] = -1  # 根节点
    self.register_buffer('parents', parents)

    # LBS权重
    self.register_buffer('lbs_weights', flame_model['weights'])  # [5023, 5]

    # 固定眼球和脖子姿态
    self.register_buffer('eye_pose', torch.zeros([1, 6], dtype=torch.float32))
    self.register_buffer('neck_pose', torch.zeros([1, 3], dtype=torch.float32))

    # Landmark相关 (略)
    # ...
```

### forward() - 生成顶点

**代码详解** (FLAME.py:117-167):

```python
def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None, verts_sclae=None):
    """
    输入:
        shape_params: [N, 300]
        expression_params: [N, 100]
        pose_params: [N, 6] or [N, 3] (自动补全为6)
        eye_pose_params: [N, 6]
    返回:
        vertices: [N, 5023, 3]
        landmarks: [N, 70, 3] (如果no_lmks=False)
    """
    batch_size = shape_params.shape[0]

    # ========== 默认参数 ==========
    if pose_params is None:
        pose_params = self.eye_pose.expand(batch_size, -1)
    if eye_pose_params is None:
        eye_pose_params = self.eye_pose.expand(batch_size, -1)
    if expression_params is None:
        expression_params = torch.zeros(batch_size, self.cfg.n_exp).to(shape_params.device)

    # 如果pose只有3维,补全为6维
    if pose_params.shape[-1] == 3:
        pose_params = torch.cat([
            torch.zeros(batch_size, 3).to(pose_params.device),  # 全局旋转placeholder
            pose_params  # 下颌旋转
        ], dim=-1)

    # ========== 拼接参数 ==========
    # betas = [shape (300) | expression (100)]
    betas = torch.cat([shape_params, expression_params], dim=1)

    # full_pose = [global (3) | neck (3) | jaw (3) | eyes (6)]
    full_pose = torch.cat([
        pose_params[:, :3],                  # 全局旋转
        self.neck_pose.expand(batch_size, -1),  # 脖子(固定为0)
        pose_params[:, 3:],                  # 下颌旋转
        eye_pose_params                      # 眼球旋转
    ], dim=1)

    # ========== LBS (Linear Blend Skinning) ==========
    template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

    vertices, _ = lbs(
        betas, full_pose, template_vertices,
        self.shapedirs, self.posedirs,
        self.J_regressor, self.parents,
        self.lbs_weights, dtype=self.dtype,
        detach_pose_correctives=False
    )

    # ========== 缩放 ==========
    if self.no_lmks:
        return vertices * self.scale

    # ========== 提取Landmarks ==========
    if self.lmks_type == 'lmks70':
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
    # ...

    return vertices * self.scale, landmarks3d * self.scale
```

## LBS (Linear Blend Skinning)

**核心思想**: 每个顶点受多个骨骼影响，加权平均变换

```python
# app/flame_model/lbs.py

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, ...):
    """
    输入:
        betas: [N, 400] (300 shape + 100 expression)
        pose: [N, 15] (5个关节 * 3维轴角)
        v_template: [N, V, 3]
        ...
    输出:
        vertices: [N, V, 3]
        joints: [N, J, 3]
    """
    # 1. Shape blending
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. 回归骨骼位置
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Pose转旋转矩阵
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(N, -1, 3, 3)

    # 4. Pose blending
    pose_feature = (rot_mats[:, 1:] - torch.eye(3)).view(N, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(N, -1, 3)
    v_posed = v_shaped + pose_offsets

    # 5. 计算全局变换矩阵
    T = transform_mat(rot_mats, J, parents)  # [N, J, 4, 4]

    # 6. 加权混合
    W = lbs_weights.unsqueeze(0).expand(N, -1, -1)  # [N, V, J]
    T_shaped = torch.matmul(W, T.view(N, J, 16)).view(N, V, 4, 4)

    # 7. 应用变换
    v_posed_homo = torch.cat([v_posed, torch.ones_like(v_posed[..., :1])], dim=-1)
    v_homo = torch.matmul(T_shaped, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    return verts, J
```

## Landmarks提取

**方法**: 重心坐标插值

```python
def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    """
    从顶点提取landmarks

    landmarks = Σ(bary_i * vertex_i)
    """
    # 找到landmark对应的三角形
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)

    # 提取三角形顶点
    lmk_vertices = torch.index_select(vertices, 1, lmk_faces.view(-1)).view(batch_size, -1, 3, 3)

    # 重心坐标插值
    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])

    return landmarks
```

## 关键技术点

### 1. Blendshapes

**原理**: 线性组合基础形状

```
v_shaped = v_base + Σ(α_i * Δv_i)

其中:
- v_base: 平均脸
- α_i: 参数权重
- Δv_i: 第i个blendshape的偏移
```

### 2. 骨骼层级

```
FLAME骨骼树:
root (0)
 ├── neck (1)
 │    └── jaw (2)
 │         ├── left_eye (3)
 │         └── right_eye (4)
```

### 3. Rodrigues公式

轴角 `θ` → 旋转矩阵 `R`:

```
R = I + sin(θ)K + (1-cos(θ))K²

其中 K 是反对称矩阵
```

## 使用示例

```python
flame = FLAMEModel(n_shape=300, n_exp=100, scale=1.0)

# 生成顶点
vertices = flame(
    shape_params=torch.randn(1, 300),      # 身份
    expression_params=torch.randn(1, 100),  # 表情
    pose_params=torch.randn(1, 6)          # 姿态
)  # [1, 5023, 3]
```

## 总结

FLAME 提供了：
1. **参数化表示**: 300+100+6 参数控制整个人脸
2. **可微分**: 支持梯度优化
3. **灵活**: 分离身份、表情、姿态
4. **标准**: 广泛用于人脸重建和动画
