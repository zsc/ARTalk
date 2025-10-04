# inference.py - 主入口与推理引擎

**文件路径**: `inference.py`
**代码行数**: 237 行
**核心职责**: 系统主入口，整合所有模块，提供推理引擎和用户界面

## 文件概述

`inference.py` 是 ARTalk 系统的**主入口文件**，包含两个核心组件：

1. **ARTAvatarInferEngine**: 推理引擎类，整合所有模型和渲染器
2. **run_gradio_app()**: Gradio Web 界面函数

## 核心类: ARTAvatarInferEngine

### 类结构图

```
ARTAvatarInferEngine
├── __init__(load_gaga, fix_pose, clip_length, device)
│   └── 初始化所有模型和渲染器
├── set_style_motion(style_motion)
│   └── 设置风格动作参考
├── inference(audio, clip_length)
│   └── 从音频推理运动序列
├── rendering(audio, pred_motions, shape_id, shape_code, save_name)
│   └── 渲染3D动画并保存视频
└── smooth_motion_savgol(motion_codes) [static]
    └── Savitzky-Golay滤波平滑运动
```

### 1. `__init__()` - 初始化

**函数签名**:
```python
def __init__(self, load_gaga=False, fix_pose=False, clip_length=750, device='cuda')
```

**参数说明**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `load_gaga` | bool | False | 是否加载GAGAvatar渲染器 |
| `fix_pose` | bool | False | 是否固定头部姿态 |
| `clip_length` | int | 750 | 最大渲染帧数 (25fps下30秒) |
| `device` | str | 'cuda' | 运行设备 |

**初始化流程**:

```
1. 加载配置和模型权重
   ├── config.json (模型配置)
   └── ARTalk_wav2vec.pt (预训练权重)

2. 初始化 BitwiseARModel
   ├── audio_encoder: wav2vec2 (1024维) 或 mimi (512维)
   ├── style_encoder: StyleEncoder (128维)
   ├── basic_vae: BITWISE_VAE (编码解码器)
   └── attn_blocks: Transformer layers (depth=16)

3. 初始化 FLAME 和渲染器
   ├── FLAMEModel (n_shape=300, n_exp=100)
   └── RenderMesh (512×512, mesh渲染)

4. (可选) 初始化 GAGAvatar
   └── GAGAvatar (高斯splat渲染器)
```

**代码详解** (inference.py:18-40):

```python
class ARTAvatarInferEngine:
    def __init__(self, load_gaga=False, fix_pose=False, clip_length=750, device='cuda'):
        self.device = device
        self.fix_pose = fix_pose
        self.clip_length = clip_length

        # 设置音频编码器类型
        audio_encoder = 'wav2vec'

        # 加载模型权重
        ckpt = torch.load('./assets/ARTalk_{}.pt'.format(audio_encoder),
                         map_location='cpu', weights_only=True)

        # 加载配置文件
        configs = json.load(open("./assets/config.json"))
        configs['AR_CONFIG']['AUDIO_ENCODER'] = audio_encoder

        # 初始化ARTalk模型
        self.ARTalk = BitwiseARModel(configs).eval().to(device)
        self.ARTalk.load_state_dict(ckpt, strict=True)

        # 初始化FLAME模型
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=1.0,
                                     no_lmks=True).to(device)

        # 初始化网格渲染器
        self.mesh_renderer = RenderMesh(image_size=512,
                                       faces=self.flame_model.get_faces(),
                                       scale=1.0)

        # 创建输出目录
        self.output_dir = 'render_results/ARTAvatar_{}'.format(audio_encoder)
        os.makedirs(self.output_dir, exist_ok=True)

        # (可选) 初始化GAGAvatar
        if load_gaga:
            from app.GAGAvatar import GAGAvatar
            self.GAGAvatar = GAGAvatar().to(device)
            self.GAGAvatar_flame = FLAMEModel(n_shape=300, n_exp=100,
                                             scale=5.0, no_lmks=True).to(device)
```

### 2. `set_style_motion()` - 设置风格动作

**函数签名**:
```python
def set_style_motion(self, style_motion)
```

**功能**: 加载或设置风格参考动作序列

**参数**:
- `style_motion`: str 或 Tensor
  - str: 风格文件名 (从 `assets/style_motion/` 加载)
  - Tensor: 形状必须为 `[50, 106]`

**代码详解** (inference.py:41-45):

```python
def set_style_motion(self, style_motion):
    # 如果是字符串，从文件加载
    if isinstance(style_motion, str):
        style_motion = torch.load('assets/style_motion/{}.pt'.format(style_motion),
                                 map_location='cpu', weights_only=True)

    # 验证形状: [50帧, 106维motion参数]
    assert style_motion.shape == (50, 106), \
        f'Invalid style_motion shape: {style_motion.shape}.'

    # 添加batch维度并转移到设备
    self.style_motion = style_motion[None].to(self.device)
```

### 3. `inference()` - 核心推理函数

**函数签名**:
```python
def inference(self, audio, clip_length=None)
```

**功能**: 从音频推理生成运动参数序列

**输入**:
- `audio`: Tensor `[audio_length]` - 16kHz 单声道音频
- `clip_length`: int - 输出帧数限制 (None则使用默认值)

**输出**:
- `pred_motions`: Tensor `[T, 106]` - 运动参数序列

**推理流程图**:

```
audio [16000*T_sec]
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 准备输入                          │
│    audio_batch = {                  │
│        'audio': audio[None],        │
│        'style_motion': style_motion │
│    }                                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. BitwiseARModel.inference()       │
│    ├── 风格编码                      │
│    ├── 音频分块编码                  │
│    ├── 多尺度自回归生成              │
│    └── VAE解码                       │
└─────────────────────────────────────┘
    │ pred_motions [T, 106]
    ▼
┌─────────────────────────────────────┐
│ 3. Savitzky-Golay滤波平滑           │
│    motion_smoothed =                │
│    smooth_motion_savgol(motions)    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. 后处理                            │
│    ├── 截断到clip_length             │
│    └── (可选) 固定姿态               │
└─────────────────────────────────────┘
    │
    ▼
pred_motions [clip_length, 106]
```

**代码详解** (inference.py:47-57):

```python
def inference(self, audio, clip_length=None):
    # 构建输入batch
    audio_batch = {
        'audio': audio[None].to(self.device),      # [1, audio_len]
        'style_motion': self.style_motion          # [1, 50, 106]
    }

    print('Inferring motion...')
    # 调用BitwiseARModel推理
    pred_motions = self.ARTalk.inference(audio_batch,
                                        with_gtmotion=False)[0]  # [T, 106]

    # 确定输出长度
    clip_length = clip_length if clip_length is not None else self.clip_length

    # Savitzky-Golay滤波平滑
    pred_motions = self.smooth_motion_savgol(pred_motions)[:clip_length]

    # (可选) 固定头部姿态
    if self.fix_pose:
        pred_motions[..., 100:103] *= 0.0  # 全局旋转置零

    print('Done!')

    # 清除全局平移 (保留旋转和表情)
    pred_motions[..., 104:] *= 0.0

    return pred_motions
```

**Motion参数结构**:

```
pred_motions: [T, 106]
├── [0:100]   → Expression (表情参数)
├── [100:103] → Global rotation (全局旋转，可fix)
├── [103:106] → Jaw rotation (下颌旋转)
└── [106:]    → Global translation (全局平移，强制清零)
```

### 4. `rendering()` - 渲染函数

**函数签名**:
```python
def rendering(self, audio, pred_motions, shape_id="mesh",
              shape_code=None, save_name='ARTAvatar.mp4')
```

**功能**: 将运动序列渲染为视频

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `audio` | Tensor | 原始音频 (用于视频音轨) |
| `pred_motions` | Tensor `[T, 106]` | 运动参数序列 |
| `shape_id` | str | "mesh" 或 GAGAvatar ID |
| `shape_code` | Tensor `[1, 300]` | 形状参数 (None则用零向量) |
| `save_name` | str | 保存文件名 |

**渲染流程**:

```
选择渲染模式:
├── shape_id == "mesh" → 简单网格渲染
│   │
│   ├── 1. 准备shape_code
│   │      └── 默认: zeros(T, 300)
│   │
│   ├── 2. FLAME生成顶点
│   │      verts = flame_model(shape, motion)
│   │
│   └── 3. 网格渲染 (逐帧)
│          for v in verts:
│              rgb = mesh_renderer(v)
│
└── shape_id ∈ GAGAvatar IDs → 高斯渲染
    │
    ├── 1. 设置avatar
    │      GAGAvatar.set_avatar_id(shape_id)
    │
    ├── 2. 逐帧渲染
    │      for motion in pred_motions:
    │          batch = build_forward_batch(motion)
    │          rgb = GAGAvatar.forward_expression(batch)
    │
    └── 3. 输出
           └── 512×512 RGB + 水印
```

**代码详解** (inference.py:59-87):

```python
def rendering(self, audio, pred_motions, shape_id="mesh",
              shape_code=None, save_name='ARTAvatar.mp4'):
    print('Rendering...')
    pred_images = []

    if shape_id == "mesh":
        # ========== 简单网格渲染 ==========

        # 准备shape code
        if shape_code is None:
            # 默认: 全零shape (平均脸)
            shape_code = audio.new_zeros(1, 300).to(self.device) \
                              .expand(pred_motions.shape[0], -1)
        else:
            assert shape_code.dim() == 2, f'Invalid shape_code dim'
            assert shape_code.shape[0] == 1
            shape_code = shape_code.to(self.device) \
                                  .expand(pred_motions.shape[0], -1)

        # FLAME生成顶点序列
        verts = self.ARTalk.basic_vae.get_flame_verts(
            self.flame_model, shape_code, pred_motions, with_global=True
        )

        # 逐帧渲染
        for v in tqdm(verts):
            rgb = self.mesh_renderer(v[None])[0]
            pred_images.append(rgb.cpu()[0] / 255.0)

    else:
        # ========== GAGAvatar高斯渲染 ==========

        # 设置avatar ID
        self.GAGAvatar.set_avatar_id(shape_id)

        # 逐帧渲染
        for motion in tqdm(pred_motions):
            # 构建batch (包含FLAME顶点)
            batch = self.GAGAvatar.build_forward_batch(
                motion[None], self.GAGAvatar_flame
            )
            # 高斯渲染
            rgb = self.GAGAvatar.forward_expression(batch)
            pred_images.append(rgb.cpu()[0])

    print('Done!')

    # ========== 保存视频 ==========
    print('Saving video...')
    pred_images = torch.stack(pred_images)  # [T, 512, 512, 3]

    # 截断音频到匹配视频长度
    audio = audio[:int(pred_images.shape[0]/25.0*16000)]

    # 保存MP4 (25fps, AAC音频)
    dump_path = os.path.join(self.output_dir, '{}.mp4'.format(save_name))
    write_video(pred_images*255.0, dump_path, 25.0, audio, 16000, "aac")

    print('Done!')
```

### 5. `smooth_motion_savgol()` - 运动平滑

**函数签名**:
```python
@staticmethod
def smooth_motion_savgol(motion_codes)
```

**功能**: 使用 Savitzky-Golay 滤波器平滑运动序列

**算法原理**:
- Savitzky-Golay 滤波器是一种**多项式平滑滤波器**
- 在滑动窗口内拟合多项式，用拟合值替换中心点
- 能保持信号的高频特征（相比简单移动平均）

**参数设置**:
| 运动类型 | window_length | polyorder | 说明 |
|---------|---------------|-----------|------|
| 表情+下颌 | 5 | 2 | 轻度平滑 |
| 全局旋转 | 9 | 3 | 强平滑（头部运动） |

**代码详解** (inference.py:89-95):

```python
@staticmethod
def smooth_motion_savgol(motion_codes):
    from scipy.signal import savgol_filter

    # 转为numpy
    motion_np = motion_codes.clone().detach().cpu().numpy()

    # 平滑所有参数 (window=5, order=2)
    motion_np_smoothed = savgol_filter(
        motion_np, window_length=5, polyorder=2, axis=0
    )

    # 额外平滑全局旋转 (window=9, order=3)
    motion_np_smoothed[..., 100:103] = savgol_filter(
        motion_np[..., 100:103], window_length=9, polyorder=3, axis=0
    )

    # 转回tensor
    return torch.tensor(motion_np_smoothed).type_as(motion_codes)
```

**滤波效果示意**:

```
原始motion (有抖动):
  │ /\  /\  /\  /\
  │/  \/  \/  \/  \

平滑后:
  │   ___/‾‾‾\___
  │ /            \
```

## Gradio界面: run_gradio_app()

**函数签名**:
```python
def run_gradio_app(engine)
```

**功能**: 启动Gradio Web界面

### 界面布局

```
┌─────────────────────────────────────────────────────────────┐
│                         ARTalk                              │
├─────────────────────────────────────────────────────────────┤
│  Input Audio & Text    │  Avatar Control  │  Generated Video│
│  ┌──────────────────┐  │ ┌──────────────┐ │ ┌────────────┐ │
│  │ ○ Audio          │  │ │ Appearance   │ │ │            │ │
│  │ ○ Text           │  │ │  ├─ mesh     │ │ │   Video    │ │
│  │                  │  │ │  ├─ 11.jpg   │ │ │   Player   │ │
│  │ [Upload Audio]   │  │ │  └─ 12.jpg   │ │ │            │ │
│  │      或          │  │ │              │ │ └────────────┘ │
│  │ [Input Text]     │  │ │ Style        │ │                │
│  │ [Select Lang]    │  │ │  ├─ default  │ │ [Download .pt] │
│  │                  │  │ │  ├─ natural_0│ │                │
│  └──────────────────┘  │ │  └─ happy_1  │ │                │
│                        │ └──────────────┘ │                │
│       [Generate Button]                                     │
├─────────────────────────────────────────────────────────────┤
│                        Examples                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心回调: process_audio()

**流程图** (inference.py:99-125):

```
用户点击 [Generate]
    │
    ▼
┌─────────────────────────┐
│ 1. 输入验证              │
│    - Audio/Text 非空     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 2. (如果Text) TTS生成   │
│    gtts → tts_output.wav│
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 3. 音频预处理            │
│    - 重采样到16kHz       │
│    - 单声道              │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 4. 设置风格              │
│    engine.set_style_    │
│    motion(style_id)     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 5. 推理                  │
│    pred_motions =       │
│    engine.inference()   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 6. 渲染                  │
│    engine.rendering()   │
└─────────────────────────┘
    │
    ▼
返回: (video_path, motion_path)
```

**代码详解** (inference.py:99-125):

```python
def process_audio(input_type, audio_input, text_input,
                 text_language, shape_id, style_id):
    # ========== 输入验证 ==========
    if input_type == "Audio" and audio_input is None:
        gr.Warning("Please upload an audio file")
        return None
    if input_type == "Text" and (text_input is None or len(text_input.strip()) == 0):
        gr.Warning("Please input text content")
        return None

    # ========== TTS处理 ==========
    if input_type == "Text":
        gtts_lang = {
            "English": "en", "中文": "zh", "日本語": "ja",
            "Deutsch": "de", "Français": "fr", "Español": "es"
        }
        tts = gTTS(text=text_input, lang=gtts_lang[text_language])
        tts.save("./render_results/tts_output.wav")
        audio_input = "./render_results/tts_output.wav"

    # ========== 音频加载 ==========
    audio, sr = torchaudio.load(audio_input)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

    # ========== 设置风格 ==========
    if style_id == "default":
        engine.style_motion = None  # 使用默认风格
    else:
        engine.set_style_motion(style_id)

    # ========== 推理 ==========
    pred_motions = engine.inference(audio)

    # ========== 渲染 ==========
    save_name = f'{audio_input.split("/")[-1].split(".")[0]}_{style_id.replace(".", "_")}_{shape_id.replace(".", "_")}'
    engine.rendering(audio, pred_motions, shape_id=shape_id, save_name=save_name)

    # ========== 保存motion ==========
    torch.save(pred_motions.float().cpu(),
              os.path.join(engine.output_dir, '{}_motions.pt'.format(save_name)))

    # 返回视频和motion文件路径
    return (
        os.path.join(engine.output_dir, '{}.mp4'.format(save_name)),
        os.path.join(engine.output_dir, '{}_motions.pt'.format(save_name))
    )
```

## 主程序入口

**代码详解** (inference.py:213-238):

```python
if __name__ == '__main__':
    # 设置浮点精度 (加速matmul)
    torch.set_float32_matmul_precision('high')

    # ========== 命令行参数 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', '-a', default=None, type=str)
    parser.add_argument('--clip_length', '-l', default=750, type=int)
    parser.add_argument("--shape_id", '-i', default='mesh', type=str)
    parser.add_argument("--style_id", '-s', default='default', type=str)
    parser.add_argument("--run_app", action='store_true')
    args = parser.parse_args()

    # ========== 初始化引擎 ==========
    engine = ARTAvatarInferEngine(
        load_gaga=True,           # 加载GAGAvatar
        fix_pose=False,           # 不固定姿态
        clip_length=args.clip_length
    )

    # ========== 启动模式选择 ==========
    if args.run_app:
        # Web界面模式
        run_gradio_app(engine)
    else:
        # 命令行模式
        shape_id = 'mesh' if args.shape_id not in engine.GAGAvatar.all_gagavatar_id.keys() else args.shape_id

        # 加载音频
        audio, sr = torchaudio.load(args.audio_path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

        # 设置风格
        base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        save_name = f'{base_name}_{args.style_id.replace(".", "_")}_{args.shape_id.replace(".", "_")}'
        engine.set_style_motion(args.style_id)

        # 推理并渲染
        pred_motions = engine.inference(audio)
        engine.rendering(audio, pred_motions, shape_id=args.shape_id, save_name=save_name)
```

## 使用示例

### 1. Web界面模式

```bash
python inference.py --run_app
```

然后访问 `http://localhost:8960`

### 2. 命令行模式

```bash
# 使用mesh渲染
python inference.py \
    -a demo/eng1.wav \
    -i mesh \
    -s natural_0 \
    -l 750

# 使用GAGAvatar渲染
python inference.py \
    -a demo/eng1.wav \
    -i 12.jpg \
    -s happy_1 \
    -l 500
```

### 3. Python API

```python
from inference import ARTAvatarInferEngine
import torchaudio

# 初始化
engine = ARTAvatarInferEngine(load_gaga=True)

# 加载音频
audio, sr = torchaudio.load("demo/eng1.wav")
audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

# 设置风格
engine.set_style_motion("natural_0")

# 推理
motions = engine.inference(audio, clip_length=500)

# 渲染
engine.rendering(audio, motions, shape_id="12.jpg", save_name="result")
```

## 关键技术点

### 1. 音频采样率统一

所有音频必须转为 **16kHz 单声道**:

```python
audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
```

### 2. 帧率与音频同步

- 视频帧率: 25 FPS
- 音频采样率: 16000 Hz
- 换算关系: `T_frames = ceil(audio_length / 16000 * 25)`

### 3. Motion裁剪策略

```python
# 音频长度匹配视频
audio = audio[:int(n_frames / 25.0 * 16000)]
```

### 4. 批处理优化

虽然代码支持batch，但inference时强制 `batch_size=1`:

```python
assert batch_size == 1, "Only support batch size 1 for inference."
```

原因: 自回归生成依赖历史状态，难以并行

## 性能优化建议

1. **预加载模型**: 避免重复初始化
2. **使用FP16**: `torch.cuda.amp.autocast()`
3. **限制clip_length**: 减少内存占用
4. **禁用GAGAvatar**: 提升5倍速度

## 常见问题

### Q1: CUDA out of memory?

**A**: 减小 `clip_length` 或使用CPU

```python
engine = ARTAvatarInferEngine(clip_length=250, device='cpu')
```

### Q2: 渲染速度慢?

**A**: 关闭GAGAvatar

```python
engine = ARTAvatarInferEngine(load_gaga=False)
```

### Q3: 如何添加新的风格?

**A**: 将 `[50, 106]` 的motion tensor保存到 `assets/style_motion/`

```python
torch.save(your_style_motion, 'assets/style_motion/my_style.pt')
engine.set_style_motion('my_style')
```

## 依赖关系

```
inference.py
├── app.models.BitwiseARModel
├── app.flame_model.FLAMEModel
├── app.flame_model.RenderMesh
├── app.GAGAvatar.GAGAvatar (可选)
├── app.utils_videos.write_video
├── torchaudio (音频IO)
├── gradio (Web界面)
└── gtts (文本转语音)
```

## 总结

`inference.py` 是整个系统的**指挥中心**，协调各个模块完成从音频到视频的完整流程：

1. **初始化**: 加载所有模型和渲染器
2. **推理**: 调用BitwiseARModel生成motion
3. **渲染**: 选择mesh或GAGAvatar渲染
4. **输出**: 保存MP4视频和motion文件

理解这个文件是理解整个ARTalk系统的关键入口。
