# BokehMe 管线详解（中文笔记）

> **本文档由本项目（aberration-bokeh）撰写**，对照本仓库实际代码逐行核实，
> 用于吃透"技术骨架来源"（见项目 DECISIONS D3：BokehMe v1 代码动手 + BokehMe++ 论文参考）。
> ⚠️ 本目录在主项目中被 gitignore，本文件不入主项目版本库；主项目侧的入口链接在 references.md。
>
> 论文：*BokehMe: When Neural Rendering Meets Classical Rendering*（CVPR 2022 Oral）
> 续作：*BokehMe++*（TPAMI 2025，未开源，结构同源）

---

## 1. 一句话思想

**经典渲染器**物理可控、任意分辨率、任意模糊量，但在**深度边界**处必然出错
（深度图本身在边界就不可靠 + 散射模型对遮挡的处理是近似）；
**神经渲染器**能把边界学对，但难以任意分辨率/任意模糊量地泛化。
BokehMe 的答案：**两个都要——用一张神经网络预测的"误差图"把两者融合**：

```
bokeh_pred = bokeh_classical · (1 − error_map) + bokeh_neural · error_map
```

error_map 大的地方（≈深度边界带）信神经，其余广大区域信物理。
这一思想被本项目完整继承（我们的细化网同样只负责"修边界"）。

---

## 2. 输入与控制参数（demo.py 第 116–128 行）

| 参数 | 默认 | 含义 |
|---|---|---|
| `image_path` / `disp_path` | inputs/21.jpg/.png | RGB 图 + **同名灰度视差图**（0~255，读入后归一到 0~1，近大远小） |
| `K` | 60 | 模糊参数（≈最大 CoC 直径像素量级） |
| `disp_focus` | 90/255 | 对焦视差（0~1） |
| `gamma` | 4（范围 1~5） | 伽马：渲染在 `image**gamma` 的"准线性域"进行，γ 越大高光越亮斑化 |
| `--highlight` + 阈值/比率 | 关 | 可选高光增强（见 §3 Step 0） |
| `defocus_scale` | 10 | 内部缩放：`defocus = K·(disp − disp_focus)/10`，把 CoC 压到网络友好的数值范围 |

核心换算（demo.py 第 178 行）：**带符号离焦图**
`defocus = K·(disp − disp_focus)/defocus_scale` ——
正负区分焦前/焦后。这正是本项目 `signed_coc()` 的同款公式（我们多了对焦容差 δ0）。

---

## 3. 管线分步（对照 demo.py `pipeline()`，第 35–76 行）

### Step 0（可选）：高光增强（第 168–175 行）

让过曝点源在虚化后变成明亮散景球（JPEG 把 >1 的真实亮度截断了，这里"还魂"）：

```python
mask1 = clip(tanh(200·(|disp − disp_focus|² − 0.01)))   # 离焦区（焦内不增强）
mask2 = clip(tanh(10·(image − 220/255)))                 # 高亮像素（软阈值）
image *= 1 + mask1·mask2·0.4                             # 增强 40%
```

对照：本项目的 `srgb_to_linear()+highlight_gain` 思路同源，
但我们在**线性光强域**做（物理上更正），且阈值/增益可微。

### Step 1：经典渲染器（classical_renderer/scatter.py，CUDA/cupy 实现）

**散射（scatter）式渲染**：每个源像素把自己的能量"撒"到以自己为圆心、
半径 `|defocus|` 的圆盘内（demo.py 第 36 行先 `image**gamma` 进入准线性域）：

```cuda
// scatter.py 内嵌 CUDA kernel（第 30–48 行），对每个源像素：
weight = (0.5 + 0.5·tanh(4·(R − dist))) / (R² + 0.2)   // 软边圆盘 + 能量归一
atomicAdd(bokehCum,  weight · image[源像素])             // 颜色累加
atomicAdd(weightCum, weight)                             // 权重累加
bokeh = bokehCum / weightCum                             // 归一化
```

要点：
- `tanh(4(R−dist))` = 软化的圆盘边缘（类比我们光瞳遮罩的 sigmoid 软化）；
- `1/(R²+0.2)` 让大圆盘摊薄能量 → 能量守恒；
- **天然处理"散射式遮挡"**：前景像素的能量会撒到背景上方（颜色直接竞争），
  没有显式分层——这是与本项目"分层 gather + alpha 合成"的本质区别；
- 同时输出 `defocus_dilate`（被散射"碰到"的最大 defocus 膨胀图），
  供 Step 3 判断"哪些区域接近焦内、需要精修"；
- ⚠️ **不可微**：`_FunctionRender.backward` 被注释掉（scatter.py 第 158–160 行），
  且依赖 cupy 手写 kernel。**这正是本项目要替换它的核心理由之一**
  （我们的渲染器全链路可微，才能支撑指纹反演）。
- `scatter_ex.py` 是扩展版：支持 `poly_sides=6` 等**多边形光圈**（demo 第 37 行注释里有用法）。

### Step 2：ARNet —— 低分辨率"自适应渲染"（demo.py 第 42–48 行）

神经部分的聪明之处：**把图像缩到"最大模糊 ≈ 1px"的尺度再渲染**：

```python
adapt_scale = max(|defocus|.max(), 1)        # 比如最大 CoC=32 → 缩小 32 倍
image_re   = interpolate(image,   1/adapt_scale)
defocus_re = interpolate(defocus, 1/adapt_scale) / adapt_scale   # defocus 同步缩
bokeh_neural, error_map = arnet(image_re, defocus_re, gamma)
```

任意大的模糊量都被归一到网络见过的范围（"scale-arbitrary"），
这是它能泛化到任意 K 的关键。ARNet 同时输出 **error_map**（sigmoid，第 149 行），
即"我认为经典渲染哪里不可靠"——监督来自训练时经典渲染与 GT 的真实误差。

### Step 3：IUNet —— 迭代上采样精修（demo.py 第 53–72 行）

低清 bokeh_neural 要回到全分辨率。直接放大会糊掉焦内/边界的细节，于是：

```python
for scale in range(log2(adapt_scale)):       # 每次 ×2，逐级回到全分辨率
    bokeh_refine = iunet(image_re, defocus_re.clamp(-1,1), bokeh_neural, gamma)
    mask = gaussian_blur((|defocus_dilate_re| < 1).float(), ...)   # "近焦带"掩码
    bokeh_neural = mask·bokeh_refine + (1−mask)·upsample(bokeh_neural)
```

精修只发生在 `|defocus_dilate|<1` 的**近焦带**（这里有需要锐利的细节）；
远焦外区域本来就模糊，双线性放大零成本也无损——计算量被花在刀刃上。
掩码用高斯羽化避免拼接痕。

### Step 4：误差图融合（demo.py 第 74 行）

```python
bokeh_pred = bokeh_classical·(1 − error_map) + bokeh_neural·error_map
```

产出（outputs/<名字>/）：`bokeh_pred.jpg`（最终）、`bokeh_classical.jpg`、
`bokeh_neural.jpg`、`error_map.jpg`、`defocus.jpg`（蓝红=焦前后）等，
加 `--save_intermediate` 还能看 IUNet 每级的中间结果（`bokeh_neural_s*.jpg`）。

---

## 4. 网络结构（neural_renderer.py，两个网都很小）

两个网络共用同一套骨架（第 96–208 行）：

```
输入拼接 → Space2Depth(×2 下采样, 通道×4) → 3×3 conv → N 个残差块 → 3×3 conv
        → PixelShuffle(×2 上采样) → 输出
```

| | ARNet | IUNet |
|---|---|---|
| 输入 (in_channels) | 5 = RGB + defocus + γ | 8 = RGB + defocus + 粗 bokeh RGB + γ |
| 输出 | 4 = bokeh RGB + error_map | 3 = bokeh RGB |
| 中间通道 | 128 | 64 |
| 残差块数 | 3（`distinct_source` 连接，出自 LapSRN） | 3 |

**条件注入方式**（值得注意）：γ 被归一化到 0~1（demo 第 41 行
`(γ−γmin)/(γmax−γmin)`）后，作为**常数通道**直接 concat 进输入
（neural_renderer.py 第 140、200 行 `torch.ones_like(x[:,:1])*gamma`）。
这是条件注入的最朴素形态——本项目计划用 **FiLM**（特征级仿射调制）注入
13 维控制向量，表达力更强；但"常数通道"的简单性也值得作为消融基线。

**残差块**：`Space2Depth` 是用 unfold 实现的像素重排（×2 下采样、通道×4），
与 `PixelShuffle` 互为逆操作——整个网络在 1/2 分辨率的特征上工作，省算力。

---

## 5. 训练方式（论文知识，本仓库未附训练代码）

- **数据**：BLB 合成数据集（Blender 渲染），离散深度层场景，配精确 GT。
- **关键 trick——深度扰动**：训练输入侧的视差图加人工扰动（边界错位等），
  GT 用干净深度渲染。网络被迫学会"在深度不可靠时修边界"。
  **本项目 data/synth.py 的"GT 干净、输入脏"完整继承了这一思想。**
- error_map 的监督：经典渲染结果与 GT 的实际误差（阈值化/软化后）作为目标。

---

## 6. 怎么跑（本机注意事项）

```bash
cd third_party/BokehMe
python demo.py --K 60 --disp_focus 0.35 --gamma 4 --highlight --save_intermediate
# 输入：inputs/21.jpg + inputs/21.png（同名视差图）；输出：outputs/21/
```

⚠️ **依赖 cupy**（经典渲染器是 cupy 手写 kernel）：主项目的 `bokeh` env 未装。
要跑的话 `pip install cupy-cuda12x`（匹配 CUDA 12.8）。只想看神经部分/对照结果，
可以读 `outputs/21/` 里仓库自带的预渲染结果。权重已附带（checkpoints/arnet.pth, iunet.pth）。

---

## 7. 与本项目（aberration-bokeh）的对照速查

| BokehMe 的做法 | 本项目的对应 | 关系 |
|---|---|---|
| `defocus = K(D−d_f)/scale` | `signed_coc()`（多了对焦容差 δ0） | **继承** |
| 软边均匀圆盘核（tanh） | 像差化 pupil→PSF（球差/彗差/猫眼/色差…） | **替换**（创新核心） |
| 散射式渲染（cupy，不可微） | 分层 gather + tile FFT 卷积（torch，全可微） | **替换**（支撑指纹反演） |
| ARNet+IUNet+误差图融合 | refine/network.py 蓝本（M2 待实现） | **继承** |
| γ 常数通道条件注入 | FiLM 注入 13 维 ctrl_vec | **升级**（常通道留作消融） |
| 深度扰动训练 trick | data/synth.py 三连扰动 | **继承并细化** |
| 高光增强（tanh 掩码，sRGB 域） | 线性域 highlight_gain（可微软阈值） | **改良** |
| scatter_ex 多边形光圈 | 光瞳 n_blades 遮罩（顺带衍射星芒潜力） | 物理化的同款 |
| adapt_scale 任意模糊量归一 | 暂未需要（PSF 现场算+核自适应裁剪） | 思路备用 |

一句话：**骨架（混合渲染+误差图融合+深度扰动）来自 BokehMe，
心脏（PSF 物理引擎）和神经条件化（FiLM）是本项目的替换与升级。**

---

## 8. 文件清单

| 文件 | 内容 |
|---|---|
| `demo.py` | 完整推理管线（本笔记 §3 的行号都指它） |
| `neural_renderer.py` | ARNet + IUNet 定义（§4） |
| `classical_renderer/scatter.py` | 经典散射渲染器（cupy CUDA kernel，圆形光圈） |
| `classical_renderer/scatter_ex.py` | 扩展版：可调光圈形状（`poly_sides`） |
| `checkpoints/arnet.pth, iunet.pth` | 官方预训练权重（随仓库附带） |
| `inputs/` | 示例输入：RGB + 同名视差图（13/21/3 三组） |
| `outputs/21/` | 仓库自带的一组预渲染结果（没装 cupy 也能看效果） |
| `pdf/` | 论文 PDF |
