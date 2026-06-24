# 项目进度看板 (PROJECT STATUS)

> **角色**：本项目的「现在做到哪、怎么跑、接下来做什么」的**活文档**，每次推进后更新。
> 最近更新：2026-06-21。
>
> **文档体系**（六份各司其职，避免信息散落）：
> | 文档 | 看什么 |
> |---|---|
> | [GUIDE.md](GUIDE.md) | 入门导读（教学版）：流程 / 技术路线 / 与既有方法异同 / 创新点 / 代码地图 |
> | [CLAUDE.md](CLAUDE.md) | 项目愿景 / 架构 / 像差公式 / 硬约束 / 路线图（稳定，单一事实来源） |
> | **本文件 PROJECT_STATUS.md** | 进度、环境、运行方式、待办、验证记录（活文档） |
> | [DECISIONS.md](DECISIONS.md) | 关键决策与**理由**（只追加；写论文/答辩的依据来源） |
> | [NETWORK_DESIGN.md](NETWORK_DESIGN.md) | 细化网设计路线与创新点（v1 存档 / 红线 / P1–P3 升级 / 数据集策略） |
> | [references.md](references.md) | 参考文献清单 |

---

## 1. 一句话现状（2026-06-22）

> **🔴 重大路线决策（2026-06-22，D49）：镜头指纹反演 (RQ3) 降级为 future work（新阶段 P-FP），
> 主线主攻 RQ1 可控性 + RQ2 轻量性 —— 各类像差可控的轻量化单图散景渲染。** 红线不变（反而更核心，
> 由 RQ1 独立支撑，见 NETWORK_DESIGN §1）；可微性降为保留设计属性（非硬约束）；评测/路线图据此校准。
> **网络职责再界定（D50 / NETWORK_DESIGN §11）**：细化网"必要但收窄" = 合成边界不可知内容 + 抗深度
> 误差，不修全局深度（交深度侧冻结先验）、不修渲染遮挡（交渲染器 Dr.Bokeh）。
> **下一主战场：M3 评测体系（图像层解耦 + 可控性保真，RQ1/RQ2 硬数据）。**

**M2 实质完工，train_run7（NAFNet P2a）50k 完成（NAF 边界 L1 0.0605≈film_res 基线 0.0604，打平）。渲染器多轮修复：①几何光线 PSF（D40）硬边；②高光核心深度统一（D43）消嵌套环；③swirl 增强 + ④叶片曲率参数化（D43/D44）；⑤高光管线检讨（D44/D45）：D44 撤回激进 HDR 放大（gain→0）根治溢出/假弥散圆；D45 把 gamma 4→2.2（标准 sRGB）恢复被 `**(1/4)` 压平的球差盘内对比（soap 亮边/cream 亮心）+ 放宽 auto_focus K 上限修背景虚化不足；⑥焦内锐度（D46）：balance 用最小弥散圆系数 1.3（焦内锐化 35%）+ K 上限回调 100；⑦soap=强球差(W040+W060)忠实复刻肥皂泡锐亮环+暗心(D47)；cream 改用轻球差+变迹apodization(D48)做真奶油柔虚化(纯欠矫正球差是"亮心保留结构=没虚化",变迹才是柔边填充=真虚化+焦内更锐)。效果集 `outputs/samples/review_0622/`。下一步：渲染器核心合成机制打磨（用户主导），细化网重训待核心定稿。** Plan B
（matte 选择式重渲，D29）已根治找补：train_run5 完成后发现 **matte_gt 定义 bug 致主体人脸
被虚化**（局部 min/max 把合焦主体判成中间层），**D30 改用带符号 CoC 锚点定义修复**
（r≥0→α=1 保前景、背景线性衰减），train_run6（50k，COCO，matte+AMP+batch6）验收：
**人脸清晰、无找补、边界 L1 净收益 −0.0033**。随后做了一轮深度优化与文献对照（7 篇精读）：
- **渲染器加固**：D31 自适应分层（修弥散圆中心亮点）、D33 软边缘溯源（物理正确非 bug）+
  新增变迹/球色差、D34 高阶球差 W060（**像差 9→12 种**）+ WR 逐区平顶、**D35 焦带吸附修
  锐前景遮挡漏光**（散景盘半透明叠加不受影响、GT 不变）。
- **架构审查（D36）**：发现并修复①**新像差未进 ctrl_vec→训练失配**（`sample_random(extended=False)`
  默认不在训练采样 W060/变迹/球色差，保持 13 维条件口径；demo/指纹用 PRESETS）；②深度
  min-max→百分位归一（离群稳健）。核验 snap/GT 口径一致、能量守恒、真实性（范围内真实自然）。
- **P2a（NAFNet 块）**：`block_type` 可切换 + 梯度检查点（NAF 全分辨率激活显存大）；首个
  对照 **train_run7 = naf@同预算(144/72,713k) vs film_res(707k)**，验收=合成边界 L1 ≤ 基线。
- **散景盘软边根治（D40，渲染器核心升级）**：散景盘外缘软是【波动光学 |FFT|² 的固有软边】
  （恒 ~13% 盘半径，与 apod/光谱/分辨率无关——受采样上限只能仿真小离焦、停在衍射 regime）。
  新增**几何光线散射 PSF**（`psf_mode='geom'` 默认）：落点 ∝ ∇W、按 A² splat → **硬边**
  （实测边宽 3~4% vs 波动 17~20%）+ soap 亮边环/cream 亮心/猫眼/多边形等焦散自然涌现 + 可微 +
  **更快**（splat vs 9×FFT）+ 无采样上限。`'wave'` 转可选（星芒/衍射/指纹精细反演）。注意：合成
  GT 改 geom → 生产细化网应在 geom 上重训（train_run7 wave 不受影响、架构结论仍有效）。
- **用户反馈两问修复（D38/D39，渲染/深度侧，不动训练）**：① **夜景散景盘暗+硬边发软+像折返镜头**
  →【WR 逆色调映射/饱和 ramp】`peak=1+gain·sat`（旧固定 `1+gain` 盘到不了饱和、灰暗软边）；**盘形=W040
  控、亮度=gain 控两独立旋钮**：用 `W040≈1.5(填充亮边)+gain≈120(亮不炸)`，**别用 W040=3/gain=500**
  （成空心环炸白=折返 donut）。盘外缘 7px 是离焦衍射物理边（非 bug）。验证 `outputs/samples/night/cmp_soap_W1.5_g120.png`。
  ② **背景虚化太弱+女生糊**（自我修正）→ 真因**不是**缺非线性，而是**测试用了低质量缓存 `vitb@518`**
  （把女生脸与背景树 merge 到同视差）+ **tol 过宽**。换默认 **`vitl@770`**（女生 0.198 vs 背景 0.02 本就分开）
  + tol 收紧(0.15→0.10)、**不加 γ** → 女生清晰+背景虚化（`outputs/samples/p2_vitl_g1.0.png`）。
  曾加的 `remap_disparity(γ=0.7)` **已撤回默认关闭**（γ<1 会推高 d_f 虚掉主体）。

**下一主战场（train_run7 后）：M3 评测体系**（图像层解耦矩阵 + 可控性保真），论文 RQ1/RQ2
硬数据，ROI 最高。文献定位铁证：**7 篇 SOTA 0 篇建模像差**（Wu2010/2013 做像差但走全光追、
不可微/需处方），我们=可微+轻量单图+解耦可控+指纹反演。

---

## 2. 里程碑进度（对照 CLAUDE.md 第 10 节）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **M0** | 环境 + 基线复现（DA V2 / BokehMe / 数据集） | ✅ GPU 恢复；DA V2 + optics cuda 跑通；`bokeh` env 就绪 |
| **M1** | pupil→PSF 生成器 + 在线合成；验证肥皂泡/旋焦/单侧亮边 | ✅ PSF 生成器+标定+解耦+12 种像差；在线合成 |
| **M2** | 接入可微渲染 + 训练边界细化网，单卡跑通完整管线 | 🟢 **实质完工**：渲染器✅(D26-D35 多轮加固)、细化网✅(Plan B/D29-D30)、训练✅、e2e✅(人脸清晰/无找补)；P2a 骨干对照 train_run7 收尾中 |
| **M3** | 可控性保真、解耦度、消融实验 | ⚪ **主线主战场（D49）**（图像层解耦矩阵 + 可控性 PSNR + with/without-net & P1b/P2a 消融）|
| **M4** | 真实集评估（降权）+ 论文撰写投稿（**主线收官，不含指纹**，D49）| ⚪ 未开始 |
| **P-FP** | （future，主线投稿后）镜头指纹反演 RQ3：真实样张反求系数 + 单波前保真验证 + BETD 定量 + 风格迁移 demo | ⚪ future（D49 降级，非主线硬约束）|

🟢 进行中且有产出　🟡 部分就位　⚪ 未开始

---

## 3. 模块进度

| 模块 | 文件 | 状态 | 说明 / 验证 |
|---|---|---|---|
| 像差系数 | [optics/aberrations.py](optics/aberrations.py) | ✅ 实现 | **12 种像差**（波前 6：W020/W040/**W060**/W131/W222/W220；振幅 3：多边形/猫眼/**变迹**；色差 3：LoCA/LaCA/**球色差**，D33/D34）；`sample_random(extended=False)` 训练默认只采 ctrl_vec 内 13 维（W060/变迹/球色差 gate 出训练，D36）；9 预设含 nisen/stf/spherochromatic |
| 复光瞳 | [optics/pupil.py](optics/pupil.py) | ✅ 实现 | 波前 W、振幅遮罩（多边形/猫眼，sigmoid 软化保可微）、`complex_pupil`(含 phase_scale 光谱平均接口)、`relative_transmission`(猫眼边角失光 T(H)) |
| PSF | [optics/psf.py](optics/psf.py) | ✅ 实现 | `pupil_to_psf`(\|FFT\|²+归一+混叠/裁剪 guard)、`rgb_psf`(LoCA+LaCA 径向平移)、`build_psf_dictionary`(推理缓存,含 T(H))、`sample_psf` |
| PSF 验证 | [optics/visualize.py](optics/visualize.py) | ✅ 实现并跑通 | 产物见 `outputs/psf_test/`，§6 有验证记录 |
| 离焦标定 | [optics/calibrate.py](optics/calibrate.py) | ✅ 实现并跑通 | `r_px ≈ 4·W020` 实测斜率 3.93，JSON 落盘供渲染器换算（§6） |
| 解耦矩阵 | [optics/decoupling.py](optics/decoupling.py) | ✅ 实现并跑通 | PSF 层面效应签名矩阵，对角占优（§6，DECISIONS D11） |
| 深度 | [depth/estimator.py](depth/estimator.py) | ✅ 实现并跑通(cuda) | 后端无关封装，默认 **DA V2-Base/vitb**（D25）；视差归一化 min-max→**[0.5,99.5] 百分位**（离群稳健，D36）；DA3 留作消融备选（D2） |
| 渲染器 | [render/renderer.py](render/renderer.py) | ✅ 实现并出图 | 分层 gather + tile 视场相关（D8/D9）：全程可微、PSF 现场算、标定换算、T(H) 失光；**WR 平顶方波(D27)+逐区平顶(D34)**、**psf_extent_px 完整外缘裁剪(D26)**、**Plan B B_bg 重渲+α_gt(D29/D30)**、**自适应分层修中心亮点(D31)**、**焦带吸附修锐前景遮挡漏光(D35)**；能量守恒已验证 |
| 渲染 demo | [render/demo.py](render/demo.py) | ✅ 实现并跑通 | 真实照片→视差→6 预设出图，见 §6 |
| 指纹标定 | [render/fingerprint_demo.py](render/fingerprint_demo.py) | ✅ 实现并验证 | RQ3 自标定：两阶段延拓反求 (d_f,K,W040)，误差 <0.5%（§6，DECISIONS D12） |
| 细化网 | [refine/network.py](refine/network.py) | ✅ v2(P1)+Plan B+P2a | ARNet+IUNet+FiLM(13 维)+17ch guide+边界带门控；**Plan B matte 头 706k**（D29，网络只出几何 α 选择式重渲）；**P2a：`block_type='naf'` NAFNet 块可切换**（LayerNorm2d+SimpleGate+SCA，同 ch 约半参→加宽 144/72 同预算 713k）+ 梯度检查点(D32)；`block_type='film_res'` 复现 v1 |
| 条件构建 | [refine/conditioning.py](refine/conditioning.py) | ✅ 新增 (P1) | PSF 描述子表(单色, D11 同口径)→逐像素 9ch 条件图（H/方位角/r_eq/elong/取向/环比/T）+ 视差边缘带门控；`python -m refine.conditioning` 自检 |
| 在线合成 | [data/synth.py](data/synth.py) | ✅ 实现并跑通 | 程序化前景+真实背景裁块+随机像差→(全焦图,扰动视差,散景GT,c) 四元组；GT 走 `composite_blurred_layers` 精确分层；139ms/样本@cuda（§6） |
| 训练 | [train/train.py](train/train.py) | ✅ 实现并跑通 | 在线合成→物理渲染(no_grad)→细化；Plan B 分支(matte BCE-on-logits，D29-D30)；**AMP 混合精度 + 梯度检查点**(P2a)、`--block/--ar-mid/--iu-mid` 选骨干；COCO 118k 背景，train_run6(film_res)✅/train_run7(naf)收尾中 |
| 端到端 demo | [render/e2e_demo.py](render/e2e_demo.py) | ✅ 实现并验证 | 真实照片→DA V2→tile 渲染→整图 matte 选择式细化（Plan B，D29）；按 ckpt 重建网络(block/宽度)；21.jpg 验收：人脸清晰/无找补/无中心亮点 |
| 评测 | [eval/metrics.py](eval/metrics.py) | ⚪ stub | M3 |

**上游基线**（`third_party/`，已 gitignore）：
- [third_party/BokehMe](third_party/BokehMe) — BokehMe v1(CVPR2022)，**自带权重** `arnet.pth`+`iunet.pth` + demo 输入图，就绪。
- [third_party/Depth-Anything-V2](third_party/Depth-Anything-V2) — 代码就位；Small 权重已下载到 `checkpoints/depth_anything_v2_vits.pth`(95MB) 并软链入仓库期望路径。

---

## 4. 环境状态（重要，含阻塞项）

| 项 | 实况 |
|---|---|
| **GPU/CUDA** | ✅ 已恢复（2026-06-10 重启后驱动三层一致 580.159.03）。RTX 5070 12GB，torch cuda 正常 |
| 默认 `python` | anaconda3 **base**（py3.13），未装 torch |
| 现有 torch env | `ultralytics`/`yoloworld`（YOLO 项目用，torch 2.9+cu128）——**勿污染**；目前所有验证暂借它跑，确认无问题 |
| **专用 env `bokeh`** | ✅ 已建成并验收（python3.11 + torch cu128 + requirements.txt；`data.synth` 全管线在其上跑通）。**后续一律用它**：`~/anaconda3/envs/bokeh/bin/python` 或 `conda activate bokeh` |
| CPU 库 | base 已有 numpy/scipy/matplotlib/pillow，足够跑 optics 物理验证 |

---

## 5. 如何运行（Quick Reference）

> 环境就绪后在 `bokeh` env 下、项目根目录执行。

```bash
# 1) PSF 物理验证（不需 GPU，CPU 即可）→ outputs/psf_test/
python -m optics.visualize
python -m optics.calibrate     # 离焦标定：W020↔CoC 像素半径 → defocus_calibration.{png,json}
python -m optics.decoupling    # 解耦矩阵：效应签名 → decoupling_matrix.png

# 2) 深度推理冒烟（需 torch；GPU 或 CPU 皆可）→ outputs/depth_test/
python depth/estimator.py                      # 默认跑 DA V2 仓库自带示例图
python depth/estimator.py path/to/your.jpg     # 指定图片

# 3) DA V2 上游脚本（确认基线）
cd third_party/Depth-Anything-V2 && python run.py --encoder vits \
    --img-path assets/examples --outdir ../../outputs/davis_test && cd ../..

# 4) BokehMe 神经渲染 demo（参考基线）
cd third_party/BokehMe && python demo.py && cd ..

# 5) 本项目渲染器 demo（需 torch+GPU 更快；CPU 也能跑只是慢）→ outputs/render_test/
python -m render.demo                      # 默认 BokehMe demo 图，5 个镜头预设出图 + 夜景灯点面板
python -m render.demo path/to/your.jpg     # 指定图片

# 6) 细化网自检（参数量/形状/可微性/可视化）→ outputs/refine_test/smoke.png
python -m refine.network

# 7) 细化网训练（在线合成，单卡 12GB 内）→ outputs/train_run/
python -m train.train --smoke              # 30 步快速验证
python -m train.train --iters 10000 --out outputs/train_run1   # 验证规模训练
```

---

## 6. 验证记录（已完成的实证）

**[2026-06-23] 7×7 点光源像差验证工具 + 真实性评估 + 三处修正**（决策 **D52**）
- 新建 `render/aberration_grid.py`：7×7 点光源阵列 × 背景/焦平面/前景三工况 × 13 类像差，
  肉眼+定量逐一核对渲染真实性。产物 `outputs/aberration_grid/`（montage + 各 strip + 诊断 `_*.png`）。
- **评估结论**：绝大多数像差物理正确（球差形态反转/彗差彗星状径向一致/像散长轴前后翻转90°/
  场曲 H² 离焦/猫眼切向长轴=旋焦/多边形+星芒联动/变迹高斯渐隐/LaCA 红外蓝内/球色差镶边）。
- **三处修正**：① LoCA 红-蓝轴→**绿-品红轴**(`loca_rgb=(+a,−a,+a)`,焦后绿/焦前品红,真实摄影标志,
  待细化网重训生效)；② **tile 接缝消除**(`render_tiled` 改 50% 重叠+Hann 窗羽化,盘横跨 tile 边界
  不再有十字缝；tile 数约 4×、仅影响 e2e,训练 `render_field_patch` 不受影响)；③ **nisen** W060
  1.3→2.0(二线更分明)+撤回"中途环"误判(大盘下焦散必在外缘,二线性靠外缘尖环+暗心实现)。
- **顺带**：星芒做成与 `n_blades` 联动的演示(非独立预设)、修复 in-focus 显示归一化噪声、
  球差合焦改 `balance_spherical_focus=True`(对焦最小弥散圆,焦内只剩轻 glow)。

**[2026-06-23] tile 路径能量诊断 + 亮度差异溯源**（决策 **D54**）
- 用户问"有的像差虚化后整体亮度与别的不同,是否正确"。溯源出三个来源：① **显示归一化**(验证脚本每图
  除自身 max → peak/mean 不同的像差整体明暗不同,**主因、非物理**)；② **物理正确**(散景盘越大每像素越暗
  =能量守恒、猫眼 T(H) 失光实测总能量 2.59<3、变迹边缘吸收)；③ **tile 路径能量误差**(视场像差 ideal+3%/
  场曲+6%/彗差+9.5%,盘边缘偏亮)。
- tile 误差根因(逐项排除 halo/overlap/归一化后锁定)：tile gather 下点光源散景盘跨多 tile,盘不同部分被
  各自 (H桶,方位角) PSF 渲染再拼接、块间 PSF 不同 → 能量偏离;**强制 azim=0 即完全守恒 3.000**,证实根因
  是 PSF 空变(rotate_psf 插值 + 视场变化),非归一化/halo/overlap。
- 决策：**接受为已知限制**(只影响 e2e 整图推理视觉,**不影响训练**——训练走 render_field_patch 单 PSF 整块=
  守恒);`_composite_tile` 归一化移到全图做一次(口径更一致)。产物 `_unified_brightness.png`(统一亮度=真实
  相对亮度)。回归:验证脚本+demo swirl+import 全通过。

**[2026-06-23] 能量分布系统评估 + balance 补偿 W060 + LoCA/nisen 调幅**（决策 **D53**）
- 用户反馈 LoCA 太夸张、W060(nisen)前后弥散圆大小不一致；借机系统评估 13 类像差能量分布。
- **评估**：能量守恒所有通道 sum=1.000✓；前后盘对称(球差类不对称是物理±20%;nisen 修前异常 2.53×)；
  能量集中度 peak/mean 均合理(ideal 1.07/球差 2~2.3/nisen 3~4/变迹 4/色差 1.2~1.6)、无非物理奇点。
- **修**：① **balance 补偿 W060**(最小弥散圆系数1.5,原只补 W040)——根治 nisen 前后不对称(1.88→1.26)
  +焦内锐化(9.7→3px),根因是 balance 不补 W060 的人为加剧;② nisen W060 2.0→1.3(撤 D52);
  ③ LoCA 验证幅度 1.5→0.6(盘边细镶边非整盘染色;sample_random 0~0.4 本就真实,不变)。
- 训练分布不变(extended=False 不采 W060)、球差 under/over(W060=0) 形态不变;回归 soap/cream/nisen 有限值✓。

**[2026-06-22] cream 改用轻球差+变迹：修正"亮心没虚化"**（决策 **D48**）
- 用户反馈 cream PSF 中间亮周围暗 → "没虚化只变糊"。根因：纯欠矫正球差 PSF=圆顶/亮心 → 卷积保留焦外
  结构（Helios 式柔化≠虚化，实测背景枝条隐约可见 vs ideal 完全化开）。
- 解法：真正奶油【虚化】靠【变迹 apodization】（STF 式柔边填充盘），非欠矫正球差。cream=W040=−1.0（略亮心）
  +apodization=1.5（柔边填充虚化）。三得：柔虚化+略亮心+焦内更锐（变迹焦内影响远小于强球差，修了 D47 焦内软）。
- 验证：real21 cream 背景柔化开+焦内女生更清晰；night 盘=柔边填充盘；回归无崩溃。soap 保持 D47 强球差。
- 教训：奶油柔虚化应靠变迹（振幅），非欠矫正球差（相位/亮心）——后者是 Helios 式亮心保留结构。

**[2026-06-22] soap/cream 忠实复刻真实镜头：强球差 W040+W060**（决策 **D47**，cream 部分被 D48 修正）
- 联网对照真实标准（Trioplan 评测）：soap=锐亮环+暗心填充、cream=亮心柔渐隐。旧中等球差=soap 碗状弱环、
  cream 几乎平（不像真实）。物理 trade-off（强特征↔焦内锐）中用户选"忠实复刻"（接受焦内 glow）。
- **W060 是关键**：仅 W040 大只是碗更深；加高阶 W060 才把 soap 塑成"平底+陡锐环"（边/心 1.6→7.9）、
  cream 塑成"圆顶亮心渐隐"。soap W040=2.0+W060=0.8、cream W040=−2.0+W060=−0.8。
- 验证：night panel soap=肥皂泡亮环暗心、cream=奶油亮心（像真实）；real21 灯串二线性/亮心明显；回归无崩溃。
- W060 是扩展像差（不进训练采样）；预设用于 demo/showcase/指纹，训练走 sample_random 不受影响。

**[2026-06-22] 焦内锐度修复（balance 最小弥散圆 c=1.3）+ 撤回 soap W040=2.0 + K 回调**（决策 **D46**）
- **焦内软（soap/cream）**：旧 balance 用最小 RMS 系数（W020=−W040），欠补偿球差焦移。改用**最小弥散圆**
  系数 1.3（W020=−1.3·W040）——摄影对焦准则。焦内 90% 能量半径 W040=2.0 时 5.8→3.7px、1.5 时 4.3→2.9px
  （锐化 ~35%），焦外亮边几乎不变。soap W040 撤回 2.0→1.5（D45 增大致焦内过软）。
- **澄清"太多东西当点光源"**：gain=0 起管线**无点光源检测/放大**；红球/人群成盘是纯虚化（散景本质），
  soap 亮边 + 过大 K 让盘显眼。非误检测。K 上限 120→100（K 158→132）让背景虚化适中、盘更自然。
- **验证**：real21 焦内女生清晰（接近 ideal）+ 背景虚化适中 + 红球/人群盘温和；全预设回归无崩溃。
- **教训**：追焦外特征（亮边）不应靠加大像差系数（恶化焦内）；用物理对焦准则（最小弥散圆）解耦焦内/焦外。

**[2026-06-22] gamma 回归 2.2（修正 D44）+ 背景虚化修复 + soap 增强**（决策 **D45**，soap W040 部分被 D46 回调）
- **①②③ gamma=4 压平球差对比**：geom soap 盘亮边线性域 6.5×，`**(1/γ)` 显示编码 γ=4→1.6×（几乎消失）、
  γ=2.2→2.3×（保留）。D44 为提亮散景球用高 γ，副作用=幂律压平 soap 亮边/cream 亮心 + 像素响应过强/颗粒。
  改回标准 sRGB **gamma=2.2**（demo/e2e/night）→ 球差对比恢复、更柔和。
- **④ 背景虚化不足**：auto_focus "最大 CoC 60px" 上限（旧 wave 采样上限遗留）被极近前景(视差0.8)撑大 max_dev
  → K 压到 79、背景 CoC 仅 10px。geom 无采样上限 → 放宽 120px + 背景目标 30px → K=158、背景 CoC 20px。
- **soap 二线性**：W040 1.5→2.0（真实 Trioplan 级过矫正更强）→ 枯枝虚化双线更明显。
- **验证**：night panel soap 亮边环/cream 亮心恢复、real21 背景虚化翻倍、整体柔和（`review_0622/`）。
- **教训**：gamma 一个旋钮兼显示编码+亮斑化→调它隐藏压对比副作用；职责应分离（编码用2.2、亮斑用乘性）。

**[2026-06-22] 高光管线根本检讨（D44）：回归 BokehMe gamma 域亮斑化 + 叶片曲率**（决策 **D44**，gamma 部分被 D45 修正）
- **根因**：用户反馈点光源溢出/能量放大几倍 + 原图无点光源处冒假弥散圆 + 不如三天前自然。对照 BokehMe
  论文坐实：D38 把【可选+温和(×1.4默认关)】的高光增强做成了【默认+激进(gain30=放大30倍)】。BokehMe 散景球
  靠 **gamma=4 准线性域** `render(image**γ)**(1/γ)` 增亮（保持亮点亮度、不溢出、无模糊区恒等），非能量放大。
  我们从 LDR 猜测放大 30 倍 → 真点光源溢出 + 天空(>0.82占88.7%)/雪/脸被绝对阈值误判放大成假弥散圆。
- **修复**：demo/e2e/lights_panel 改 **gamma=4 + highlight_gain=0**（纯 gamma 域亮斑化）；D38/D41/D42 放大
  机制保留为可选(默认关)。新增光圈叶片曲率 `blade_curvature∈[0,1]`(0直边→1圆，角保持尖)，hexagon 预设 0.25。
- **验证**：real21 灯串从死白连片→柔和暖金有层次、前景雪地不再冒假弥散圆；9 预设形态清晰、hexagon 圆角弧边；
  全预设+sample_random(含曲率)回归无崩溃。效果集 `review_0622/`（00 grid/02 night 9 预设）。
- **教训**：对照别家方案要看其【默认行为与量级】，而非只看机制名称；缺 HDR 输入时不应从 LDR 猜测激进放大。

**[2026-06-22] 用户反馈 4 问题修复：灯珠过曝/嵌套环/swirl**（决策 **D42/D43**，部分被 D44 修正）
- **①过曝死白（D42）**：实测原图灯串核心确实饱和(0.89%>0.98)，但 WR `sat` 从检测阈值 0.82 起放大
  → d=0.9 中等灯珠被放大 27 倍。新增 `highlight_hdr_thresh=0.92`，HDR 放大只对真饱和(d→1)生效；
  demo/e2e gain 60→30。验证：灯串死白 26.9%→11.3%、彩色度 0.271→0.333、中等灯珠 peak 27→1。
- **②③嵌套环（D43）**：证明非 PSF bug（geom 单盘边/心=1.00、孤立灯珠干净）；真因 DA V2 对小灯珠
  深度估计不可靠(同灯珠像素 std 0.36→CoC 差 28px)→分层叠加。新增 `unify_highlight_core_depth`
  (核心区统一为最亮像素深度)，demo/e2e 集成。验证：球从"暗心亮边"→"均匀实心"、副作用极小。
- **④swirl 增强（D43）**：vignette_strength 0.5→0.9、radius 1.2→1.05；四角盘明显切成朝中心眼形。
- **澄清**：密集灯串相邻球加性重叠的蜂窝亮纹是物理真实(真实散景同样)，非 bug，D42 降 gain 已减轻。
- **回归**：全 9 预设×geom/wave×新参数无崩溃/无溢出；demo 21.jpg 端到端跑通(深度统一 2824 核心)。
  效果集 `outputs/samples/review_0622/`（00 网格/01 灯串放大/02 夜景 9 预设/03 D40/04 D41）。

**[2026-06-22] 散景盘重叠亮度溢出根治：HDR 软肩色调映射**（决策 **D41**）
- **诊断**：用户反馈"焦外盘叠在一起时亮度溢出"。实测密集灯串（100 点）**40% 像素线性能量 >1.0**
  （最高 1.3~1.4）。根因：散景盘是加性 HDR 光（同层 `out=conv(lin,psf)`，重叠处相加=物理正确），
  但旧 `linear_to_srgb` 只 `clamp(min=0)`、不处理上界 → 超 1 部分保存为 8-bit 时硬裁成**平白块**。
- **修复**：联网查证（MJP/Wronski/DoF 专利均为"HDR 加性 + 末端色调曲线滚降"，非硬裁）→ 在 gamma
  前插入软肩 `_soft_shoulder`：`L≤0.8` 恒等（不动主体/曝光）、`L>0.8` 指数滚降逼近 1。`highlight_tonemap`
  默认开、仅 `gain>0` 启用。**不用全局 Reinhard**（会压暗中间调改曝光），软肩是"保 LDR 只救 HDR 溢出"。
- **验证**：溢出 **40.1%→0%**、最高 1.307→0.999；近白 [0.9,1] 占比 18.4%→58.5%（平白救成有梯度近白）；
  gain=0 软肩开/关 **byte 级不变**；曲线单调/knee 下恒等/上确界=1.0/可微；GT 路径（gain=4/200）均 ≤1 有限。
  对比图 `outputs/samples/overlap_tonemap_fix.png`、夜景面板 `demo_lights_grid.png`（重叠盘保留边界）。
- **正交问题（非本次引入、已记 D41）**：`x**(1/2.2)` 在 x=0 导数→∞ 致对系数回传 NaN（4 种 mode/tonemap
  组合下一致，与本次无关）；指纹标定本就用掩膜损失+wave 模式（D40），留作后续 eps-clamp 加固。


**[2026-06-21] 全项目架构/代码/真实性审查 + 两处修复**（决策 **D36**）
- **新像差未进 ctrl_vec → 训练失配**（修复）：D33/D34 加的 W060/变迹/球色差进了 `sample_random`
  却没进 ctrl_vec（13 维 FiLM）；球色差是逐通道、连单色描述子图也抓不到 → 训练全盲加噪。
  `sample_random(extended=False)` 默认 gate 出训练（验证：训练样本三者恒 0，既有 W040/loca 正常）。
- **深度 min-max→百分位**（修复）：离群点独占端点压窄主体视差区间 → CoC 尺度漂移；改 [0.5,99.5]
  百分位（验证：含极端离群点时主体范围 [0.04,0.96]，旧法会压到 ~0）。
- **一致性核验（无问题）**：snap 训练/推理同口径、GT(精确 alpha)/输入(tent)渲染口径一致、
  matte_gt 从真视差算、能量守恒 0.5→0.5。**真实性评估**：物理保真成立；限制（被遮背景不
  inpaint/HDR 近似/相对视差/中等离焦封顶/>5 阶球差）已知、细化网兜底或属正交 future work。

**[2026-06-21] 渲染器深度优化 + 7 篇文献对照**（决策 **D31–D35**）
- **D31 自适应分层修弥散圆中心亮点**：诊断为 tent 把点源拆到两 CoC 差异过大的层（K=84/8 层
  步长 12px）→ 大盘叠小亮盘。`effective_n_layers` 按 K 自适应保步长 ≤3px（中心/盘比 1.91→1.0）。
- **D33 软边缘溯源（非 bug）**：逐项隔离证明边缘宽由【离焦量】定（小盘近焦过渡区本就软，物理
  正确），非 softness/光谱。demo K=40→75 进几何区。**新增变迹(STF)+球色差**（边缘/内盘 0.97→0.28；
  盘边 R≠B 有色镶边）。
- **D34 高阶球差 W060**（ρ⁶，盘内带状/真 nisen）+ **WR 逐区平顶**（按 bloom 估各高光相对亮度，
  亮源 peak 10 vs 暗源 4；合成锐利点无 bloom→守卫回落全局，训练分布不变）。
- **D35 焦带吸附修锐前景遮挡漏光**：锐前景中心渗背景 0.35→0.00；散景盘重叠仍半透明叠加；GT 不变。
- **文献对照（Wu2010/2013/MPIB/Dr.Bokeh/BokehDiff/Bokehlicious/BokehMe++）**：7 篇 0 篇建模像差，
  我们=可微+轻量+解耦可控的唯一；K 是相对模糊(同 BokehMe，非物理 f-stop)，单图无法也无需。

**[2026-06-20] Plan B 验收：train_run6（D30 修人脸虚化）+ P2a 启动**（决策 **D30/D32**）
- **train_run5 验收发现 bug**：matte_gt 旧定义（局部 d_fg/d_bg 归一化位置）在有更近前景时把合焦
  主体判成中间层 α≈0.2 → **e2e 人脸被虚化**。**D30 改带符号 CoC 锚点**：r≥0(合焦/前景)→α=1、
  背景线性衰减（女生 r=0→α=1.000）。train_run6（50k，naf 无、film_res，matte+AMP+batch6）验收：
  **人脸清晰、无找补、无中心亮点、边界 L1 phys=0.0638→refined=0.0604（净 −0.0033，优于 run5 的 +0.0087）**。
- **P2a NAFNet 块**（D32）：实现 `NAFFiLMBlock`(LayerNorm2d+SimpleGate+SCA+FiLM) 可切换；同 ch 约半参
  → 加宽 144/72 同预算 713k；全分辨率激活显存大 → **梯度检查点**(9.7GB,batch6 放进 12GB)。
  自动接力（train_run6 完→自动起 train_run7）。**train_run7 = naf@同预算**，~28h，验收=边界 L1 ≤ 0.0604。
- **AMP 混合精度**：film_res +9% 速度、省 0.9GB 显存，loss 一致。

**[2026-06-18] train_run4 验收 + Plan B v0（matte 选择式重渲）全链路实现 + 50k 重训**（决策 **D29**）
- **train_run4 验收**：50k 收尾（边界 L1 phys=0.0532→fused=0.0424，净 −20%），但 21.jpg
  找补**减轻未根除**（mask 0.26→0.18，仍 ~2× 训练值；zoom 直比仍重新锐化背景树枝、`ideal`
  预设同样出现 → 纯网络找补）。坐实 D24 → 启用 Plan B。
- **Plan B v0 落地**（matte 选择式重渲）：`B=band·(α·B_fg+(1−α)·B_bg)+(1−band)·B_phys`，
  网络只出 1ch matte α，B_fg/B_bg 均渲染器产物 → 找补结构上不可能。各件 smoke 通过：
  ① `renderer.push_band_to_background`（带内视差压局部背景，B_bg 带内 |Δ|=0.144 带外 0.006）
  + `foreground_occupancy_gt`（α_gt 真值）；② `RefineNet(matte_mode=True)` **706,917 参数**
  （< v1 同条件 719k），matte∈[0,1]、带外 splice 恰为 B_phys（|Δ|=0.0000）；③ `synth.matte_gt`
  （带内均值=目标 ᾱ 0.34）；④ `train.refine_forward` matte 分支；⑤ `e2e_demo.refine_full_image_matte`。
- **关键踩坑（BCE）**：matte 直监用 L1 时 ᾱ 被 recon 早期压到 0 后**永久卡死**（L1 logit 梯度
  ∝σ' 在饱和处消失，train_probe_matte 实证 400 步 ᾱ=0.000 不恢复，refined 反劣 +0.0289）。
  改 **BCE-on-logits**（logit 梯度=α−tgt 不消失）→ ᾱ 0.10→0.32 收敛到真值 0.34。
- **e2e 探针（400 步欠训）**：matte 路径跑通（matte 均值 0.337）；21.jpg 背景树枝找补**显著
  轻于 run4**（`_planB_twig_{ideal,soap_bubble}.png`：refined 仅余柔结构，非 run4 的锐利纹理）。
- **认知**：synth 边界 L1 **不是** Plan B 合适判据（合成验证集由前景边缘误差主导，v0 不修前景
  bleed）；Plan B 主判据 = e2e 21.jpg 找补是否消失（D29 ②）。**50k 重训**：`outputs/train_run5/`
  （COCO 118k，matte_mode，1.2s/it ≈17h，2026-06-18 启动）。

**[2026-06-18] 渲染器真实性复核（第二轮，肉眼反馈驱动）**（决策见 DECISIONS D28）
- **WR 漏检彩色高光**（修复）：旧 WR 按亮度均值判阈值 → 饱和红 (lum~0.34) 漏检、彩色
  弥散圆偏暗 + 半成形怪异亮边。改用【通道最大值】判阈值 + `chroma=lin/detect` 上色，
  红/蓝光斑与白光斑一视同仁地变亮、锐化。
- **理想盘的相干衍射亮边环**（修复）：`|FFT{硬边光瞳}|²` 在盘边产生伪亮环（理想盘边缘
  1.25× 内部），被误读为"像差没解耦"。实为相干仿真伪影（PSF 层面 cream/soap/ideal 本就
  分离：亮心 0.59 / 亮环 2.4× / 平）。对光瞳边缘 apodization（softness 0.02→0.04，
  新增 `pupil.DEFAULT_SOFTNESS`）压到 1.10×，球差亮环/猫眼/多边形不受影响。重标定
  斜率 3.93→3.87。光谱平均抹盘内环、apodization 抹盘边环，分工互补。
- **复验**：6 预设×{global,tile}×WR×apodization 零告警、有限非负；端到端可微；decoupling
  对角元素 +1.00/+1.00/−0.90/−1.00（保持）。视觉 `outputs/render_test/renderer_fixes.png`、
  `demo_lights_grid.png`（各风格清晰可分、彩色光斑明亮）。

**[2026-06-18] 物理渲染器复核：两处修复 + 全链路可靠性复验**（决策见 DECISIONS D26/D27）
- **D26 PSF 裁剪/halo 漏算非离焦像差**（修复）：`_layer_psf` 旧版裁剪半径只用 `4|W020|`，
  近焦面强球差（soap/cream，W040±1.5~2）PSF 外缘被裁 **最高 27% 能量**，截掉的正是
  肥皂泡亮环、且重归一化把截断掩盖。新增 `psf_extent_px`（计入 8|W040|+视场项），
  漏裁 **27% → ≤0.1%**（全预设含彗差/像散高像高）。视觉 `outputs/render_test/soap_crop_fix.png`。
- **D27 高光权重重分配 WR**（新增，接 BokehMe++ Fig.3）：诊断旧 `srgb_to_linear` 只做
  乘性提亮（中心剖面 0.37→9→0.37 尖峰=高斯波，**未解决** WR 问题）。新增
  `redistribute_highlights`：散射前把高光掰成平顶方波（WR 后剖面 0.45→9→0.45 平顶+2px 锐边）。
  Fig.3 复现 `outputs/render_test/wr_fig3.png`（红/蓝高光 w/ WR 边缘更锐、内部更匀、叠加分层更清）。
- **全链路可靠性复验**：6 预设 × {global, tile} × WR 开启，在"采样/裁剪 guard 提为 error"
  下**零告警、输出有限非负**；对 image/`highlight_gain`/`W040`/`K` 全可微（梯度有限非零，
  镜头指纹标定链路不破）。物理公式与符号约定均未改，向后兼容。

**[2026-06-09] PSF 生成器物理验证** — `python -m optics.visualize`（CPU，借用 ultralytics 的 torch）
- 产物：`outputs/psf_test/presets.png`、`outputs/psf_test/w040_sweep.png`（人工读图核对）。
- 结论：CLAUDE.md 第 3 节**第一档 7 个效果全部涌现**——奶油柔散、肥皂泡亮边、六边形散景、
  洋葱圈同心环、猫眼、单侧亮边、球差符号翻转的形态反转。六边形与洋葱圈尤为干净。
- 偏弱项：猫眼切口不够戏剧化（`vignette_strength` 偏小，可调大）。

**[2026-06-09] 可微性验证** — PSF 对像差系数 `W040` 反传，梯度 ≈ 0.0084，**非零且有限**；
PSF 能量归一 sum=1.0。→ 端到端训练与镜头指纹标定的可微链路成立（CLAUDE.md 第 12 节硬要求）。

**[2026-06-10] 项目自审 + PSF 生成器加固**（决策见 DECISIONS D8–D11）
- **离焦标定**（`python -m optics.calibrate`）：实测 `r_px = 3.93·W020 + 0.18`，
  与几何光学理论 `r=4·W020` 最大相对误差 8.1%（小离焦端的衍射偏置，截距已吸收）；
  曲线单调、无混叠告警。映射落盘 `outputs/psf_test/defocus_calibration.json`，渲染器直接读用。
- **解耦矩阵**（`python -m optics.decoupling`）：5 旋钮×5 签名（球差→环比、彗差→偏度、
  像散→log(σx/σy)、猫眼→透过率、洋葱圈→涟漪），对角元素 |·| = 1.00/1.00/0.78/1.00/1.00，
  **对角占优成立**。已知真实物理串扰：猫眼↔像散共享拉伸签名但**符号相反**（−1.00 vs +0.78）；
  猫眼对偏度有 +0.31 串扰（猫眼+离焦本就移动散景球，论文如实报告）。
  产物 `outputs/psf_test/decoupling_matrix.png` —— RQ1 的第一份可量化证据。
- **物理修正回归验证**：① LaCA 改径向平移后，H=1 处 R/G/B 质心 x = 72.1/70.1/68.1
  （±2px 方向正确，H=0 自动消失）；② T(H) 随像高单调下降 1.0→0.99→0.85；
  ③ 混叠 guard 在 W020=60(>N/8=32) 时正确告警；④ 洋葱圈相位版可微（梯度非零有限）；
  ⑤ 重跑 `optics.visualize`：7 个第一档效果**全部保持涌现，无回归**。

**[2026-06-10] GPU 恢复 + M2 渲染器 first light**（`python -m render.demo`，RTX 5070 12GB）
- 环境：重启后驱动三层一致（580.159.03），torch cuda 正常；optics/depth 均在 cuda 跑通。
- **首张散景图**：BokehMe demo 图 (1024×1280) → DA V2 视差（方向正确，人工核对）→
  6 个镜头预设渲染（`outputs/render_test/demo_*.png`，总图 `demo_grid.png`）。
  人工过目结论：✅ 对焦主体锐利（ideal 中眼镜清晰）；✅ 遮挡正确（前景树枝以模糊
  边缘压在主体上、无渗色硬边）；✅ 背景灯光出亮斑散景球；✅ soap_bubble 的主体柔化
  经查是**球差柔化合焦面的真实物理**（Trioplan 式柔光，对照 W040=0 的预设主体清晰）。
- **速度**：全局路径 0.06s/张@1024×1280×24层（cuda）；tile 视场相关路径 2.0s/张。
- **视场相关验证**（点光源阵列 `psf_grid_catseye.png` + 放大图 `diag_grid_zoom.png`）：
  角落 H=0.81 亮斑沿径向压扁、中心正圆 → **tile 旋转方向正确，猫眼/旋焦生效**。
- **可微性**：散景输出对 d_f / K / W040 反传梯度均非零有限（global 与 tile 两路径）；
  修复了 `coc_to_w020` 中 math.copysign/Python max 断梯度链的问题（torch 分支）。
- **已知限制（待改进）**：① tile 接缝——亮斑横跨 tile 边界时有可见不连续，
  改进方向：tile 重叠 + 羽化混合（如 50% overlap + Hann 窗）；② 边界细化网未接入，
  深度边缘的渗色/硬边留给细化网（M2 后半）。

**[2026-06-10] 镜头指纹自标定验证（RQ3 首份证据）** — `python -m render.fingerprint_demo`
- 实验：已知真值 (d_f=0.24, K=30, W040=+1.5) 渲染目标图，从错误初值
  (0.55, 12, −0.8) 梯度下降反求。**两阶段延拓协议**（先低阶 (d_f,K)、后联合放开
  W040 从 0 延拓——单阶段联合优化会掉局部极小，踩坑记录见 DECISIONS D12）。
- 结果：恢复 (0.241, 29.9, +1.495)，**相对误差 0.5% / 0.4% / 0.3%**；
  420 iters 共 11.2s（27ms/iter，384px@RTX 5070）；L1 损失 0.06 → 3e-4。
- 产物：`outputs/fingerprint_test/{convergence,comparison}.png`、`result.json`。
  收敛曲线三参数全部落到真值线 → **"从样张反求像差系数"的可微链路成立**。

**[2026-06-10] 在线合成流水线跑通**（`python -m data.synth`，**bokeh 专用 env 首跑=环境验收✅**）
- 实现：程序化前景（傅里叶轮廓 blob+噪声纹理+软 alpha）+ 真实照片背景裁块 +
  HDR 高光贴片 + `sample_random` 独立采样像差 → GT 走 `composite_blurred_layers`
  精确分层合成；输入侧视差**三连扰动**（边界形变/高斯模糊/低频噪声，模拟 DA V2 误差）。
- 视场相关监督技巧：每样本采样 (H_field, azimuth) 作为"裁块在画幅上的位置"，
  单 PSF 渲染整块（与 tile 近似 D8 物理一致），猫眼/彗差/像散/LaCA 由此获得廉价监督；
  (H, azimuth) 一并写入 14 维控制向量 `ctrl_vec`（顺序文档化在 make_sample 内）。
- 人工核对 `outputs/synth_test/samples.png`：扰动视差边界可见地不贴/发糊✓、
  对焦层锐利✓、前/背景虚化方向正确✓、暗背景高光成亮斑散景球✓。
- 速度：**139 ms/样本**（512²@RTX 5070）——在线训练吞吐足够。
- 背景池当前为**占位**（DA V2 示例 + BokehMe 输入图共 ~25 张）；正式训练需换成
  大相册目录（改 `SynthConfig.bg_dirs` 即可）。

**[2026-06-10] 用户审图 → 四问题修复与复核**（根因分析见 DECISIONS **D13**）
- ① 深度边界软坡 → 新增 `snap_disparity_edges` 视差边缘吸附，主体轮廓干净
  （`demo_disparity_snapped.png` 对照原图）。
- ② **猫眼方向 90° 真 bug**（rotate_psf 旋转符号反；旧验证用顶边亮斑是退化情形
  ——轴向 mod π 在正方向上 ±a 不可分辨）。修复后点阵四角猫眼切向排列=正确旋焦
  （`psf_grid_catseye.png` 重新生成并复核）。
- ③ demo 背景 CoC 太小（~10px）看不清盘结构 → K 按"背景 CoC≈35px"反推 +
  新增夜景灯点展示 `demo_lights_grid.png`：六边形/洋葱圈/肥皂泡亮边/猫眼全部清晰可辨。
- ④ soap/cream "看反"实为验证离焦符号用错：亮边环在 sign(W040)≠sign(W020) 侧，
  预设命名按背景侧(W020<0)定义，预设值本身正确。visualize 改为双侧展示、
  decoupling 测试离焦改为 −12（ring_ratio 符号转正：对角 +1.00/+1.00/−0.83/−1.00/+1.00）。
  **2026-06-09 验证记录中 soap/cream 两项实为反相误判，以本条为准。**

**[2026-06-10] 虚化过量/主体被虚 → 对焦容差 + 球差焦移补偿**（DECISIONS **D14**）
- 诊断：① DA V2 相对视差把主体拉伸出 ~0.2 跨度（归一化固有现象），K 大时主体
  内部被虚；② 球差预设在 W020=0 焦面本就有满幅光晕（缺"对焦到最佳焦面"环节）。
- 修复：`RenderControl` 新增 **focus_tolerance δ0**（demo 实测主体跨度 ±0.104，
  带内 CoC=0）与 **balance_spherical_focus**（全层叠加 −W040 平衡离焦，默认开）；
  demo 虚化目标调温和（背景 CoC≈13px / 上限 60px）。
- 复核：全部预设主体锐利（含 soap/cream，残留轻微柔光=球差实拍观感）、
  背景风格保留；指纹自标定回归通过（0.8%/0.0%/0.2%）。两控制均可微。

**[2026-06-10] 范围调整：洋葱圈移除 + 光谱平均修复盘面能量不均**（DECISIONS **D15/D16**）
- **洋葱圈全链路移除**（用户决定）：像差字段/采样/预设/解耦签名/可视化/demo 全部清理，
  `ctrl_vec` 14→**13 维**；解耦矩阵 4×4 重跑对角 +1.00/+1.00/−0.83/−1.00 依旧占优；
  Sivokon & Thorpe 参考退役。§6 中早于本条的"7 个效果/洋葱圈"记录以本条为准。
- **光谱平均**（盘内能量不均的根因=单色相干菲涅耳环；真实相机宽光谱积分后盘面均匀）：
  `rgb_psf` 默认 ±6% 带宽 3 子波长加权平均，A/B 对照 `outputs/psf_test/spectral_ab.png`
  ——盘面变均匀、边缘特征（亮边/六边形角点/猫眼）保留。
- 全链路回归：标定斜率不变(3.93)、指纹 1.1%/0.0%/0.1%、synth 159ms/样本、
  demo + 夜景灯点面板重新生成并人工复核（盘面均匀✓）。

**[2026-06-10] 全库代码审查**（DECISIONS **D17**）
- 修正性 bug：**alpha 卷积误用 R 通道**（三处合成代码）→ 改为均值核，LoCA 下边界
  不再彩色失配，且 alpha 卷积省 2/3（synth 实测 **159→106 ms/样本**）。
- 质量修复：tile 逐块乘 T(H) 的亮度阶梯 → **逐像素平滑 T 图**（H 桶积分+线性插值）。
- 同步：configs/base.yaml 对齐现状（缓存定位/光谱平均/cond_dim=13/渲染参数）；
  refine/eval stub 注明接口约定；全库过时注释清理 + 原理性注释补强
  （FFT 零填充、偶数核对齐、单色签名、软阶跃等）。
- 回归：标定/解耦/指纹(0.4%/0.1%/0.1%)/梯度/demo/灯点面板全部通过。

**[2026-06-11] M2 收官启动：COCO 背景池 + 50k 正式训练 + 端到端 demo 实现**（DECISIONS **D20**）
- 训练基建：`train.train` 新增 `--bg-dirs`（背景池目录）与 `--resume`（断点续训）；
  `BackgroundPool` 加坏图重抽防护（118k 张整夜跑不因单张读失败崩）。
  COCO smoke（30 步）通过：118,287 张背景、0.44s/it、5.7GB，损失行为正常。
- **50k 步正式训练启动**（`outputs/train_run2/`，日志 train.log）。
- **端到端 demo**（`python -m render.e2e_demo`）：真实照片→DA V2→tile 渲染→
  512² 滑窗细化（窗心 (H,azimuth) 拼 ctrl_vec + Hann² 融合，D20）→与纯渲染对照
  面板（含误差图最活跃处放大窗）。机制验证：512² 合成样本上滑窗边界 L1
  0.0353→0.0314，与直接前向(0.0312)一致=收益不因滑窗丢失。
- 已知现象：10k 旧权重（25 张占位背景）在真实照片上把虚化前景细枝"找补"回
  锐利纹理（过拟合占位背景池）——正是换 COCO 的动机，待 50k 权重复核。

**[2026-06-15] 默认深度骨干 Small → Base（vitb, 97M）**（DECISIONS **D25**）
- **动因**：真实图（21.jpg）Small 深度质量不行——全局关系错（背景树丛判成近）+ 主体内部斑驳。
- **认知修正**：DA V2 是**冻结仅推理骨干、不算可训练参数** → 换大它**不碰 RQ2 轻量性**
  （之前把"换冻结 DA"和"扩可训练容量"混为一谈是误判）。
- **证据**（depth_small_vs_base.png）：Base 在全局深度关系 + 主体分离 + 主体内部均匀性
  明显更好；细枝糊未质变（单目固有难题，符合预期）。未上 Large（边际收益递减）。
- **影响面**：只动推理端，训练端不跑 DA（用合成视差）不受影响。许可 vitb=CC-BY-NC-4.0
  （学术非商用可用，论文须声明）。**配套待办**：扰动按 Small 误差标定，run4 后 e2e
  核图观察、必要时按 Base 重标定 `perturb_disparity`。可逆：encoder='vits' 切回。

**[2026-06-15] 路线压力测试：红线复核 + Plan B 确立**（DECISIONS **D24**）
- **质疑**："能否用全新网络 / 开源全焦-虚化数据集替代当前细化网？"——对核心红线
  （physics→network）的正面压力测试。
- **结论**：① **否决端到端生成网络**（用开源虚化数据直接训）= 风格来自网络权重 →
  RQ1/RQ3/差异化三支柱全塌，退回 BokehMe 老路；② 开源（全焦,虚化）数据**不能做
  边界修复 GT**——边界过渡的正确形态取决于盘形状，同一边界几何在不同像差系数下
  过渡完全不同，GT 只能自有渲染器合成；③ 开源数据角色不变=真实微调/评测/指纹（§8）；
  ④ **坚持原路线**，找补是分布问题非路线死症。
- **Plan B 确立**：若 run4 仍残留找补，启用**几何修正型网络**（网络只预测 matte/深度
  修正 → 物理重渲，网络不接触颜色 → 结构性根治找补，NETWORK_DESIGN §6.3）。

**[2026-06-15] train_run3 验收 → 找补根因坐实 + train_run4 重训**（DECISIONS **D23**）
- **train_run3 结果**：训练健康——更难分布（phys 边界 L1 0.0521→0.0667）上净收益
  反增（−0.0157→−0.0216，相对降幅 30%→32%），loss 平稳无塌缩。但 e2e 真实图
  （21.jpg 密集树枝背景）：**找补减轻但未根除**——zoom 对比 P1 锐度明显轻于 v1
  但树枝仍被部分锐化；mask 均值 0.25（v1 0.26）。
- **诊断**（band_diag.png 实证）：① 门控对密集背景**根本性失效**——边界处处跳变，
  扩张最大 CoC 后 band 均值 0.806≈全图、主体反被挖空，物理上分不开"主体遮挡边界"
  与"背景树枝间边界"；② 根因在数据——细薄结构此前只作近前景、对焦时 GT 清晰 →
  网络学成"细薄→锐化"，真实背景树枝恰该虚化，泛化方向相反。
- **对策（两路并举，用户拍板）**：① **背景细薄层**（data/synth.py p_thin_bg=0.4，
  背景侧插细薄层 + 强制对焦前景 → 重度虚化 GT，纠正泛化偏差，bg_thin_check.png 验证✓）；
  ② **焦邻近门**（boundary_band focus_gate，远背景碎边缘压制；坦诚对 21.jpg 仅
  0.806→0.755，门控救不了场，保留作边界稀疏图的兜底）。
- **train_run4**：`outputs/train_run4/`（50k，COCO 118k，同 P1 架构 + 数据修正，
  2026-06-15 15:20 启动）。验收主靶仍是 21.jpg 找补是否根除。

**[2026-06-12] 细化网 v2 (P1) 全链路实施 + 50k 重训启动**（DECISIONS **D22**）
- **实施内容**（一次重训打包）：① P1a 空间物理条件图（H/sin az/cos az 逐像素，
  虚拟画幅精确极坐标几何）；② P1b PSF 描述子图（r_eq/elong/各向异性取向/环比/T，
  单色 PSF 表 + 离焦 tent×像高 tent 双轴插值，与 D11 签名同词汇表）；③ 细薄前景
  （树枝/草丝/栅栏，p=0.5）；④ mask 边界带门控（m·band，带外神经分支结构性失效）；
  ⑤ H 子带 GT（跨度 <0.12/<0.30 → 1/2/3 带，输入侧同组 centers/weights）。
- **验证**：conditioning 自检（r_eq 随离焦单调✓ 猫眼 elong 随 H 增✓ T(H) 降✓
  条件图/边界带可视化人工核对✓）；network 自检（**719,027 参数** <2M✓，梯度链✓，
  透传 7e-4✓）；synth 可视化（细薄前景出现且 GT 正确虚化✓，144ms/样本）；
  train smoke（0.86s/it，5.8GB，门控下 m̄ 0.33→0.17）；e2e v2 整图单前向
  （0.13s/张@1024×1280，smoke 权重机制验证✓）。
- **50k 重训**：`outputs/train_run3/`（COCO 118k 背景，setsid 脱离会话，
  2026-06-12 15:31 启动，0.86s/it ≈12h）。验收门槛（NETWORK_DESIGN §9）：
  ①合成边界 L1 ≤ 0.0364×0.9≈0.033；②整图前向四角一致性；③真实照片找补消失/明显减轻。

**[2026-06-12] 50k COCO 训练完成 + e2e 核图：找补现象归因修正（重要）**
- **50k 训练正常收尾**（`outputs/train_run2/`，无中断）：最终边界 L1
  phys=0.0521 → fused=0.0364（**净降 30%**，优于 10k 的 −27% 且验证集更难）；
  0.42s/it、峰值 5.7GB；训练时 m̄≈0.06，mask 集中边界（红线守住）。
- **e2e 核图**（`python -m render.e2e_demo`，50k 权重，3 预设）：
  `outputs/e2e_test/e2e_grid.png` + 放大件 `inspect_{mask,zoom_phys,zoom_refined}.png`。
  ✅ 主体轮廓处 mask 成干净亮边；✅ 细化 0.09–0.11s/张。
  ❌ **找补现象未消失**：zoom 窗内被正确虚化的前景树枝/叶，细化后重新锐化
  （网络从全焦输入抄回纹理）；❌ mask 均值 0.26（训练 0.06 的 4 倍），
  大面积点亮在前景枝叶与背景纹理区，不限于深度边界。
- **归因修正**：换 COCO（118k 背景）没有解决 → "占位背景池过拟合"不是根因。
  新归因：① 合成前景全是光滑 blob，**没有细薄多边界结构**（枝/叶/发丝），
  真实细枝对网络而言"处处是边界"，按边界策略抄锐利输入；② mask 监督只教了
  "哪里该亮"，没约束真实纹理上的误触发。
- **对策（并入 P1 重训，不单独烧训练）**：a) synth 前景加细薄结构类
  （程序化枝条/栅栏/发丝状 alpha）；b) mask 边界带门控（用视差边缘带物理限制 m
  的活动范围，physics→network，红线内）；c) P1b 描述子图让网络知道"此处核半径
  30px"，从条件上抑制锐化倾向。验收沿用 NETWORK_DESIGN §9 P1 门槛③。

**[2026-06-11] 边界细化网 + 训练循环落地，M2 完整管线跑通**（DECISIONS **D18/D19**）
- **细化网**（`python -m refine.network`）：残差式轻量 ARNet(96ch×3 块@1/4 分辨率)
  +IUNet(48ch×2 块@全分辨率)+FiLM 注入 13 维 ctrl_vec，误差图融合
  B = m·B_neural + (1−m)·B_phys。**684,035 参数**（2M 上限的 34%）。
  冒烟测试：前向形状✓、梯度对网络参数与 ctrl_vec 均非零✓（踩坑：输出头严格
  零初始化会阻断 FiLM 梯度，改 std=1e-4 近零初始化，D18③）、
  透传起点 |B−B_phys|=4e-4✓。
- **训练输入侧物理渲染**：新增 `render_field_patch`（renderer.py）——与 GT 的
  composite_blurred_layers 共享 (H_field, azimuth) 单 PSF 视场近似 + T(H)，
  从扰动视差出发（与真实推理同路径：扰动 → snap_edges → 渲染）。
- **误差图塌缩与修复（D19，重要踩坑）**：只用融合损失训练 1500 步，mask 全图→0、
  网络退化成纯透传。加三项：mask 显式监督（m_gt=clamp(10·|B_phys−GT|)，λ=0.2）、
  神经分支直监（不经门控，λ=0.5）、边界加权重建（1+4·m_gt）。修复后 mask 精确
  点亮在物体边界。
- **10k 步验证训练**（`outputs/train_run1/`，RTX 5070，~75 分钟）：
  **边界区域 L1（m_gt>0.2 处）0.0516 → 0.0377，较物理基线净降 27%**，
  1k 步零收益 → 7k 步后趋平；可视化人工核对：边界误差环明显变淡、
  过曝边界红色饱和区收窄（viz_it010000.png）。速度 0.44s/it@512²/batch4，
  峰值显存 5.7GB（12GB 内余量充足）。LPIPS-VGG 感知损失正常工作。
- 评测口径新增：**细化网净收益一律看边界 L1**（全图 L1 被 >90% 平坦区主导，
  对该任务不灵敏）——`evaluate_boundary`（train/train.py）。

---

## 7. 待办 / 开放问题

**近期（M1 收尾）**
- [ ] 调大 `vignette_strength`，重扫猫眼/旋焦，确认戏剧化效果。
- [ ] 补彗差 W131 / 像散 W222（第二档）的可视化核对。
- [x] ~~**离焦标定**~~：已完成（`optics/calibrate.py`，r≈3.93·W020，见 §6）。
- [ ] GPU 恢复后，optics + depth 各在 cuda 上跑一遍，确认无设备相关问题。
- [ ] 实现 `data/synth.py` 在线合成（前景+背景+随机像差→配对；**GT 用已知 alpha 精确
      分层合成、输入侧深度加扰动，见 CLAUDE.md 第 7 节更新**）。

**M2 进行中（渲染器+合成+细化网+训练已落地，见 DECISIONS D8/D9/D18/D19 与 §6 验证）**
- [x] ~~`render/renderer.py`~~：已实现并验证（分层 gather + tile + 现场 PSF + T(H) + HDR 高光）。
- [x] ~~`data/synth.py`~~：已实现并验证（精确 GT + 扰动视差 + 13 维 ctrl_vec）。
- [x] ~~`refine/network.py`~~：已实现并验证（684k 参数，10k 步边界 L1 −27%，D18/D19）。
- [x] ~~`train/train.py`~~：已实现并跑通（L1+LPIPS+mask 监督+边界加权，0.44s/it）。
- [x] ~~**正式训练 (v1)**~~：50k 完成（边界 L1 −30%）；e2e 核图发现找补未解 →
      归因修正 D22，v1 权重存档 `outputs/train_run2/` 作消融基线。
- [🔄] **P1 50k 重训**：`outputs/train_run3/` 运行中（≈12h，2026-06-12 15:31 起）。
      完成后：①对照 NETWORK_DESIGN §9 验收门槛；②e2e 整图前向重跑核图
      （找补应消失/明显减轻）= M2 真正收官。
- [ ] tile 接缝改进：重叠 + 羽化混合（50% overlap + Hann 窗），消除亮斑跨界不连续。
- [ ] 细化网容量上探（如 96→128ch）：仅当 50k 步训练的边界收益不满意时（2M 内余量 3 倍）。

**M2 后：细化网 v2 升级（路线详见 [NETWORK_DESIGN.md](NETWORK_DESIGN.md)，决策 D21/D22）**
- [x] ~~**P1a+P1b（一次重训）**~~：已实施（空间条件图+描述子图+细薄前景+边界带门控
      +H 子带 GT，见 §6 [2026-06-12] 与 D22），50k 重训运行中。
- [ ] P2a NAFNet 块替换（+P3a alpha matte 辅助头随车）——P1 验收后。
- [ ] P2b CoC 感知注意力 / P2c 容量 96→128——仅当边界 L1 不达标（触发条件见 NETWORK_DESIGN §9）。
- [ ] 数据集下载：RealBokeh 3MP（HuggingFace）、VABD（GitHub）、BETD（CodaLab 注册）。
- [ ] M3 末：BETD 指纹定量评测（Sony↔Canon 50mm 转换对 GT）= RQ3 主实验。

**待决（需要时再议）**
- [ ] PSF 字典分桶数（H_bins / defocus_bins）与显存的权衡——注意字典已降级为
      推理缓存（D9），此项优先级下降。
- [ ] 是否需要 DA3 作深度消融——仅当边界质量成瓶颈时（DECISIONS D2 已留接口）。
- [ ] 镜头指纹升格为头牌应用："镜头风格迁移" demo（普通镜头照片 → Helios 旋焦风格），
      M4 论文叙事的差异化王牌。

---

## 8. 下一步候选（择一推进）

1. **M2 收官（主线）**：50k 步正式训练（运行中）→ 真实照片端到端 demo 用正式权重核图。
2. **细化网 v2（P1）**：空间物理条件图 + PSF 描述子注入，一次重训（NETWORK_DESIGN §4，D21）。
3. **渲染器打磨**：tile 重叠羽化消接缝；调大猫眼戏剧化效果；补第二档（彗差/像散）可视化。
4. **M3 预热**：图像层面的可控性保真 / 解耦签名评测（复用 optics/decoupling.py 的签名体系）。
4. ~~镜头指纹原型~~：✅ 已完成（自标定误差 <0.5%，见 §6 与 DECISIONS D12）。
   后续扩展：扩参（vignette/色差）逐级延拓 + M4 真实镜头样张标定。
5. ~~细化网 + 训练~~：✅ 已完成（684k 参数，10k 步边界 L1 −27%，见 §6 与 D18/D19）。
