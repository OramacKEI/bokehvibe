# 项目进度看板 (PROJECT STATUS)

> **角色**：本项目的「现在做到哪、怎么跑、接下来做什么」的**活文档**，每次推进后更新。
> 最近更新：2026-06-12。
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

## 1. 一句话现状

**细化网 v2 (P1) 已实施，50k 重训进行中**：v1 的 50k COCO 基线完成（合成边界 L1
−30%）但 e2e 核图发现"找补"未解（归因修正 → DECISIONS **D22**）。P1a 空间条件图
+ P1b PSF 描述子图 + 三联找补对策（细薄前景/边界带门控/描述子条件）已全部落地并
smoke 通过；**50k 重训运行中**（`outputs/train_run3/`，0.86s/it ≈12h，
2026-06-12 15:31 启动）。e2e 已支持整图单前向（0.13s/张，D20 滑窗对 v2 权重退役）。

---

## 2. 里程碑进度（对照 CLAUDE.md 第 10 节）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **M0** | 环境 + 基线复现（DA V2 / BokehMe / 数据集） | 🟢 GPU 恢复；DA V2 + optics 已在 cuda 跑通；`bokeh` env 构建中 |
| **M1** | pupil→PSF 生成器 + 在线合成；验证肥皂泡/旋焦/单侧亮边 | 🟢 PSF 生成器✅并验证+标定+解耦；在线合成✅ |
| **M2** | 接入可微渲染 + 训练边界细化网，单卡跑通完整管线 | 🟢 渲染器✅、细化网✅、训练✅（10k 步边界 L1 −27%）；待更长训练+真实照片端到端 demo |
| **M3** | 可控性保真、解耦度、消融实验 | ⚪ 未开始 |
| **M4** | 真实集评估 + 镜头指纹 demo + 论文 | ⚪ 未开始 |

🟢 进行中且有产出　🟡 部分就位　⚪ 未开始

---

## 3. 模块进度

| 模块 | 文件 | 状态 | 说明 / 验证 |
|---|---|---|---|
| 像差系数 | [optics/aberrations.py](optics/aberrations.py) | ✅ 实现 | dataclass + 独立采样 `sample_random` + 6 个风格预设 `PRESETS`（洋葱圈已移除，D15） |
| 复光瞳 | [optics/pupil.py](optics/pupil.py) | ✅ 实现 | 波前 W、振幅遮罩（多边形/猫眼，sigmoid 软化保可微）、`complex_pupil`(含 phase_scale 光谱平均接口)、`relative_transmission`(猫眼边角失光 T(H)) |
| PSF | [optics/psf.py](optics/psf.py) | ✅ 实现 | `pupil_to_psf`(\|FFT\|²+归一+混叠/裁剪 guard)、`rgb_psf`(LoCA+LaCA 径向平移)、`build_psf_dictionary`(推理缓存,含 T(H))、`sample_psf` |
| PSF 验证 | [optics/visualize.py](optics/visualize.py) | ✅ 实现并跑通 | 产物见 `outputs/psf_test/`，§6 有验证记录 |
| 离焦标定 | [optics/calibrate.py](optics/calibrate.py) | ✅ 实现并跑通 | `r_px ≈ 4·W020` 实测斜率 3.93，JSON 落盘供渲染器换算（§6） |
| 解耦矩阵 | [optics/decoupling.py](optics/decoupling.py) | ✅ 实现并跑通 | PSF 层面效应签名矩阵，对角占优（§6，DECISIONS D11） |
| 深度 | [depth/estimator.py](depth/estimator.py) | ✅ 实现并跑通(cuda) | 后端无关封装，默认 DA V2-Small；DA3 留作消融备选（见 DECISIONS D2） |
| 渲染器 | [render/renderer.py](render/renderer.py) | ✅ 实现并出图 | 分层 gather + tile 视场相关（D8/D9）：`render_global`/`render_tiled`，PSF 现场算、标定换算、HDR 高光、T(H) 失光、全程可微 |
| 渲染 demo | [render/demo.py](render/demo.py) | ✅ 实现并跑通 | 真实照片→视差→6 预设出图，见 §6 |
| 指纹标定 | [render/fingerprint_demo.py](render/fingerprint_demo.py) | ✅ 实现并验证 | RQ3 自标定：两阶段延拓反求 (d_f,K,W040)，误差 <0.5%（§6，DECISIONS D12） |
| 细化网 | [refine/network.py](refine/network.py) | ✅ v2 (P1) | ARNet+IUNet+FiLM(13 维)+**17ch guide**（8 基础+9 物理条件图）+mask 边界带门控；**719k 参数**；开关全关=复现 v1（684k）；`python -m refine.network` 自检 |
| 条件构建 | [refine/conditioning.py](refine/conditioning.py) | ✅ 新增 (P1) | PSF 描述子表(单色, D11 同口径)→逐像素 9ch 条件图（H/方位角/r_eq/elong/取向/环比/T）+ 视差边缘带门控；`python -m refine.conditioning` 自检 |
| 在线合成 | [data/synth.py](data/synth.py) | ✅ 实现并跑通 | 程序化前景+真实背景裁块+随机像差→(全焦图,扰动视差,散景GT,c) 四元组；GT 走 `composite_blurred_layers` 精确分层；139ms/样本@cuda（§6） |
| 训练 | [train/train.py](train/train.py) | ✅ 实现并跑通 | 在线合成→物理渲染(no_grad)→细化→L1+LPIPS+mask 监督+边界加权（D19）；0.44s/it@512²/b4、峰值 5.7GB；10k 步边界 L1 −27%（§6）；已加 `--bg-dirs`/`--resume`，COCO 118k 背景池 50k 步正式训练进行中 |
| 端到端 demo | [render/e2e_demo.py](render/e2e_demo.py) | ✅ 实现，待正式权重 | 真实照片→DA V2→tile 渲染→512² 滑窗细化+窗心视场条件+Hann 融合（D20）；机制验证：滑窗收益与直接前向一致 |
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
