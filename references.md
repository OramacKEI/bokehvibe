# 参考文献清单 — 光学像差可控的轻量级散景渲染

> 按在本项目中的**角色**分组，每篇标注为何重要、是否有开源代码。
> 文档体系：[CLAUDE.md](CLAUDE.md)（总纲）、[PROJECT_STATUS.md](PROJECT_STATUS.md)（进度）、[DECISIONS.md](DECISIONS.md)（决策理由）、[NETWORK_DESIGN.md](NETWORK_DESIGN.md)（细化网设计路线）。
> 阅读顺序建议：BokehMe++（吃透架构）→ BokehMe 代码（动手复现）→ Dr.Bokeh（渲染器思路）→ Depth Anything V2（跑通推理）。光学理论三篇待 M1 设计 PSF 生成器时再深入。
> 第六/七组为 2026-06-11 网络升级文献调研所增（背景与结论见 DECISIONS D21、NETWORK_DESIGN.md）。
> **2026-06-21 深度精读 7 篇**（BokehMe++/BokehDiff/Bokehlicious/Dr.Bokeh/Wu2010/Wu2013/MPIB）→ 系统对照见 **DECISIONS D34**。
> 铁证：7 篇 SOTA **0 篇建模像差**；Wu2010/13 做像差但走全光追（不可微/需处方）。我们=可微+轻量+解耦可控的唯一。
> **D49（2026-06-22）：镜头指纹反演 (RQ3) 降级为 future（新阶段 P-FP）。第五组（盲反演/标定）、
> BETD 数据集、特色镜头样张均为 P-FP 阶段所需，主线不接入。**
> **第八组为 2026-06-22 渲染工程调研**（散景盘重叠 HDR 溢出修复）所增 → 见 **DECISIONS D41**。
> **第九组为 2026-06-22 真实镜头散景形态标定**（soap bubble/cream 剖面，区分球差柔化 vs 变迹虚化）→ 见 **DECISIONS D47/D48**。

---

## 一、技术骨干（必读，架构直接基于它们）

### BokehMe++ — *Harmonious Fusion of Classical and Neural Rendering for Versatile Bokeh Creation* (TPAMI 2025)
- **为何重要：** 本项目的技术骨干基础。细化网结构（ARNet + IUNet）、误差图融合、猫眼公式（K0/K, z_l）均源自此。支持可调模糊量 / 焦平面 / 高光模式 / 猫眼。
- 论文全文（OpenReview，免费）：https://openreview.net/forum?id=535chkWtSr
- IEEE Xplore：https://ieeexplore.ieee.org/document/10756626/
- 作者主页（含其它相关工作）：https://juewenpeng.github.io/
- 本地副本：`refs/bokehme2.pdf`

### BokehMe — *When Neural Rendering Meets Classical Rendering* (CVPR 2022 Oral)
- **为何重要：** BokehMe++ 的前身，**有官方开源代码 + BLB 数据集**。M0 复现的实际下手对象（++ 版未必开源，先从这个跑起）。
- 官方代码（PyTorch）：https://github.com/JuewenPeng/BokehMe
- **本地管线详解笔记**：[third_party/BokehMe/PIPELINE_CN.md](third_party/BokehMe/PIPELINE_CN.md)
  （逐行核对代码写成：散射 kernel / ARNet+IUNet / 误差图融合 / 与本项目的继承-替换对照表。
  ⚠️ 该目录被 gitignore，重克隆上游仓库会丢失此文件，必要时从备份恢复）

### Dr.Bokeh — *DiffeRentiable Occlusion-aware Bokeh Rendering* (CVPR 2024)
- **为何重要：** 本项目可微遮挡感知渲染器的借鉴对象。提出更精确的滤波式渲染方程 + 占用感知渲染，在渲染阶段直接解决边界颜色溢出与部分遮挡，无需后处理。
- arXiv：https://arxiv.org/abs/2308.08843
- 项目页（Purdue CGVLab）：https://www.cs.purdue.edu/cgvlab/www/publications/Sheng24CVPR/

---

## 二、深度骨干（冻结使用的模块）

### Depth Anything V2 (NeurIPS 2024)
- **为何重要：** 管线最上游，输出相对视差 D。本项目**默认 Base/vitb**（97M，仅推理，深度质量优于
  Small，D25）；`encoder='vits'` 可切回 Small。视差归一化用 [0.5,99.5] 百分位（离群稳健，D36）。
- **许可注意：** 仅 Small(vits) 是 Apache-2.0；Base/Large 为 CC-BY-NC-4.0（学术非商用，论文须声明）。
- arXiv：https://arxiv.org/abs/2406.09414 ｜ 代码：https://github.com/DepthAnything/Depth-Anything-V2
- 本地副本：`refs/depthany2.pdf`

---

## 三、对照基线（理解差异化定位，不必复现）

### Bokehlicious — *Photorealistic Bokeh Rendering with Controllable Apertures* (ICCV 2025)
- **为何重要：** 端到端、隐式 PSF、仅光圈控制。提供 **RealBokeh 数据集**（本项目真实验证集来源）。差异化对照：本项目有显式物理控制 + 焦平面控制 + 可解释性。
- 项目页：https://timseizinger.github.io/BokehliciousProjectPage/
- arXiv：https://arxiv.org/abs/2503.16067
- 本地副本：`refs/Bokehlicious.pdf`

### BokehDiff — *Neural Lens Blur with One-Step Diffusion* (ICCV 2025)
- **为何重要：** "轻量"差异化对标对象。官方训练脚本基于 RealVisXL_V5.0（SDXL 级骨干），印证"不与扩散拼算力"的判断。其视差图同样用 Depth Anything V2 估计，思路一致。
- arXiv：https://arxiv.org/abs/2507.18060
- 官方代码：https://github.com/FreeButUselessSoul/bokehdiff

### BokehFlow — *Depth-Free Controllable Bokeh Rendering via Flow Matching* (2025.11)
- **为何重要：** 生成派最新代表（撞车检查对象）。流匹配直接做全焦→散景的分布传输，
  无需深度；控制（焦点区域+模糊强度）经 Bokeh Control Adapter 以 cross-attention 注入。
  控制粒度仍停留在"虚多少/虚哪里"——**未触及像差系数级风格控制**，差异化成立。
- arXiv：https://arxiv.org/abs/2511.15066

### Generative Refocusing — *Flexible Defocus Control from a Single Image* (2025.12)
- **为何重要：** 撞车检查对象。两段式（DeblurNet 先恢复全焦 + BokehNet 再合成浅景深），
  用 EXIF 元数据捕捉真实光学特性、支持文本引导与自定义光圈形状。最接近"镜头特性条件"
  的生成派工作，但条件是 EXIF 黑盒嵌入，非物理系数，不可解耦不可反演。
- arXiv：https://arxiv.org/abs/2512.16923

### VABM — *Variable Aperture Bokeh Rendering via Customized Focal Plane Guidance* (2024)
- **为何重要：** 轻量对标（4.4M 参数）+ 提供 **VABD 数据集**（见第七组）。
  焦平面引导图思路与本项目 CoC 提示通道同源；只控光圈，无像差维度。
- arXiv：https://arxiv.org/abs/2410.14400
- 官方代码/数据：https://github.com/MoTong-AI-studio/VABM

---

## 四、光学理论（pupil→PSF 生成器的物理依据）

> 偏老 / 偏期刊，网上未必有免费 PDF，建议通过早稻田图书馆数据库（IEEE / SPIE / Springer）检索。

### Sivokon & Thorpe — *Theory of Bokeh Image Structure in Camera Lenses with an Aspheric Surface* (Optical Engineering, 2014)
- **状态：⚠️ 已随范围调整退役**（洋葱圈于 2026-06-10 移出研究范围，见 DECISIONS D15）。
  原为洋葱圈周期扰动项 `[1 + ε·cos(2πρ/T)]` 的理论来源（非球面加工瑕疵建模），保留备查。

### Wu et al. — *Realistic Rendering of Bokeh Effect Based on Optical Aberrations* (The Visual Computer, 2010)
- **为何重要：** 像差→散景的**先行工作**（最该划清界限的对象）。**已精读（D34）**：用真实多片镜组
  处方做全光线追踪（Snell 折射），§4 现象学与我们**完全一致**（球差亮心/亮环+前后反转、彗差彗尾、
  像散椭圆 Fig7）——是我们模型正确性的铁证。但**需镜头处方、非可微、极贵、要 3D 场景**。
  本项目增量 = 可微（训练+指纹反演）+ 轻量单图 + 解耦可控 + 参数化波前（无需处方）。
- Springer：https://link.springer.com/article/10.1007/s00371-010-0459-5 ｜ 本地：`refs/2010_vc_bokehrendering_jiazewu.pdf`

### Wu et al. — *Rendering Realistic Spectral Bokeh due to Lens Stops and Aberrations* (The Visual Computer, 2012/13)
- **为何重要：** 上篇续作，加**光谱/色差**（色散方程 n(λ)+真实玻璃库+SWC 波长簇）。**已精读（D34）**：
  色差现象学一致（盘边紫绿镶边、只影响高光）。我们用 LoCA/LaCA/球色差+3 子波长**廉价可微近似**同款。
- Springer：https://link.springer.com/article/10.1007/s00371-012-0673-4 ｜ 本地：`refs/2013_vc_bokehrendering_jiazewu.pdf`

### MPIB — *An MPI-based Bokeh Rendering Framework for Realistic Partial Occlusion Effects* (ECCV 2022)
- **为何重要：** 同作者组 MPI 散景框架。**已精读（D34）**：核心贡献=**背景 inpainting**（Sobel 梯度
  定遮挡掩膜→LaMa 补全被遮背景，Eq5-6），解决虚前景揭示背景时的部分遮挡。与我们 D35 焦带修法
  正交（我们修锐前景漏光，不补被遮内容）；P3b "MPIB-lite inpaint" 暂缓（低优先，细化网兜底）。
- 本地：`refs/MPIBokeh.pdf`

---

## 五、邻近领域：盲像差估计 / PSF 标定（RQ3 镜头指纹的 related work，future / P-FP 阶段）

> **D49（2026-06-22）：RQ3 镜头指纹反演已降级为 future work（P-FP 阶段），本组为该阶段所需，
> 主线不阻塞、不必现在接入。** 这些工作从图像**估计/标定像差或 PSF**，目标是"矫正像差"；
> 本项目的指纹反演方法相似、方向相反（"复刻风格"）。P-FP 阶段论文 related work 须覆盖。

### Eboli et al. — *Fast Two-step Blind Optical Aberration Correction* (ECCV 2022)
- **为何重要：** 从单张图像盲估计光学像差再矫正——与镜头指纹标定在"从照片反求像差"上同构。
- arXiv：https://arxiv.org/abs/2208.00950

### CircleFlow — *Flow-Guided Camera Blur Estimation using a Circle Grid Target* (2025)
- **为何重要：** 用标定靶估计相机模糊/PSF 的近作，指纹标定实验设计可参考其协议。
- arXiv：https://arxiv.org/pdf/2512.00796

### *Optical Aberration Correction in Postprocessing using Imaging Simulation*
- **为何重要：** 成像仿真驱动的像差矫正，"用物理仿真生成训练数据"思路与本项目在线合成同源。
- arXiv：https://arxiv.org/pdf/2305.05867

---

## 六、细化网条件机制与骨干（v2 升级路线的方法学依据，见 NETWORK_DESIGN.md）

### NTIRE 2023 — *Lens-to-Lens Bokeh Effect Transformation Challenge Report* (CVPRW 2023)
- **为何重要：** "多镜头条件化散景网络"的社区基准——条件方式（镜头标签+光圈数值嵌入）
  是本项目物理系数条件的**黑盒下位对照组**（NETWORK_DESIGN P1c 消融）；同时发布 BETD 数据集。
- 报告：https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Conde_Lens-to-Lens_Bokeh_Effect_Transformation._NTIRE_2023_Challenge_Report_CVPRW_2023_paper.pdf
- Starter 代码：https://github.com/Glass-Imaging/NTIRE23BokehTransformation

### BokehOrNot — *Transforming Bokeh Effect with Image Transformer and Lens Metadata Embedding* (CVPRW 2023)
- **为何重要：** 镜头元数据 one-hot + 光圈数值嵌入的代表实现（Restormer 骨干 + 双输入
  Transformer 块）。消融对照组的直接参考。
- arXiv：https://arxiv.org/abs/2306.04032

### NAFBET — *Bokeh Effect Transformation with Parameter Analysis Block based on NAFNet* (CVPRW 2023)
- **为何重要：** **NAFNet 骨干在散景任务上的有效性验证**（P2a 换块的直接依据）；
  Parameter Analysis Block = 参数条件注入的又一实现。
- 论文：https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Kong_NAFBET_Bokeh_Effect_Transformation_With_Parameter_Analysis_Block_Based_on_CVPRW_2023_paper.html

### NAFNet — *Simple Baselines for Image Restoration* (ECCV 2022)
- **为何重要：** P2a 换块对象：SimpleGate + 简化通道注意力（SCA）、无激活函数设计，
  恢复类任务参数效率标杆。FiLM 可挂在其 LayerNorm 后。
- arXiv：https://arxiv.org/abs/2204.04676
- 官方代码：https://github.com/megvii-research/NAFNet

### OmniLens++ — *Blind Lens Aberration Correction via Large LensLib Pre-Training and Latent PSF Representation* (2025.11)
- **为何重要：** "PSF 作为网络条件先验"的最近思想（像差**校正**域）：因盲设定只能用
  VQVAE 学 PSF 潜表征。本项目是**非盲**（PSF 自己渲染），可解析提取描述子——
  P1b 的对照叙事：related work 必引并说明非盲优势。
- arXiv：https://arxiv.org/abs/2511.17126
- 代码（待放出）：https://github.com/zju-jiangqi/OmniLens2

### FiLM — *Visual Reasoning with a General Conditioning Layer* (AAAI 2018)
- **为何重要：** v1 条件注入机制的出处（逐通道仿射调制 γ/β）。
- arXiv：https://arxiv.org/abs/1709.07871

### SFT / SPADE — 空间特征调制的两篇源头（CVPR 2018 / CVPR 2019）
- **为何重要：** P1a"空间量应该用空间条件而非全局向量"的方法学依据：
  SFT（*Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature
  Transform*）与 SPADE（*Semantic Image Synthesis with Spatially-Adaptive Normalization*）。
  本项目第一版走输入通道，消融期对比 SFT 式逐层调制。
- SFT arXiv：https://arxiv.org/abs/1804.02815 ；SPADE arXiv：https://arxiv.org/abs/1903.07291

---

## 七、数据集（核实过可得性，2026-06）

### RealBokeh（随 Bokehlicious，ICCV 2025）
- **内容：** 23k 张 24MP，Canon R6II + RF28-70/2.0，多光圈多焦段，自动采集、对齐好。
- **用途：** 真实感微调主力 + M4 评测；f 值已知 → 标定 K 旋钮↔真实光圈映射。
- **下载：** https://huggingface.co/datasets/timseizinger/RealBokeh_3MP （3MP 版）

### BETD（NTIRE 2023 Bokeh Effect Transformation Dataset）
- **内容：** 合成前景 + Sony/Canon 50mm 双镜头实拍背景（f/1.4、f/1.8、f/16），
  **带 alpha mask、视差值、镜头元数据**。
- **用途（future / P-FP 阶段，D49）：** **RQ3 定量化的钥匙**——对两镜头各做指纹标定 → 渲染 lens-to-lens 转换 →
  与 GT 算 PSNR/SSIM（镜头指纹从定性复刻升格为定量评测，见 NETWORK_DESIGN §7-⑤）。
- **下载：** CodaLab 挑战页（需注册）：https://codalab.lisn.upsaclay.fr/competitions/10229

### VABD（Variable Aperture Bokeh Dataset，2024）
- **内容：** 535 组 × 4 光圈（f/1.8、f/2.8、f/8、f/16），同场景多档。
- **用途：** K 旋钮**连续控制**的真实评测：单一模型扫光圈档对 GT。
- **下载：** https://github.com/MoTong-AI-studio/VABM

### EBB!（Everything is Better with Bokeh!，AIM 2019/2020）
- **内容：** 4694 训练对 + 200 val + 200 test（Canon 70D，f/16↔f/1.8）；Val294 为
  BokehMe 系沿用的子集。**对齐差**（社区手工清洗版约 4464 对）。
- **用途：** 传统基准（M4 必报指标）；微调价值低于 RealBokeh。
- AIM 2020 报告：https://arxiv.org/abs/2011.04988

### LFDOF / DPDD（暂不引入）
- 光场合成/双像素的散焦数据，主战场是散焦**去模糊**（反方向）；仅在需要额外
  真实散焦统计时再评估。

**共同短板（全部数据集）**：无像差系数标注、无特色镜头（Helios/Trioplan 级）样张
→ 可控性训练只能靠自有合成管线；真实数据接入可控性叙事的桥 = 指纹标定（已降为 future / P-FP，D49——主线靠合成可控性 + 真实感降权报告）。

---

## 八、渲染工程参考：HDR 加性合成与色调映射（散景盘重叠溢出，见 DECISIONS D41）

> 非学术论文，是游戏/CG 实时散景的工程实践与专利。共同结论：散景盘是**加性 HDR 光**，
> 重叠处线性能量相加可远超 1.0（物理正确，不可用 max/screen），**末端用色调曲线滚降**回
> 显示域，而非硬裁。本项目据此在 `linear_to_srgb` 的 gamma 前插入软肩色调映射（D41）。

### MJP (Matt Pettineo) — *How To Fake Bokeh (And Make It Look Pretty Good)*
- **为何重要：** 实时散景的经典工程参考。明确"焦外是无数 CoC 圆盘叠加、镜面高光过焦点后
  铺满各自 CoC"，并讨论硬盘核 vs 可分离近似的伪影 —— 与我们 geom 硬盘核（D40）一致。
- https://therealmjp.github.io/posts/bokeh/ （另见旧版 https://mynameismjp.wordpress.com/2011/02/28/bokeh/ ）

### Bart Wronski — *Bokeh depth of field – going insane!*（DoF 系列）
- **为何重要：** 详述散景的**加性预乘 alpha 累积 + 归一化除法**（与我们 `out/acc` 同构）、
  CoC 半径加权的能量归一化。重叠高光必须在 HDR 域累加、末端 tone map，是 D41 修法的工程出处。
- https://bartwronski.com/2014/04/07/bokeh-depth-of-field-going-insane-part-1/

### US Patent 11935285 — *Real-time synthetic out of focus highlight rendering*
- **为何重要：** 非 HDR 单图**模拟 HDR**再产生亮斑散景的专利（与我们 WR 逆色调映射恢复 HDR
  同思路）：传感器饱和裁掉的高光信息无法反推，需逆映射重建后才能铺成明亮散景盘。
- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11935285

### US Patent 10572984 — *Method for inverse tone mapping of an image with visual effects*
- **为何重要：** 逆色调映射 + 视觉效果（含散景）的专利，佐证"LDR→HDR 逆映射 + 末端 tone map"
  是工业界标准管线，支撑 D38（WR 平顶按饱和度恢复 HDR）+ D41（末端软肩滚降）的组合。
- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10572984

---

## 九、真实镜头散景形态参考（soap bubble / cream 风格标定，见 DECISIONS D47/D48）

> 用于校准 soap/cream 预设的【真实剖面】，避免把"碗状弱环"当肥皂泡、把"亮心保留结构"当奶油虚化。
> 共同结论：**过矫正球差(soap)=锐利明亮边缘环+较暗填充中心+double-edge；欠矫正球差=低对比柔 glow；
> 真正的奶油【柔虚化】靠【变迹 STF】(柔边填充盘)而非欠矫正球差(后者只给亮心+保留焦外结构)。**

### Meyer-Optik Görlitz Trioplan 100mm f/2.8 — 经典 soap bubble 镜头
- **为何重要：** soap bubble 预设的真实标定对象。过矫正球差 → 背景点光源变成**明亮锐利边缘环 + 较暗中心**
  的"肥皂泡"；广角全开焦内带 ethereal glow（**强亮环 ⟺ 焦内软是物理一体**，印证 D47 trade-off）。
- 评测/样张：http://www.4photos.de/test/Soap-Bubble-Bokeh-Lenses.html

### phillipreeve.net — *Bokeh Explained*（过矫正 vs 欠矫正球差的散景剖面）
- **为何重要：** 明确"过矫正→higher contrast + outlining + double-edge；欠矫正→lower contrast + soft glow +
  darker edges"。是 D48 区分"球差亮心(柔化) vs 变迹(柔虚化)"两种机制的依据。
- https://phillipreeve.net/blog/bokeh-explained/

### Sony STF (Smooth Trans Focus) / 变迹 APD —— 真正的奶油柔虚化机制
- **为何重要：** cream 预设（D48）柔【虚化】的机制来源：光瞳径向透过率衰减 exp(−aρ²) → 柔边填充盘
  （区别于欠矫正球差的"亮心+保留结构"）。本项目 `M_apod` + `cream_soft`(轻球差+变迹) 复刻之。

---

## 附：其它研究计划中列出的参考

- *Parameterized Modeling of Spatially Varying PSF for Lens Aberration and Defocus* — J. Optical Society of Korea（空变 PSF 参数化建模）。
- Wadhwa et al. — *Synthetic Depth-of-Field with a Single-Camera Mobile Phone*（ACM TOG 2018）。
- 市场佐证（motivation 可引）：Voigtländer 2026.02 发布带**球差控制环**的 RF/Z 卡口镜头
  （https://petapixel.com/2026/02/18/voigtlanders-first-lens-with-spherical-aberration-control-comes-to-rf-and-z/ ）；
  Lomography Nour Triplet V 2.0/64 的 SA 旋钮（soft/classic/bubble 三档）——
  "球差可调"存在真实摄影需求，本项目把它软件化、连续化、可反演化。

---

*维护说明：新增参考请按角色归入对应分组，并标注"为何重要"与是否有开源代码。*
