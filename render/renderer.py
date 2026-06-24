"""
render/renderer.py
==================
可微遮挡感知散景渲染器（M2 核心，无权重的纯物理层）。

设计路线见 DECISIONS **D8/D9**（先读它们再改本文件）：
- **分层 gather**：按视差把场景切成 K 层，每层用"该层离焦量"的 PSF 做整层卷积，
  层间从远到近做 alpha 合成 + 归一化 → 遮挡近似（远处不会渗到近处前面）。
- **tile 近似**（视场相关效应）：图像切 tile，每个 tile 用其中心 (像高 H, 方位角) 的
  PSF（从 +x 字典约定旋转到 tile 朝向）——彗差/像散/猫眼/LaCA 由此显现。
- **PSF 现场算（D9）**：每层离焦量是确定值，直接调 optics 算 PSF，不走字典插值
  （避免"双环鬼影"）。每前向 ≈ 层数×H桶×3 个 256² FFT，GPU 毫秒级。
- **离焦换算（标定桥梁）**：像素 CoC 半径 ↔ W020(waves) 用 `optics/calibrate.py`
  产出的 JSON（r ≈ 3.93·W020 + 0.18），见 CLAUDE.md 第 3 节。
- **可微性**：所有步骤（PSF 生成、FFT 卷积、合成）对控制向量 c = (d_f, K, a) 可微。
  层的"视差分桶"只依赖输入视差（不依赖 c），不挡梯度。

管线位置（CLAUDE.md 第 2 节）：
    视差 D ──► 带符号 CoC ──► 分层 ──► [本文件] ──► 散景图 ──► 边界细化网(M2 后半)

坐标/单位约定：
- image: [3,H,W] float，sRGB [0,1]；disparity: [H,W] float [0,1]，**近=1 远=0**（视差！）。
- CoC 半径 r 单位=像素；r>0 背景(焦后)、r<0 前景(焦前)，符号传给 W020 的符号。
- 像高 H ∈ [0,1]：像素到画面中心距离 / 半对角线。方位角 = atan2(dy,dx)（图像坐标，y 向下）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import ClassVar
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_JSON = PROJECT_ROOT / "outputs" / "psf_test" / "defocus_calibration.json"


# ==============================================================================
# 0) 控制向量 c：渲染的全部"旋钮"打包
# ==============================================================================
@dataclass
class RenderControl:
    """渲染控制向量 c = (焦平面 d_f, 光圈 K, 像差系数 a) + 少量渲染选项。

    字段可以是 float，也可以替换成 requires_grad 的 torch 标量（指纹标定/训练时）。
    """
    focus_disparity: float = 0.5     # d_f：对焦的视差值（近=1 远=0）
    aperture_K: float = 40.0         # K：CoC 增益（像素/单位视差），r = K·(D − d_f)
    focus_tolerance: float = 0.0     # 对焦容差 δ0（视差半带宽）：|D−d_f|≤δ0 视为合焦。
                                     # 动机：DA V2 的相对视差会把主体自身拉伸出 0.1~0.3
                                     # 的跨度，K 稍大主体内部就被虚掉；δ0 把整个主体
                                     # 圈进景深带。r = K·sign(δ)·max(|δ|−δ0, 0)。
    coeffs: object = None            # AberrationCoeffs（None = 理想镜头）
    balance_spherical_focus: bool = True
    # ↑ 球差焦移补偿：球差镜头的"最佳焦面"不在 W020=0 处，实拍时摄影师会重新对焦，
    #   用离焦抵消部分球差（波前平衡：W020=−W040 时 RMS 最小，ρ⁴−ρ² 是平衡球差）。
    #   开启后给所有层加恒定 W020 偏移 −W040（=整体移焦），合焦主体核心明显收紧，
    #   而焦外的肥皂泡/奶油形态保留。关闭则严格按 W020=0 为焦面（合焦面带满幅光晕）。
    n_layers: int = 24               # 视差分层数（越多越平滑、越慢）
    pupil_size: int = 256            # 光瞳/FFT 网格（决定 W020 上限 N/8）
    gamma: float = 2.2               # sRGB→线性 的近似 gamma（HDR 高光在线性域卷积）
    highlight_gain: float = 0.0      # 高光增益（>0 开启：让过曝点源出亮斑散景/glow）
    highlight_thresh: float = 0.9    # 高光【检测】阈值（线性亮度）：>此值的像素做成平顶方波散景盘。
    highlight_hdr_thresh: float = 0.92  # 高光【HDR 能量放大】阈值（D42）：仅对【真正被裁剪到接近饱和】
                                     # (d→1) 的像素做 HDR 恢复放大；中等亮度灯珠(thresh~hdr_thresh)只做
                                     # 方波盘、不放大能量。物理依据：只有被传感器裁剪的点才丢失 HDR 信息、
                                     # 才需恢复；d<hdr_thresh 的点未过曝、真实亮度即原值、放大它=凭空造能量
                                     # （D42 修：旧版 sat 从 highlight_thresh=0.82 起放大 → d=0.9 的灯珠被放大
                                     # 到 27 倍，散景球过曝死白。分离后中等灯珠成柔和彩色盘、真饱和核心才成亮球）。
    highlight_softness: float = 0.05 # WR 方波掩膜温度（越小边缘越锐；过小则梯度消失/锯齿）。
                                     # 控制高光重分配后弥散圆边缘的锐利程度，见 redistribute_highlights。
    highlight_per_region: bool = True # WR 逐区平顶高度（D34/BokehMe++ Algorithm1）：按每个高光区
                                     # 的【bloom（柔光晕）】估各自平顶峰值，保留亮度差异（vs 全局统一 peak）。
                                     # 需 scipy；不可用时回落全局。逐区 peak 图 detach（不破坏指纹 PSF 梯度）。
    psf_mode: str = "geom"           # PSF 计算方式（D40）：'geom'=几何光线散射(默认,硬边散景盘,
                                     # 像差焦散自然涌现,无衍射软边/菲涅耳环,无采样上限)；'wave'=波动光学
                                     # |FFT{P}|²(软边,可出衍射/星芒,小离焦近焦区更准)。视觉散景盘用 geom。
    geom_samples: int = 512          # 几何散射光瞳网格分辨率（点密度；越大盘越平滑）。
    geom_softness: float = 0.01      # 几何模式孔径软度（直接=盘边软度，取小得硬边）。
    highlight_tonemap: bool = True   # HDR→显示【软肩色调映射】（D41）：散景盘是【加性 HDR 光】，
                                     # 多个盘重叠处线性能量相加可远超 1.0（实测密集灯串 40% 像素溢出）。
                                     # 旧 linear_to_srgb 只 clamp 下界 → 上界被保存时硬裁成平白块、丢盘层次。
                                     # 开启后用 knee+指数软肩把 >knee 的部分平滑滚降逼近 1（重叠区保留梯度/盘边）。
                                     # 物理依据：真实相机也是 HDR 加性 + 传感器/色调曲线滚降（非硬裁）。
    tonemap_knee: float = 0.8        # 软肩拐点（线性域）：L≤knee 恒等（不动正常曝光/主体），
                                     # L>knee 压缩。越低 HDR 余量越大但近白内容压得越多；0.8 是折中。

    def __post_init__(self):
        if self.coeffs is None:
            from optics.aberrations import AberrationCoeffs
            self.coeffs = AberrationCoeffs()

    # 自适应分层（D31）：相邻层 CoC 步长上限(px) 与层数封顶。
    # ClassVar → 不是 dataclass 字段，不进构造函数签名。
    MAX_COC_STEP_PX: ClassVar[float] = 3.0
    N_LAYERS_CAP: ClassVar[int] = 40

    def effective_n_layers(self) -> int:
        """实际渲染分层数：在配置 n_layers 基础上按光圈 K【自适应增加】，使相邻层
        的 CoC 步长 ≤ MAX_COC_STEP_PX —— 否则单个点光源会被 tent 权重拆到两个 CoC
        差异过大的层（如 17px 与 5px），合成后呈"大盘叠小亮盘"=【弥散圆中心亮点】
        伪影（D31；本质同 D9 的"双环鬼影"，只是发生在层合成而非字典插值）。

        层中心覆盖视差 [0,1]，相邻层 CoC 步长 = K/(n−1)；令其 ≤ s → n ≥ 1 + K/s。
        **只增不减**（下限=配置 n_layers）：低 K 完全不变（无回归、不动既有训练分布），
        仅大虚化(高 K)时加层；空层会被渲染循环跳过，景深集中时几乎不增算力。
        分层数是离散选择、不可微 → 这里对 K 取 detach 的 float，不影响指纹标定梯度。
        """
        import math
        import torch
        K = float(torch.as_tensor(self.aperture_K).detach().abs())
        need = int(math.ceil(1.0 + K / self.MAX_COC_STEP_PX))
        return max(int(self.n_layers), min(need, self.N_LAYERS_CAP))


# ==============================================================================
# 1) 标定换算：CoC 像素半径 ↔ W020(waves)
# ==============================================================================
def load_defocus_calibration(path: Path = CALIB_JSON) -> tuple[float, float]:
    """读离焦标定 (slope, intercept)：r_px = slope·W020 + intercept。

    JSON 由 `python -m optics.calibrate` 产出；不存在时退回理论值 (4.0, 0.0) 并提醒。
    """
    if path.exists():
        d = json.loads(path.read_text())
        return float(d["slope_px_per_wave"]), float(d["intercept_px"])
    import warnings
    warnings.warn(f"[render] 未找到标定文件 {path}，退回理论斜率 r=4·W020。"
                  f"建议先运行 python -m optics.calibrate。")
    return 4.0, 0.0


def coc_to_w020(r_px, slope: float, intercept: float, pupil_size: int):
    """带符号 CoC 像素半径 → 带符号 W020(waves)，并按采样上限 N/8 截断（防混叠）。

    【可微性】r_px 可以是 float，也可以是带梯度的 torch 标量（d_f/K 是控制向量
    的一部分，指纹标定/训练时需要梯度流过这里）——torch 分支全用张量算子，
    不能用 math.copysign / Python max（它们会把张量转 float、悄悄断梯度链）。
    """
    import torch
    limit = pupil_size / 8.0 * 0.95          # 留 5% 余量，guard 不该被触发
    if isinstance(r_px, torch.Tensor):
        mag = ((r_px.abs() - intercept).clamp(min=0.0) / slope).clamp(max=limit)
        return mag * torch.sign(r_px)
    mag = min(max(abs(r_px) - intercept, 0.0) / slope, limit)
    return math.copysign(mag, r_px)


def signed_coc(disparity, focus_disparity, aperture_K, focus_tolerance=0.0):
    """带符号散焦圈：r = K·sign(δ)·max(|δ|−δ0, 0)，δ = D − d_f。D 必须是视差（近=1 远=0）！

    r>0：D>d_f（比对焦面更近，前景，焦前）……符号约定只要全链路一致即可；
    本项目约定 r 的符号 = (D − d_f) 的符号，直接作为 W020 的符号传入 PSF。
    δ0 = 对焦容差（见 RenderControl.focus_tolerance）：景深带内 CoC 记 0，
    避免 DA V2 相对视差把主体内部的深度跨度渲染成可见模糊。
    【可微】对 float 与 torch 标量（d_f/K 带梯度时）都成立；relu 的次梯度可用。
    """
    import torch
    delta = disparity - focus_disparity
    if isinstance(delta, torch.Tensor):
        mag = (delta.abs() - focus_tolerance).clamp(min=0.0)
        return aperture_K * mag * torch.sign(delta)
    mag = max(abs(delta) - focus_tolerance, 0.0)
    return aperture_K * math.copysign(mag, delta)


# ==============================================================================
# 2) 基础算子：FFT 线性卷积 / PSF 旋转
# ==============================================================================
def fft_conv2d(img, kernel):
    """对 [C,H,W] 图像与 [C,k,k] 逐通道核做"same"线性卷积（FFT 实现，可微）。

    为什么用 FFT：散景核可达 100px+，空域卷积 O(HW·k²) 太慢；FFT 与核大小无关。
    实现：零填充到 (H+k−1, W+k−1) 避免循环卷绕，结果取中心对齐窗口。
    """
    import torch
    C, H, W = img.shape
    k = kernel.shape[-1]
    # 线性卷积的完整输出尺寸是 H+k−1：FFT 是【循环】卷积，必须零填充到这个尺寸
    # 才不会让图像边缘"卷绕"到另一侧（rfft2 的 s= 参数会自动右/下补零）。
    P_h, P_w = H + k - 1, W + k - 1
    fi = torch.fft.rfft2(img, s=(P_h, P_w))     # 图像频谱（rfft 利用实输入省一半计算）
    fk = torch.fft.rfft2(kernel, s=(P_h, P_w))  # 核频谱
    y = torch.fft.irfft2(fi * fk, s=(P_h, P_w)) # 频域逐点乘 = 空域卷积
    # "same" 对齐：完整输出里截出以核中心为锚的 H×W 窗口。
    # 核中心 = k//2，与 pupil_to_psf 的偶数 crop 约定一致（fftshift 后 DC 在 crop//2）。
    # 偶数核理论上有 0.5px 的系统偏移，但 PSF 生成与卷积取窗用的是【同一个】中心
    # 约定，全链路自洽、层间无相对错位，故无需特殊处理。
    c = k // 2
    return y[:, c:c + H, c:c + W]


def rotate_psf(psf, angle: float):
    """把 +x 约定的 PSF 旋转 angle 弧度，使其"径向轴"指向目标方位角（可微 grid_sample）。

    【符号推导（曾出过 90° bug，勿凭感觉改）】affine_grid 的矩阵作用在【采样坐标】上：
    p_in = M·p_out，即输出(p) = 输入(M·p)。M=R(+a) 时内容呈现为旋转 **−a**。
    我们要内容旋转 +angle（让 PSF 的 +x 特征轴指向方位角 angle），
    故 M 必须取 R(−angle) = [[cos, sin], [−sin, cos]]。
    注意验证陷阱：PSF 形状是"轴"(mod π)，在上下左右四个正方向上 ±angle 不可分辨，
    **必须用对角位置（如 45°）验证**——见 outputs/render_test/psf_grid_catseye.png。
    """
    import torch
    import torch.nn.functional as F
    if abs(angle) < 1e-6:
        return psf
    C, h, w = psf.shape
    ca, sa = math.cos(angle), math.sin(angle)
    theta = torch.tensor([[[ca,  sa, 0.0],
                           [-sa, ca, 0.0]]], dtype=psf.dtype, device=psf.device)
    grid = F.affine_grid(theta, size=(1, C, h, w), align_corners=False)
    out = F.grid_sample(psf[None], grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)[0]
    # 旋转重采样会轻微损失能量（边角+插值），重新归一化保持"能量分配核"语义。
    return out / (out.flatten(1).sum(-1)[:, None, None] + 1e-12)


# ==============================================================================
# 2.5) 视差边缘吸附（修 DA V2 的"边界软坡"）
# ==============================================================================
def snap_disparity_edges(disparity, window: int = 7, edge_thresh: float = 0.08,
                         iterations: int = 3):
    """把深度边界处的"软坡带"吸附到两侧台面 —— 渲染前的视差预处理。

    问题：DA V2 在物体边界给出 10~20px 的渐变坡（既不属于前景也不属于背景），
    这些像素落在中间视差层 → 渲染出"半虚化"的轮廓圈，主体边缘很拙劣。
    做法：每个像素看局部窗口的 min/max；若 max−min 超过阈值（=真边界，而非
    平缓的地面渐变），就把该像素吸附到更近的那一侧极值。迭代几轮把整条坡扫平。
    阈值的作用：地面/墙面的连续渐变在 7px 窗口内变化 ≪ 阈值 → 不受影响。

    注意：这是输入预处理（不在 c 的梯度链路上），训练时网络输入端同样适用。

    Args:
        disparity: [H,W] 视差。
        window: 局部窗口（奇数）。
        edge_thresh: 判定"真边界"的视差落差阈值。
        iterations: 吸附迭代轮数（每轮收窄一段坡）。
    """
    import torch
    import torch.nn.functional as F
    d = disparity[None, None]
    pad = window // 2
    for _ in range(iterations):
        mx = F.max_pool2d(d, window, stride=1, padding=pad)
        mn = -F.max_pool2d(-d, window, stride=1, padding=pad)
        band = (mx - mn) > edge_thresh                 # 只动真边界
        snapped = torch.where((mx - d) <= (d - mn), mx, mn)
        d = torch.where(band, snapped, d)
    return d[0, 0]


def push_band_to_background(disparity, ctrl, band=None, pad_px: int = 8):
    """把【边界带内】像素的视差压到局部背景（更远=更小视差）→ 重渲时该处按背景重度虚化。

    Plan B（DECISIONS D29）的 B_bg 渲染输入：`B_bg = render(image, push_band_to_background(...))`。
    与标准渲染 B_fg(=B_phys) 的差别仅在边界带——带内每像素的视差被换成"邻域内的
    局部背景深度"，于是边界处的内容会像背景一样被重度虚化。网络输出的 matte α 再在
    B_fg / B_bg 间逐像素选择：真前景 α→1 取 B_fg(锐)、真背景 α→0 取 B_bg(虚)。
    因 B_fg/B_bg 都是渲染器产物，输出恒为两张已模糊物理图的凸组合 →【找补结构上不可能】。

    机制：band 内逐像素取邻域内【最小视差】(= 跨过边缘的局部背景深度)。邻域半径取图内
    最大 |CoC|（背景在一个 CoC 距离内一定够得着）。在 1/4 分辨率上做腐蚀(min-pool)省算力，
    与 boundary_band 同款下采样口径；band 外保持原视差不动。

    Args:
        disparity: [H,W] 网络输入侧视差（近=1）。
        ctrl: RenderControl（算最大 |CoC| 与 band）。
        band: 可选 [1,H,W] 或 [H,W] 边界带（None 时用 refine.conditioning.boundary_band 现算）。
        pad_px: 腐蚀核余量。
    Returns:
        [H,W] 修正后视差（带内→局部背景，带外不变）。供 render_field_patch / render_tiled 使用。
    """
    import torch
    import torch.nn.functional as F
    d = disparity[None, None]                              # [1,1,H,W]
    # 邻域半径 = 图内实际最大 |CoC|（≤ 标定上限 128px），与 boundary_band 同口径。
    r_map = signed_coc(disparity, float(ctrl.focus_disparity),
                       float(ctrl.aperture_K), float(ctrl.focus_tolerance))
    rad = min(float(r_map.abs().max()), 128.0)
    k4 = 2 * math.ceil((int(rad) + pad_px) / 4) + 1
    # 1/4 分辨率腐蚀：min-pool = −max-pool(−x)。下采样取局部最小（保守偏背景）→ 大核
    # min-pool 把背景深度传播进带内 → 最近邻还原。结果偏块状无妨（该区域将被重度虚化）。
    d4 = -F.max_pool2d(-d, 4, stride=4)
    d4 = -F.max_pool2d(-d4, k4, stride=1, padding=k4 // 2)
    d_bg = F.interpolate(d4, size=d.shape[-2:], mode="nearest")[0, 0]
    if band is None:
        from refine.conditioning import boundary_band
        band = boundary_band(disparity, ctrl)
    b = band[0] if band.dim() == 3 else band
    return torch.where(b > 0.5, d_bg, disparity)


def foreground_occupancy_gt(disparity_gt, ctrl, pad_px: int = 8):
    """边界带的【前景占比】真值 α_gt ∈[0,1] —— Plan B（D29）matte 头监督。

    定义基于带符号 CoC（r = K·(D − d_f)，r>0=前景/焦前，r<0=背景/焦后）：
        α_gt = clamp(1 + r / scale_px, 0, 1)
            其中 scale_px = max(K / 8, 2)
    - r ≥ 0（合焦 + 前景）→ α_gt = 1.0：保留 B_phys（主体原样）。
    - r < 0（焦后背景）→ 从 1 线性衰减到 0；r = −scale_px 处 α_gt = 0。
    - |r| = scale_px 之外的背景（大模糊量）→ 钳位 0。

    【为什么不用旧的 (D−d_bg)/(d_fg−d_bg)】：
    旧定义在场景有比主体更近的前景物体时，主体视差成了"中间值"，
    α_gt≈0.3，导致网络在带内把主体脸部也替换为 B_bg（背景模糊渲染）→ 人脸被虚化。
    新定义以焦平面为锚点：合焦像素 r=0 直接得 α_gt=1，不受邻域前景干扰。

    Args:
        disparity_gt: [H,W] 合成样本的【真】视差（近=1）。
        ctrl: RenderControl（提供 focus_disparity / aperture_K / focus_tolerance）。
    Returns:
        [1,H,W] α_gt ∈[0,1]。
    """
    import torch
    r_map = signed_coc(disparity_gt, float(ctrl.focus_disparity),
                       float(ctrl.aperture_K), float(ctrl.focus_tolerance))
    # scale_px：背景到 α=0 的 CoC 距离，随光圈自动缩放
    scale_px = max(float(ctrl.aperture_K) / 8.0, 2.0)
    # r ≥ 0（合焦/前景）→ clamp 到 1；r < 0（背景）→ 线性下降至 0
    alpha = (1.0 + r_map.clamp(max=0.0) / scale_px).clamp(0.0, 1.0)
    return alpha[None]


# ==============================================================================
# 3) 视差分层（tent 软权重）
# ==============================================================================
def layer_weights(disparity, n_layers: int,
                  focus_disparity=None, focus_tolerance: float = 0.0):
    """把 [H,W] 视差转成 [L,H,W] 的层权重（线性 tent，相邻两层间平滑过渡）。

    为什么不用硬分桶：硬边界会在深度渐变面上产生"分层断带"伪影；
    tent 权重让每个像素按距离线性分给相邻两层，Σ_l w_l = 1，且实现简单。
    层中心固定取 [0,1] 均匀网格（不依赖控制向量 → 不挡 c 的梯度）。

    【焦带吸附修遮挡漏光（D35）】tent 把一个【锐利前景】（一定在焦平面附近，CoC≈0）
    拆到相邻两层（如 w=0.45/0.55）时，over-合成的层间遮挡是【乘性】 (1−.45)(1−.55)=.247
    漏光，而总覆盖 .45+.55=1 本应全遮 → 背景渗到锐前景上。对策：把【焦带内】
    （|D−d_f|≤tol，本就 CoC≈0）的像素视差吸附到最近层中心 → 它独占单层、tent 权重=1
    → alpha=1 全遮挡。失焦内容（散景盘）不在焦带、保持 tent+乘性（半透明叠加不受影响）；
    GT 路径用精确 blob alpha（不走本函数）→ 训练 GT 不变，仅 B_phys 输入更干净。
    仅 focus_disparity 给定且 tol>0 时激活（展示场景 tol=0 自动不触发）。

    Returns:
        weights: [L,H,W]，非负，逐像素和为 1。
        centers: [L] 各层代表视差。
    """
    import torch
    centers = torch.linspace(0.0, 1.0, n_layers,
                             device=disparity.device, dtype=disparity.dtype)
    delta = centers[1] - centers[0]
    d = disparity
    if focus_disparity is not None and focus_tolerance > 0.0:
        # 焦带内像素（CoC≈0）吸附到最近层中心：round(d/Δ)·Δ。失焦像素保持原值（tent）。
        in_focus = (d - float(focus_disparity)).abs() <= float(focus_tolerance)
        snapped = (d / delta).round() * delta
        d = torch.where(in_focus, snapped.clamp(0.0, 1.0), d)
    d = d[None]                                             # [1,H,W]
    w = (1.0 - (d - centers[:, None, None]).abs() / delta).clamp(min=0.0)
    return w / (w.sum(dim=0, keepdim=True) + 1e-12), centers


# ==============================================================================
# 4) 每层 PSF（现场算，D9）
# ==============================================================================
def _abs_f(x) -> float:
    """取标量绝对值并 detach 成 float —— coeffs 字段可能是带梯度的 torch 标量
    （镜头指纹标定时），核尺寸/halo 是离散选择不该（也不能）带梯度。"""
    import torch
    return float(torch.as_tensor(x).detach().abs())


def psf_extent_px(coeffs, w020_total, H_img: float) -> float:
    """估计 PSF 的【外缘半径】(px) —— 由各像差项在光瞳边缘(ρ=1)的最大光线斜率累加。

    物理推导（与 optics/calibrate.py 同一套：r_px = 2·(dW/dρ|_{ρ=1})，
    单位 waves 的波前在 N 点 [-1,1] 网格上 FFT 后，边缘光线落点 = 2·空间频率）：
        defocus  W020·ρ²            → 4·|W020|
        spherical W040·ρ⁴           → 8·|W040|        （旧实现【漏掉】，导致肥皂泡亮边被裁）
        coma     W131·H·ρ³·cosθ     → 6·H·|W131|      （单侧甩尾，取最大侧）
        astigmat W222·H²·ρ²·cos²θ   → 4·H²·|W222|
        field    W220·H²·ρ²         → 4·H²·|W220|
    各项按绝对值【相加】（保守上界：不同项可能在边缘同向叠加），再加安全余量。
    w020_total 是该层实际写入 PSF 的总离焦（含基底偏移与球差焦移补偿）。

    注：这是离散核尺寸/窗口的估算，全程 detach，不进梯度图。
    """
    H = abs(float(H_img))
    return (4.0 * _abs_f(w020_total)
            + 8.0 * _abs_f(coeffs.W040_spherical)
            + 12.0 * _abs_f(coeffs.W060_spherical2)     # 高阶球差 ρ⁶ → 边缘斜率 6·W060，外缘 12·|W060|（D34）
            + 6.0 * H * _abs_f(coeffs.W131_coma)
            + 4.0 * H * H * (_abs_f(coeffs.W222_astigmatism)
                             + _abs_f(coeffs.W220_field_curv))
            + 4.0)                                      # 衍射极限/软边/插值的安全余量


def _layer_psf(grid, H_img: float, coeffs, w020_layer: float, pupil_size: int,
               check: bool = False, balance_spherical: bool = False,
               psf_mode: str = "geom", geom_samples: int = 512,
               geom_softness: float = 0.01):
    """给定层离焦 w020 与像高 H，现场算一组 RGB PSF（[3,k,k]，k 自适应离焦大小）。

    crop 自适应：盘外缘半径 ≈ psf_extent_px（含【所有】像差项，不只离焦），
    crop 取 2·半径+余量（偶数，中心=crop//2），小核层卷积更快。
    旧实现只按 |W020| 估半径 → 近焦面的强球差(肥皂泡)PSF 被裁掉边缘亮环、
    丢失能量后重归一化把截断悄悄掩盖；改用 psf_extent_px 计入 W040/视场项后消除。
    balance_spherical：叠加球差焦移补偿 −W040（见 RenderControl.balance_spherical_focus）。
    psf_mode：'geom'=几何光线散射(硬边,默认) / 'wave'=波动光学 FFT(软边)，见 D40。
    """
    from optics import psf as psf_mod
    w020_eff = w020_layer
    if balance_spherical:
        # 球差焦移补偿（最小弥散圆，D46/D53）：球差镜头"最佳对焦"在【最小弥散圆】而非傍轴焦点
        # （摄影看焦内主体清晰度=弥散圆大小）。给所有层加 −Σ(系数·球差) 的恒定移焦。
        # 各球差【阶】的最小弥散圆系数是与系数值无关的物理常数（数值实测）：
        #   初级球差 W040(ρ⁴) → 1.3；高阶球差 W060(ρ⁶) → 1.5。
        # 【D53 补 W060】旧版只补 W040：nisen 等强 W060 镜头焦内残余大（90%能量半径 ~9.7px）、
        # 且前/后焦盘严重不对称（balance 偏移只含 W040 → 前后 W020 不对称、被 W060 高次项放大到
        # ~2.5×）。补上 W060 后 nisen 焦内锐化到 ~3px、前后盘比 1.9→1.3（接近初级球差），消除
        # 用户反馈的"W060 前后弥散圆大小不一致"。焦外大离焦盘的形态几乎不受影响（偏移相对大离焦小）。
        BALANCE_W040 = 1.3
        BALANCE_W060 = 1.5
        w020_eff = (w020_eff - BALANCE_W040 * coeffs.W040_spherical
                             - BALANCE_W060 * coeffs.W060_spherical2)
    # 实际写入 PSF 的总离焦（含基底偏移），用于按物理估外缘半径。
    w020_total = coeffs.W020_defocus + w020_eff
    radius = psf_extent_px(coeffs, w020_total, H_img)
    # 几何模式无 FFT 网格上限，但仍限核大小以控卷积开销；波动模式不得超 pupil_size(混叠)。
    cap = max(pupil_size, 2 * math.ceil(radius) + 8) if psf_mode == "geom" else pupil_size
    crop = int(min(cap, 2 * math.ceil(radius) + 8))
    crop += crop % 2                                    # 取偶，使 PSF 中心恰在 crop//2
    c_l = coeffs.replace(W020_defocus=coeffs.W020_defocus + w020_eff)
    if psf_mode == "geom":
        dev = grid["rho"].device
        return psf_mod.rgb_geometric_psf(H_img, c_l, crop=crop, samples=geom_samples,
                                         softness=geom_softness, device=dev)
    return psf_mod.rgb_psf(grid, H_img, c_l, crop=crop, check_sampling=check)


# ==============================================================================
# 5) 主渲染：全局 PSF 模式（H=0，快速路径）
# ==============================================================================
def _composite_layers(img_lin, weights, centers, ctrl, grid, slope, intercept,
                      H_img: float = 0.0, azimuth: float = 0.0):
    """分层卷积 + 从远到近 alpha 合成（被全局/tile 两种模式共用的核心循环）。

    遮挡近似的关键：从最远层开始向近处叠，每层用"该层模糊后的覆盖率 A_l"
    遮挡身后的累积结果——近处的清晰前景能正确压住远处的模糊背景，
    而模糊前景的半透明边缘也能让背景部分透出（边界溢色由细化网再修）。
    """
    import torch
    out = torch.zeros_like(img_lin)
    acc = torch.zeros_like(img_lin[:1])
    # centers 升序 = 视差小→大 = 远→近，正好是合成需要的顺序。
    for l in range(weights.shape[0]):
        w_l = weights[l:l + 1]                          # [1,H,W] 该层覆盖率
        if float(w_l.sum()) < 1e-6:
            continue                                     # 空层跳过（常见，省大量卷积）
        r_l = signed_coc(float(centers[l]), ctrl.focus_disparity, ctrl.aperture_K,
                         ctrl.focus_tolerance)
        w020 = coc_to_w020(r_l, slope, intercept, ctrl.pupil_size)
        psf = _layer_psf(grid, H_img, ctrl.coeffs, w020, ctrl.pupil_size,
                         balance_spherical=ctrl.balance_spherical_focus,
                         psf_mode=ctrl.psf_mode, geom_samples=ctrl.geom_samples,
                         geom_softness=ctrl.geom_softness)
        if abs(azimuth) > 1e-6:
            psf = rotate_psf(psf, azimuth)
        # 模糊后的层颜色：RGB 各用自己通道的 PSF 卷积（色差由此进入图像）。
        B = fft_conv2d(img_lin * w_l, psf)
        # 模糊后的层覆盖率 alpha：物理上 alpha 是几何遮挡，与波长无关 →
        # 用三通道 PSF 的【均值核】卷积单通道权重。
        # （旧实现误取 R 通道 [:1]，LoCA 开启时会与颜色通道失配；均值核还省 2/3 计算。）
        A = fft_conv2d(w_l, psf.mean(dim=0, keepdim=True)).clamp(0.0, 1.0)
        # 从远到近的 over 合成：当前层颜色叠在"被自己遮挡剩下的"累积结果上。
        out = B + (1.0 - A) * out
        acc = A + (1.0 - A) * acc                        # 累积覆盖率（同样的 over 规则）
    return out / acc.clamp(min=1e-6)                     # 归一化补回边缘能量损失


def render_global(image, disparity, ctrl: RenderControl):
    """全局 PSF 渲染（视场无关，H=0）：验证离焦/球差/光圈形状/LoCA 的快速路径。

    Args:
        image: [3,H,W] sRGB [0,1] 张量。
        disparity: [H,W] [0,1] 视差（近=1）。
        ctrl: RenderControl。
    Returns:
        [3,H,W] sRGB 散景图。
    """
    from optics import pupil as pupil_mod
    slope, intercept = load_defocus_calibration()
    grid = pupil_mod.make_pupil_grid(ctrl.pupil_size, device=str(image.device),
                                     dtype=image.dtype)
    img_lin = srgb_to_linear(image, ctrl)
    weights, centers = layer_weights(disparity, ctrl.effective_n_layers(),
                                     ctrl.focus_disparity, ctrl.focus_tolerance)
    out_lin = _composite_layers(img_lin, weights, centers, ctrl, grid, slope, intercept)
    return linear_to_srgb(out_lin, ctrl)


# ==============================================================================
# 6) 主渲染：tile 视场相关模式（D8 —— 彗差/像散/猫眼/旋焦/LaCA 在此显现）
# ==============================================================================
def render_tiled(image, disparity, ctrl: RenderControl, tile: int = 64,
                 H_bins: int = 6):
    """tile 近似的视场相关渲染：每 tile 用其中心 (H, 方位角) 的 PSF。

    实现要点：
    - 每 tile 取带 halo（=最大核半径）的邻域做分层卷积，保证 tile 边界处的模糊不缺邻域信息。
    - **重叠 + Hann 窗羽化（消 tile 接缝，D52）**：相邻 tile 用各自中心 (H,方位角) 的 PSF，
      硬拼接会在散景盘横跨 tile 边界处留下可见接缝（盘内十字缝/缺口）。改为 tile 以 50% 步长
      【重叠】、每 tile 的 (out, alpha) 乘 2D Hann 窗后【加权累加】→ 不同 PSF 在重叠区平滑过渡、
      盘内无缝（视场效应本就随位置缓变，混合物理上合理）。
    - **归一化口径（D54）**：`out/acc` 归一化【移到全图做一次】（不在每 tile 内做）——使重叠区按
      alpha 加权混合、与 global 路径同口径（逐 tile 独立归一化是非线性除法，重叠拼接不自洽）。
    - **已知限制：PSF 空变的能量误差（D54）**：tile gather 模式下，点光源散景盘横跨多 tile 时，
      盘的不同部分被各 tile 各自 (H桶,方位角) 的 PSF 渲染再拼接，块间 PSF 不同 → 总能量偏离 global
      约 +3%(圆对称 ideal)~+9%(强不对称如彗差)，且误差集中在盘边缘。这是 tile【近似】的固有代价
      （非归一化/halo/overlap 能修，需 scatter 或空变卷积才根治），**只影响 e2e 整图推理的视觉亮度、
      不影响训练**（训练走 render_field_patch 单 PSF 整块=与 global 同口径、守恒）。真实照片的连续
      散景相邻盘重叠、误差被平均，远不如孤立点光源显著。
    - PSF 按 (层, H 桶) 缓存（+x 约定），每 tile 只做一次旋转 → 计算量可控（重叠使 tile 数约 4×）。
    - 口径蚀的边角失光：tile 结果乘 relative_transmission(T(H))（CLAUDE.md 第 3 节）。
    """
    import torch
    from optics import pupil as pupil_mod
    slope, intercept = load_defocus_calibration()
    grid = pupil_mod.make_pupil_grid(ctrl.pupil_size, device=str(image.device),
                                     dtype=image.dtype)
    _, Hh, Ww = image.shape
    cy, cx = (Hh - 1) / 2.0, (Ww - 1) / 2.0
    half_diag = math.sqrt(cy ** 2 + cx ** 2)

    img_lin = srgb_to_linear(image, ctrl)
    weights, centers = layer_weights(disparity, ctrl.effective_n_layers(),
                                     ctrl.focus_disparity, ctrl.focus_tolerance)

    # halo = 全图可能出现的最大核半径 + 余量。必须计入【像差外扩】（球差亮环/彗差甩尾），
    # 否则强像差 PSF 的边缘在 tile 拼接处被邻域不足截断 → 接缝。取两个离焦极值
    # （视差 0 与 1）在画幅最边缘 H=1（视场效应最强）的物理外缘半径之最大值。
    # halo 是离散窗口尺寸，全程 detach（d_f/K 可能是带梯度张量），不进梯度图。
    def _extent_at_disp(d: float) -> float:
        r = signed_coc(d, ctrl.focus_disparity, ctrl.aperture_K, ctrl.focus_tolerance)
        w020 = coc_to_w020(r, slope, intercept, ctrl.pupil_size)
        w020_eff = w020
        if ctrl.balance_spherical_focus:
            # 与 _layer_psf 同口径补偿 W040+W060（D53），保证 halo 估算覆盖补偿后的实际离焦。
            w020_eff = (w020_eff - 1.3 * ctrl.coeffs.W040_spherical
                                 - 1.5 * ctrl.coeffs.W060_spherical2)
        w020_total = ctrl.coeffs.W020_defocus + w020_eff
        return psf_extent_px(ctrl.coeffs, w020_total, 1.0)
    halo = int(math.ceil(max(_extent_at_disp(0.0), _extent_at_disp(1.0)))) + 8

    # PSF 缓存：键 (层号, H桶号) → +x 约定的 [3,k,k]。tile 间复用，只差一次旋转。
    psf_cache: dict[tuple[int, int], object] = {}

    def get_psf(l: int, hb: int):
        key = (l, hb)
        if key not in psf_cache:
            r_l = signed_coc(float(centers[l]), ctrl.focus_disparity, ctrl.aperture_K,
                             ctrl.focus_tolerance)
            w020 = coc_to_w020(r_l, slope, intercept, ctrl.pupil_size)
            H_val = hb / max(H_bins - 1, 1)
            psf_cache[key] = _layer_psf(grid, H_val, ctrl.coeffs, w020, ctrl.pupil_size,
                                        balance_spherical=ctrl.balance_spherical_focus,
                                        psf_mode=ctrl.psf_mode, geom_samples=ctrl.geom_samples,
                                        geom_softness=ctrl.geom_softness)
        return psf_cache[key]

    # 重叠 + Hann 窗羽化（D52，消 tile 接缝）：tile 以 50% 步长重叠，每 tile 乘 2D Hann
    # 窗加权累加，最后除以权重和。Hann 端点加 0.05 基底防图像最边缘权重和过小（除零）。
    tile_h, tile_w = min(tile, Hh), min(tile, Ww)
    step_y, step_x = max(tile_h // 2, 1), max(tile_w // 2, 1)

    def _starts(total, win, st):
        """窗口左上角起点列表：均匀步进、最后一个对齐到 total−win 以盖满边界。"""
        if total <= win:
            return [0]
        s = list(range(0, total - win + 1, st))
        if s[-1] != total - win:
            s.append(total - win)
        return s

    hwin_y = torch.hann_window(tile_h, periodic=False, device=img_lin.device,
                               dtype=img_lin.dtype) + 0.05
    hwin_x = torch.hann_window(tile_w, periodic=False, device=img_lin.device,
                               dtype=img_lin.dtype) + 0.05
    win2d = (hwin_y[:, None] * hwin_x[None, :])[None]      # [1,tile_h,tile_w] 羽化权重

    # D54：累加【未归一化】的 out 与 alpha，末端整图除一次（保能量守恒，见 docstring）。
    out = torch.zeros_like(img_lin)
    acc_sum = torch.zeros_like(img_lin[:1])                # [1,H,W] Hann 加权的 alpha 累加（全图归一化分母）
    for ty in _starts(Hh, tile_h, step_y):
        for tx in _starts(Ww, tile_w, step_x):
            y1, x1 = ty + tile_h, tx + tile_w
            # tile 中心 → 像高 H 与方位角（图像坐标，y 向下）。
            tcy, tcx = (ty + y1 - 1) / 2.0, (tx + x1 - 1) / 2.0
            H_img = math.sqrt((tcy - cy) ** 2 + (tcx - cx) ** 2) / half_diag
            azim = math.atan2(tcy - cy, tcx - cx)
            hb = int(round(H_img * (H_bins - 1)))

            # 带 halo 的邻域（边界处自动截到图内，零填充由卷积 pad 隐式处理）。
            hy0, hx0 = max(ty - halo, 0), max(tx - halo, 0)
            hy1, hx1 = min(y1 + halo, Hh), min(x1 + halo, Ww)
            sub_img = img_lin[:, hy0:hy1, hx0:hx1]
            sub_w = weights[:, hy0:hy1, hx0:hx1]

            # 该 tile 的分层合成：PSF 取自缓存 + 旋转到 tile 方位角；返回未归一化 (out, alpha)。
            sub_out, sub_acc = _composite_tile(sub_img, sub_w, centers, ctrl, get_psf, hb, azim)
            # 裁回本 tile 窗口，out 与 alpha 各乘 Hann 窗后加权累加（重叠区平滑混合）。
            po = sub_out[:, ty - hy0:y1 - hy0, tx - hx0:x1 - hx0]
            pa = sub_acc[:, ty - hy0:y1 - hy0, tx - hx0:x1 - hx0]
            out[:, ty:y1, tx:x1] += po * win2d
            acc_sum[:, ty:y1, tx:x1] += pa * win2d
    out = out / acc_sum.clamp(min=1e-6)                    # 整图归一化一次：Hann 约分、能量守恒

    # 口径蚀的边角失光 T(H)：形状在 PSF 里，【能量】在这里乘回。
    # 用【逐像素平滑 T 图】而不是逐 tile 常量——tile 常量会在天空等平滑区域
    # 留下肉眼可见的亮度阶梯接缝。做法：在 H 桶上算 T（光瞳积分，便宜），
    # 再按每个像素的连续像高 H 在桶间线性插值。
    if ctrl.coeffs.vignette_strength != 0.0:
        T_bins = torch.stack([
            pupil_mod.relative_transmission(grid, hb / max(H_bins - 1, 1), ctrl.coeffs)
            for hb in range(H_bins)
        ])                                                 # [H_bins]，T(0)=1 单调降
        ys = torch.arange(Hh, device=out.device, dtype=out.dtype)
        xs = torch.arange(Ww, device=out.device, dtype=out.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        H_map = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt() / half_diag   # 每像素像高 [H,W]
        idx = (H_map * (H_bins - 1)).clamp(0, H_bins - 1 - 1e-6)       # 桶内连续坐标
        i0 = idx.floor().long()                            # 左桶号
        frac = idx - i0.to(idx.dtype)                      # 桶内小数部分（插值权重）
        T_map = T_bins[i0] * (1.0 - frac) + T_bins[i0 + 1] * frac      # 线性插值 [H,W]
        out = out * T_map[None]                            # 广播到 3 通道
    return linear_to_srgb(out, ctrl)


def _composite_tile(img_lin, weights, centers, ctrl, get_psf, hb: int, azim: float):
    """tile 版分层合成：与 _composite_layers 同逻辑，但 PSF 走缓存+旋转。

    【D54】返回【未归一化】的 (out, acc)——归一化 `out/acc` 由 render_tiled 在【全图】
    层面做一次（逐 tile 独立归一化会在多 tile 拼接处重复计入能量，破坏守恒）。
    """
    import torch
    out = torch.zeros_like(img_lin)
    acc = torch.zeros_like(img_lin[:1])
    for l in range(weights.shape[0]):
        w_l = weights[l:l + 1]
        if float(w_l.sum()) < 1e-6:
            continue                                      # 空层跳过，省卷积
        psf = rotate_psf(get_psf(l, hb), azim)            # 缓存的 +x PSF → 转到 tile 朝向
        B = fft_conv2d(img_lin * w_l, psf)                # 层颜色（逐通道 PSF）
        # alpha 用均值核（理由同 _composite_layers：alpha 与波长无关）。
        A = fft_conv2d(w_l, psf.mean(dim=0, keepdim=True)).clamp(0.0, 1.0)
        out = B + (1.0 - A) * out
        acc = A + (1.0 - A) * acc
    return out, acc                                       # 不在此归一化（移到全图，D54 保能量守恒）


# ==============================================================================
# 6.5) 已知 alpha 的精确分层合成（在线合成 GT 用，CLAUDE.md 第 7 节）
# ==============================================================================
def composite_blurred_layers(layers, ctrl: RenderControl,
                             H_field: float = 0.0, azimuth: float = 0.0,
                             H_centers=None, H_weights=None):
    """对【已知 alpha 的离散图层】做精确散景合成 —— data/synth.py 生成训练 GT 的核心。

    与 render_*（从估计视差出发、tent 软分层）的区别：这里每层的 (内容, alpha, 视差)
    都是合成时精确已知的 → 产出的散景图没有深度误差，是干净的监督信号；
    网络输入侧则只给"单图 + 加扰动的深度"（见 data/synth.py），二者分离是
    防 sim-to-real 边界崩坏的关键（BokehMe 的做法）。

    视场相关训练的省钱技巧：训练裁块小（≤512px），块内像高 H 近似常数 →
    每个样本把整块当作位于画幅 (H_field, azimuth) 处，用单个旋转后的 PSF 渲染。
    与 tile 近似（D8）物理一致，却让猫眼/彗差/像散也能拿到廉价监督。

    【H 子带模式（P1a 第二步，NETWORK_DESIGN §4.1）】传 H_centers + H_weights 时，
    块内像高不再视为常数：对每个子带中心分别完整渲染，再按逐像素 tent 权重混合
    —— 混合的是【渲染结果图】（同一内容、连续变化的视场效应），不是 PSF 本身，
    不触 D9 的"双环鬼影"禁区（那是不同尺寸 PSF 的图像插值）。方位角仍取块中心
    常数（近画幅中心处方位角变化剧烈，但那里 H≈0、视场效应本身消失，误差无害）。

    Args:
        layers: [(rgb [3,H,W] 线性域, alpha [1,H,W] ∈[0,1], disparity float), ...]
                顺序任意（内部按视差升序=远→近合成）。
        ctrl: RenderControl（提供 d_f, K, coeffs, pupil_size）。
        H_field: 该裁块的归一化像高（0=画面中心；子带模式下不使用）。
        azimuth: 该裁块的方位角（弧度），PSF 从 +x 约定旋转到此朝向。
        H_centers: 可选，子带像高中心列表（均匀间隔）。
        H_weights: 可选，[n_sub,H,W] 逐像素子带权重（Σ=1，tent）。

    Returns:
        [3,H,W] 线性域散景图（已乘 T(H) 边角失光；caller 自己做 gamma 还原）。
    """
    import torch
    from optics import pupil as pupil_mod
    slope, intercept = load_defocus_calibration()
    dev = layers[0][0].device
    grid = pupil_mod.make_pupil_grid(ctrl.pupil_size, device=str(dev),
                                     dtype=layers[0][0].dtype)

    def _render_at(H_val: float):
        """单一像高 H_val 下的完整分层合成（原单 PSF 路径）。"""
        out = torch.zeros_like(layers[0][0])
        acc = torch.zeros_like(layers[0][1])
        for rgb, alpha, d in sorted(layers, key=lambda t: float(t[2])):
            r = signed_coc(float(d), ctrl.focus_disparity, ctrl.aperture_K,
                           ctrl.focus_tolerance)
            w020 = coc_to_w020(r, slope, intercept, ctrl.pupil_size)
            psf = rotate_psf(_layer_psf(grid, H_val, ctrl.coeffs, w020,
                                        ctrl.pupil_size,
                                        balance_spherical=ctrl.balance_spherical_focus,
                                        psf_mode=ctrl.psf_mode, geom_samples=ctrl.geom_samples,
                                        geom_softness=ctrl.geom_softness),
                             azimuth)
            B = fft_conv2d(rgb * alpha, psf)              # 预乘 alpha 的层颜色卷积
            # alpha 用均值核（与 _composite_layers 同理：几何遮挡与波长无关）。
            A = fft_conv2d(alpha, psf.mean(dim=0, keepdim=True)).clamp(0.0, 1.0)
            out = B + (1.0 - A) * out
            acc = A + (1.0 - A) * acc
        out = out / acc.clamp(min=1e-6)
        T = pupil_mod.relative_transmission(grid, H_val, ctrl.coeffs)
        return out * T                                    # 每子带乘各自 T(H)

    if H_centers is None or len(H_centers) == 1:
        return _render_at(float(H_centers[0]) if H_centers else H_field)
    out = torch.zeros_like(layers[0][0])
    for b, h in enumerate(H_centers):
        out = out + H_weights[b:b + 1] * _render_at(float(h))
    return out


# ==============================================================================
# 7) HDR 高光（glow/亮斑散景的来源，参考 BokehMe highlight 思路）
# ==============================================================================
def redistribute_highlights(lin, ctrl: RenderControl):
    """高光权重重分配（WR, BokehMe++ 的 Weight Redistribution）——锐利亮斑散景的要点。

    【问题】8-bit / 有限动态范围的全焦原图里，过曝高光点在拍摄时已经能量溢出，
    并带一圈柔光晕：它看起来比真实物理尺寸【大】、边缘已是颜色【渐变】。直接拿这个
    "带衰减边缘"的高光去和散景核卷积，衰减会被叠加放大 → 弥散圆(CoC)边缘像高斯一样
    缓慢衰减，光斑扁平、糊成一团，缺乏真实单反那种晶莹剔透的层次（BokehMe++ Fig 3
    的 w/o WR 反例）。上一版我们只做【乘性提亮】(lin·(1+gain·w))：高光是变亮了，
    但内部仍是带峰的渐变、边缘仍软——正是"高斯波"，没解决这个问题。

    【解法】在散射【之前】把高光区域的能量(权重)分布从"高斯波"强行掰成"方波"：
    内部压成均匀【平顶】、边缘锐化成陡峭【跳变】。于是 PSF ⊛ 平顶方波 ≈ PSF 本身，
    弥散圆获得锐利边缘 + 均匀内部；红蓝高光叠加时呈现实体玻璃折射般的【分层】感
    （w/ WR）。

    【机制】（对照原文三步）：
      1) 以亮度阈值把高光划分为内部 M1 / 边缘 M0 —— sigmoid 软掩膜 m，温度
         highlight_softness 很小 → ~1px 锐边（"方波"的陡跳）。
      2) 压缩内部数值范围 → 均匀平顶：用各像素【自身色度】× 统一平顶亮度 peak 替换原值。
         色度 = lin / 自身亮度，保留红/蓝高光的【色相】；统一 peak 抹平内部的亮度起伏。
      3) 边缘按锐利掩膜 m 重新分配权重：内部取平顶、外部保持原值，m 决定过渡。
    锐利掩膜天然排除了高光外圈【去饱和的柔光晕】（它在阈值以下，m≈0），不会把白雾散出去。

    【逐区平顶（D34，highlight_per_region，默认开）】全局统一 peak 会把所有高光（无论原本
    多亮）压到同一平顶高度 → 丢失高光间的相对亮度。BokehMe++ Algorithm1 按【每个高光区】
    的属性设各自峰值。我们的物理代理：clipped 核都饱和到 ~1 无法分辨亮度，但真实越亮的源
    bloom（柔光晕）越强越宽 → 用【每个区核外柔晕的能量/核尺寸】估其相对亮度，缩放该区平顶。
    对大面积平亮区（如天空，核大、晕少）天然不过度提亮。逐区 peak 图 detach（视作 HDR 估计
    常量），不影响 chroma/m 对 image 的梯度，也不影响指纹对系数的 PSF 梯度链路。

    【可微】chroma/m 全 torch 算子可微；逐区 peak 走 numpy/scipy（detach）。
    """
    import torch
    # 【检测用通道最大值，不用亮度均值】过曝是【逐通道】饱和的：一支饱和的红高光
    # (R≈1,G≈B≈0.15) 亮度均值只有 ~0.34，若按均值判阈值会【漏检】→ 彩色光斑不被
    # 重分配/提亮、停留在原始柔糊态（曾导致彩色弥散圆显著偏暗 + 半成形的不自然亮边）。
    # 改用通道最大值 detect：任一通道饱和即判为高光，红/蓝光斑与白光斑一视同仁。
    detect = lin.amax(dim=0, keepdim=True)                      # [1,H,W] 逐像素最亮通道
    # 方波 footprint：温度 highlight_softness 越小，空间边缘越接近 1px 的硬跳变。
    m = torch.sigmoid((detect - ctrl.highlight_thresh) / ctrl.highlight_softness)
    # 【逆色调映射 / highlight enhancement（BokehMe++ §III-C，D38；D42 修正阈值）】裁剪高光的真实
    # HDR 亮度已被 LDR 上限抹掉、无法从像素值反推；故按【饱和程度 sat】把平顶从 1 线性抬到 1+gain。
    # **D42 关键修正**：sat 用独立的 `highlight_hdr_thresh`（≈0.92）而非检测阈值（≈0.82）——只有
    # 真正被传感器裁剪到接近饱和 (d→1) 的点才丢失了 HDR 信息、才该恢复放大；d<hdr_thresh 的中等亮度
    # 灯珠并未过曝、真实亮度即原值，放大它=凭空造能量 → 散景球过曝死白（用户反馈：原图灯珠没过曝、
    # 输出却放大了能量）。分离后：中等灯珠 sat≈0、peak≈1（成柔和彩色盘、不放大）；真饱和核心 sat→1、
    # peak→1+gain（成明亮散景球）。检测掩膜 m 仍用 highlight_thresh（中等灯珠照样做方波盘、只是不放大）。
    hdr_t = max(ctrl.highlight_hdr_thresh, ctrl.highlight_thresh)  # 保证 ≥ 检测阈值
    sat = ((detect - hdr_t) / (1.0 - hdr_t + 1e-6)).clamp(0.0, 1.0)
    peak = 1.0 + ctrl.highlight_gain * sat                     # 逐像素平顶高度（仅真饱和→1+gain）
    if ctrl.highlight_per_region:
        # 逐高光区取 max(peak) → 均匀方波平顶（BokehMe++ Algorithm1 的区内均匀化）。
        rp = _per_region_peak(detect, peak, float(ctrl.highlight_thresh))
        if rp is not None:
            peak = rp                                          # [1,H,W] 逐区均匀平顶（detach）
    # 平顶按【饱和通道=peak】上色：chroma = lin/detect 把最亮通道归一到 1，× peak →
    # 饱和通道顶到 peak、其余通道按原比例缩放（保色相）。白光→(peak,peak,peak)，
    # 红光→(peak,~,~) 自然钳成饱和红，避免按亮度归一时彩色被过度放大。
    chroma = lin / (detect + 1e-6)
    flat = chroma * peak                                        # 均匀平顶（饱和通道顶到 peak）
    # 内部=平顶方波、外部=原样，由锐利掩膜 m 线性混合（M1 取 flat，M0 取 lin）。
    return lin * (1.0 - m) + flat * m


def _per_region_peak(detect, peak_itm, thresh: float):
    """逐高光区把逆色调映射峰值【区内均匀化】（D38，BokehMe++ Algorithm1 区内均匀平顶）。

    逆色调映射给的是逐像素 HDR 估计 `peak_itm`；同一个光斑内部若亮度有起伏，平顶会不平、
    散景盘内部花。这里把每个连通高光区的平顶统一取该区 max(peak_itm)，得到干净方波平顶
    （盘内均匀、边缘锐利）。对饱和点光本就均匀，主要让部分饱和的光斑也成均匀平顶。

    Args:
        detect: [1,H,W] 逐像素通道最大值。
        peak_itm: [1,H,W] 逐像素逆色调映射平顶高度。
        thresh: 高光硬阈值（定义连通核）。
    Returns:
        [1,H,W] 区内均匀化后的平顶（torch，与 detect 同 device/dtype）；
        无 scipy 或无高光区时返回 None（caller 回落逐像素 itm）。
    """
    import torch
    try:
        from scipy import ndimage
        import numpy as np
    except Exception:
        return None
    d = detect[0].detach().cpu().numpy().astype("float32")
    pk = peak_itm[0].detach().cpu().numpy().astype("float32")
    core = d > thresh
    labels, n = ndimage.label(core)
    if n == 0:
        return None
    idx = list(range(1, n + 1))
    # 每区取 max(peak_itm) 作均匀平顶高度。
    reg_max = np.asarray(ndimage.maximum(pk, labels, index=idx), dtype="float32")
    lut = np.concatenate([[0.0], reg_max]).astype("float32")    # 0=背景标签
    out = lut[labels]                                           # 核像素取区峰，背景 0
    peak_map = np.where(core, out, pk)                         # 核外保留逐像素 itm
    return torch.from_numpy(peak_map).to(detect)[None]


def unify_highlight_core_depth(image, disparity, thresh: float = 0.92):
    """高光核心连通区的视差【统一为核心最亮像素的深度】——消除单灯珠嵌套环（D43）。

    【问题】真实图里一个过曝小灯珠跨多个像素，深度图（DA V2）对这些小亮点估计不可靠，
    同一灯珠的像素常被判成不同视差（实测灯串边界局部 std 高达 0.36 → CoC 差 28px）。
    渲染时这些像素被 tent 分层分到【不同 CoC 的层】→ 同一灯珠渲染成大小不一的多个盘叠加
    → 用户反馈的"光斑外围套一圈淡光斑 / 嵌套同心圆"。注意这【不是 PSF/渲染器 bug】
    （geom PSF 单盘剖面完全均匀、孤立单深度灯珠渲染干净），纯粹是深度噪声经分层放大。

    【解法】把每个【真饱和高光核心】(detect>thresh) 连通区的所有像素视差，统一为该区
    【最亮像素】(灯珠中心，最可靠的单一深度样本) 的视差。于是单灯珠 → 单一深度 → 单盘。
    只动真饱和核心（thresh≈0.92，少量像素）→ 副作用极小（实测灯串区视差均值前后不变）。

    【为何用最亮像素而非中位数】中位数受灯珠边缘混入的背景像素污染、且可能恰落在焦带
    （实测把灯珠推成合焦清晰）；最亮像素=灯珠核心，深度最具代表性。

    【不可微但无妨】深度是网络【输入】、不在像差系数的梯度路径上 → 用 numpy/scipy 安全。
    仅用于 demo/e2e 的【真实图】（合成训练的 GT 深度精确、不需要本处理）。

    Args:
        image: [3,H,W] sRGB 全焦图（用通道最大值检测高光核心）。
        disparity: [H,W] 视差图。
        thresh: 高光核心检测阈值（线性亮度，默认 0.92 只取真饱和）。
    Returns:
        (统一后的视差 [H,W], 处理的核心区数 n)；无 scipy 或无核心时原样返回。
    """
    import torch
    try:
        from scipy import ndimage
    except Exception:
        return disparity, 0
    detect = image.amax(dim=0).detach().cpu().numpy()
    d = disparity.detach().cpu().numpy().copy()
    core = detect > thresh
    labels, n = ndimage.label(core)
    if n == 0:
        return disparity, 0
    # 每个连通核心区取【最亮像素】位置，用其视差统一整区。
    pos = ndimage.maximum_position(detect, labels, index=list(range(1, n + 1)))
    disp_np = disparity.detach().cpu().numpy()
    for i, (py, px) in enumerate(pos, 1):
        d[labels == i] = disp_np[py, px]
    return torch.from_numpy(d).to(disparity), n


def srgb_to_linear(image, ctrl: RenderControl):
    """sRGB→线性 + 可选高光权重重分配(WR)。物理上卷积发生在线性光强域；
    WR 把过曝点源重塑成"平顶方波"（真实亮度远超 1.0 + 锐利边缘）——这正是
    锐利亮斑散景/glow 的能量来源，也是 BokehMe++ Fig 3 强调的焦外风格化要点。"""
    import torch
    lin = image.clamp(min=0.0) ** ctrl.gamma
    if ctrl.highlight_gain > 0.0:
        lin = redistribute_highlights(lin, ctrl)
    return lin


def _soft_shoulder(lin, knee: float):
    """HDR 软肩压缩（D41）：把 [0,∞) 的线性光强映射进 [0,1]，但只动【高光】。

    曲线分两段（C1 连续、处处可微 → 不破坏指纹标定/训练梯度）：
      · L ≤ knee：恒等 f(L)=L —— 正常曝光/主体/中间调【完全不动】。
      · L > knee：指数软肩 f(L)=knee+(1−knee)·(1−exp(−(L−knee)/(1−knee)))
                  —— 单调升、渐近逼近 1（永不超过），拐点处导数=1 与恒等段无缝衔接。

    【为何这样修】散景盘是加性 HDR 光：N 个盘重叠处线性能量相加可远超 1（实测密集
    灯串 40% 像素 >1）。直接 clamp/保存会把这些区域硬裁成【平白块】——同一片纯白、
    丢失盘边界与红蓝叠加层次（用户反馈的"亮度溢出"）。软肩让 1.0~1.4 的重叠区映射到
    0.93~0.99 而非全 1：① 重叠核之间仍有亮度梯度 → 盘边/层次可见；② 极亮处仍逼近白
    （真实饱和高光本就接近白）。这正是游戏/CG 散景与真实相机的标准做法：HDR 加性合成
    + 末端色调曲线滚降，而非硬裁（见 D41 文献：MJP/Wronski/多篇 DoF 专利）。
    """
    import torch
    over = (lin - knee).clamp(min=0.0)                     # 仅超过 knee 的部分参与压缩
    rolled = knee + (1.0 - knee) * (1.0 - torch.exp(-over / (1.0 - knee + 1e-6)))
    return torch.where(lin > knee, rolled, lin)            # knee 以下原样、以上软肩


def linear_to_srgb(lin, ctrl: RenderControl):
    """线性→sRGB：先做 HDR 软肩色调映射（仅高光开启时），再 gamma 编码。

    色调映射放在 gamma 【之前】（线性/scene-referred 域）符合标准成像管线：
    HDR 加性合成 → tone map → 显示域 → OETF(gamma)。软肩只在有 HDR 高光
    (highlight_gain>0) 时启用：无高光时内容本就 ≤1、不会溢出，跳过以保持
    既有渲染分布 byte 级不变（不扰动无高光的训练样本）。
    """
    lin = lin.clamp(min=0.0)
    if ctrl.highlight_tonemap and ctrl.highlight_gain > 0.0:
        lin = _soft_shoulder(lin, ctrl.tonemap_knee)
    return lin ** (1.0 / ctrl.gamma)


# ==============================================================================
# 8) 统一入口
# ==============================================================================
def render(image, disparity, ctrl: RenderControl, field_varying: bool = True,
           **tile_kwargs):
    """主入口：field_varying=True 走 tile 视场相关路径（完整效果），
    False 走全局 PSF 快速路径（H=0，视场相关效应消失，调试/对照用）。"""
    if field_varying:
        return render_tiled(image, disparity, ctrl, **tile_kwargs)
    return render_global(image, disparity, ctrl)


def render_field_patch(image, disparity, ctrl: RenderControl,
                       H_field: float = 0.0, azimuth: float = 0.0,
                       H_centers=None, H_weights=None):
    """单 PSF 视场近似渲染：把整张（小）图当作位于画幅 (H_field, azimuth) 处。

    用途：训练时【网络输入侧】的物理渲染。与 data/synth.py 生成 GT 用的
    composite_blurred_layers 共享同一套视场近似（单个旋转后的 PSF 渲染整块，
    与 tile 近似 D8 物理一致），但出发点不同：
        GT   ← 已知 (内容, alpha, 视差) 图层，精确合成；
        这里 ← 单图 + 估计/扰动视差，tent 软分层（与真实推理同路径）。
    两者之差（深度误差导致的边界瑕疵）正是细化网要学习修复的内容。

    【H 子带模式】与 composite_blurred_layers 的子带参数完全同口径——
    GT 用了子带，输入侧渲染必须用【同一组】(H_centers, H_weights)，否则
    输入/GT 之差会混入视场近似不一致项，细化网会把它当"瑕疵"全图乱修。

    Args:
        image: [3,H,W] sRGB [0,1]；disparity: [H,W] 视差 [0,1]（近=1）。
        H_field / azimuth: 该裁块在画幅上的归一化像高与方位角（与样本 ctrl_vec 一致）。
        H_centers / H_weights: 可选子带（见 composite_blurred_layers）。
    Returns:
        [3,H,W] sRGB 散景图（已乘 T(H) 边角失光，与 GT 口径一致）。
    """
    import torch
    from optics import pupil as pupil_mod
    slope, intercept = load_defocus_calibration()
    grid = pupil_mod.make_pupil_grid(ctrl.pupil_size, device=str(image.device),
                                     dtype=image.dtype)
    img_lin = srgb_to_linear(image, ctrl)
    weights, centers = layer_weights(disparity, ctrl.effective_n_layers(),
                                     ctrl.focus_disparity, ctrl.focus_tolerance)

    def _render_at(H_val: float):
        out_lin = _composite_layers(img_lin, weights, centers, ctrl, grid,
                                    slope, intercept, H_img=H_val, azimuth=azimuth)
        # GT 侧（composite_blurred_layers）乘了 T(H) 边角失光，这里同样要乘，
        # 否则输入/GT 亮度口径不一致会被细化网误学成"全局调亮"。
        T = pupil_mod.relative_transmission(grid, H_val, ctrl.coeffs)
        return out_lin * T

    if H_centers is None or len(H_centers) == 1:
        out_lin = _render_at(float(H_centers[0]) if H_centers else H_field)
    else:
        out_lin = torch.zeros_like(img_lin)
        for b, h in enumerate(H_centers):
            out_lin = out_lin + H_weights[b:b + 1] * _render_at(float(h))
    return linear_to_srgb(out_lin, ctrl)
