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
    highlight_thresh: float = 0.9    # 高光判定阈值（线性亮度）

    def __post_init__(self):
        if self.coeffs is None:
            from optics.aberrations import AberrationCoeffs
            self.coeffs = AberrationCoeffs()


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


# ==============================================================================
# 3) 视差分层（tent 软权重）
# ==============================================================================
def layer_weights(disparity, n_layers: int):
    """把 [H,W] 视差转成 [L,H,W] 的层权重（线性 tent，相邻两层间平滑过渡）。

    为什么不用硬分桶：硬边界会在深度渐变面上产生"分层断带"伪影；
    tent 权重让每个像素按距离线性分给相邻两层，Σ_l w_l = 1，且实现简单。
    层中心固定取 [0,1] 均匀网格（不依赖控制向量 → 不挡 c 的梯度）。

    Returns:
        weights: [L,H,W]，非负，逐像素和为 1。
        centers: [L] 各层代表视差。
    """
    import torch
    centers = torch.linspace(0.0, 1.0, n_layers,
                             device=disparity.device, dtype=disparity.dtype)
    delta = centers[1] - centers[0]
    d = disparity[None]                                      # [1,H,W]
    w = (1.0 - (d - centers[:, None, None]).abs() / delta).clamp(min=0.0)
    return w / (w.sum(dim=0, keepdim=True) + 1e-12), centers


# ==============================================================================
# 4) 每层 PSF（现场算，D9）
# ==============================================================================
def _layer_psf(grid, H_img: float, coeffs, w020_layer: float, pupil_size: int,
               check: bool = False, balance_spherical: bool = False):
    """给定层离焦 w020 与像高 H，现场算一组 RGB PSF（[3,k,k]，k 自适应离焦大小）。

    crop 自适应：盘半径 ≈ slope·|W020|，crop 取 2·半径+余量（偶数，中心=crop//2），
    小离焦层用小核 → FFT 卷积更快。
    balance_spherical：叠加球差焦移补偿 −W040（见 RenderControl.balance_spherical_focus）。
    """
    import torch
    from optics import psf as psf_mod
    w020_eff = w020_layer
    if balance_spherical:
        w020_eff = w020_eff - coeffs.W040_spherical
    # crop 是离散的核尺寸选择，不需要（也不能）带梯度 → 用 detach 后的数值估算。
    w_abs = float(torch.as_tensor(w020_eff).detach().abs())
    radius = 4.0 * w_abs + 4.0                          # 理论半径 + 安全余量(衍射/像差外扩)
    crop = int(min(pupil_size, 2 * math.ceil(radius) + 8))
    crop += crop % 2                                    # 取偶，使 PSF 中心恰在 crop//2
    c_l = coeffs.replace(W020_defocus=coeffs.W020_defocus + w020_eff)
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
                         balance_spherical=ctrl.balance_spherical_focus)
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
    weights, centers = layer_weights(disparity, ctrl.n_layers)
    out_lin = _composite_layers(img_lin, weights, centers, ctrl, grid, slope, intercept)
    return linear_to_srgb(out_lin, ctrl)


# ==============================================================================
# 6) 主渲染：tile 视场相关模式（D8 —— 彗差/像散/猫眼/旋焦/LaCA 在此显现）
# ==============================================================================
def render_tiled(image, disparity, ctrl: RenderControl, tile: int = 64,
                 H_bins: int = 6):
    """tile 近似的视场相关渲染：每 tile 用其中心 (H, 方位角) 的 PSF。

    实现要点：
    - 每 tile 取带 halo（=最大核半径）的邻域做分层卷积，再裁回 tile —— 保证
      tile 边界处的模糊不缺邻域信息，拼接无缝。
    - PSF 按 (层, H 桶) 缓存（+x 约定），每 tile 只做一次旋转 → 计算量可控。
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
    weights, centers = layer_weights(disparity, ctrl.n_layers)

    # halo = 全图可能出现的最大核半径（用最大 |CoC| 估计）+ 余量。
    # halo 是离散的窗口尺寸，不需要梯度 → detach 成 float 再取整（d_f/K 可能是带梯度张量）。
    r0 = float(torch.as_tensor(signed_coc(0.0, ctrl.focus_disparity, ctrl.aperture_K,
                                          ctrl.focus_tolerance)).detach())
    r1 = float(torch.as_tensor(signed_coc(1.0, ctrl.focus_disparity, ctrl.aperture_K,
                                          ctrl.focus_tolerance)).detach())
    halo = int(math.ceil(max(abs(r0), abs(r1)))) + 8

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
                                        balance_spherical=ctrl.balance_spherical_focus)
        return psf_cache[key]

    out = torch.zeros_like(img_lin)
    for ty in range(0, Hh, tile):
        for tx in range(0, Ww, tile):
            y1, x1 = min(ty + tile, Hh), min(tx + tile, Ww)
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

            # 该 tile 的分层合成：PSF 取自缓存 + 旋转到 tile 方位角。
            sub = _composite_tile(sub_img, sub_w, centers, ctrl, get_psf, hb, azim)
            # 从带 halo 的结果里裁回本 tile 的窗口。
            out[:, ty:y1, tx:x1] = sub[:, ty - hy0:y1 - hy0, tx - hx0:x1 - hx0]

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
    """tile 版分层合成：与 _composite_layers 同逻辑，但 PSF 走缓存+旋转。"""
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
    return out / acc.clamp(min=1e-6)


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
                                        balance_spherical=ctrl.balance_spherical_focus),
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
def srgb_to_linear(image, ctrl: RenderControl):
    """sRGB→线性 + 可选高光提升。物理上卷积发生在线性光强域；
    高光提升模拟"过曝点源的真实亮度远超 1.0"——这正是亮斑散景/glow 的能量来源。"""
    import torch
    lin = image.clamp(min=0.0) ** ctrl.gamma
    if ctrl.highlight_gain > 0.0:
        lum = lin.mean(dim=0, keepdim=True)
        w = torch.sigmoid((lum - ctrl.highlight_thresh) / 0.02)   # 软阈值，可微
        lin = lin * (1.0 + ctrl.highlight_gain * w)
    return lin


def linear_to_srgb(lin, ctrl: RenderControl):
    """线性→sRGB（卷积后过曝部分自然软剪裁回 [0,1]）。"""
    return lin.clamp(min=0.0) ** (1.0 / ctrl.gamma)


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
    weights, centers = layer_weights(disparity, ctrl.n_layers)

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
