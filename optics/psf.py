"""
optics/psf.py
=============
PSF 生成与「PSF 字典/查找表」—— 项目技术心脏的后半。

核心关系（CLAUDE.md 第 3 节）：
    PSF(H, 离焦) = | FFT2D{ P(ρ,θ; H) } |²
即对复光瞳做二维傅里叶变换、取模平方，得到该像高、该离焦量下的点扩散函数。

逐通道色差（默认启用，成本仅 3× PSF）：
    R/G/B 各算一次。LoCA 来自各通道 W020 偏移（在 pupil.wavefront 里注入）；
    LaCA 来自各通道 PSF 随像高的【径向平移】（本文件 _radial_shift 实现，沿 +x 约定，
    与猫眼一致，渲染时随方位角一起旋转）。

================================================================================
【PSF 字典的定位：推理缓存，而非训练主路径】（见 DECISIONS D9）
PSF 只随三个量变化：(像高分桶 H, 带符号离焦量, 像差向量 a)。
- 【训练/标定时】渲染器按视差分层，每层离焦量是确定的 → 直接对这些确定值
  现场算 PSF（每前向约 层数×H桶×3 个 FFT，GPU 上毫秒级），完全绕过插值，
  且对 a 的梯度链路最直接。
- 【推理时】镜头系数 a 固定 → 可用 build_psf_dictionary 预计算缓存加速。
- 【插值警告】两个不同尺寸的环状 PSF（如肥皂泡亮边）线性混合会产生"双环鬼影"
  而非中间尺寸的环——sample_psf 的插值仅适合离焦轴足够密、或盘内结构平缓的场合。

【可微性】 |FFT{P}|² 对所有系数可微 → 同一套代码既支撑端到端训练，
也支撑镜头指纹标定（对 a 梯度下降匹配目标镜头样张）。

【离焦如何驱动 PSF 大小】
带符号离焦量直接作为波前的 W020(defocus) 系数（单位 waves）注入：
    |W020| 越大，二次相位越陡，|FFT|² 越铺开 → 散景盘越大；
    W020 的【符号】区分焦前/焦后 → 配合球差符号即可涌现"前/背景形态反转"。
"""

from __future__ import annotations

from . import pupil as _pupil


# ------------------------------------------------------------------------------
# 1) 单个复光瞳 → 单个 PSF
# ------------------------------------------------------------------------------
def pupil_to_psf(P, crop: int | None = None, check_sampling: bool = True):
    """ |FFT2D{P}|² 并做能量归一化（sum=1）。对 P（及其依赖的系数）可微。

    步骤：
      1. fft2(P)：到像面振幅分布（注意 fft2 默认把零频放在角上）。
      2. fftshift：把零频(PSF 中心)搬到图像正中，便于裁剪与渲染对齐。
      3. abs()**2：取模平方 → 光强 PSF（非负实数）。
      4. 【guard】混叠与裁剪检查（见下）。
      5. 可选中心裁剪到 crop×crop：散景盘通常只占中心一小块，裁剪省显存。
      6. 归一化：除以总和，保证渲染时亮度守恒（PSF 是"能量分配核"）。

    【为什么需要 guard】相位采样要求相邻格点相位差 < π。纯离焦下这等价于
        W020 < N/8 (waves)，N=光瞳网格边长（N=256 → 上限约 32 waves，叠加 W040 更紧）。
    超限后 PSF 会绕回（wrap-around）堆到边界，而"裁剪+重归一化"会把这个错误
    悄悄掩盖掉。因此在裁剪【之前】检查全图边界带能量；裁剪时再检查被裁掉的能量。

    Args:
        P: [N,N] complex 张量（来自 pupil.complex_pupil）。
        crop: 若给定，裁出中心 crop×crop 区域。None=保留 N×N。
        check_sampling: 是否做混叠/裁剪检查（默认开；字典批量生成时可关以免重复告警）。

    Returns:
        psf: [crop,crop] 或 [N,N] 实张量，非负，sum=1。
    """
    import torch
    field = torch.fft.fftshift(torch.fft.fft2(P))   # 像面复振幅，中心化
    psf = (field.real ** 2 + field.imag ** 2)       # |·|² = 光强（等价 abs()**2，但更省一次开方）

    total = psf.sum() + 1e-12

    if check_sampling:
        import warnings
        N = psf.shape[-1]
        # 混叠检查：全图最外 4 像素边界带的能量占比。正常 PSF 能量集中在中心，
        # 边界带应≈0；若显著非零，说明相位欠采样导致 wrap-around。
        b = 4
        inner = psf[b:N - b, b:N - b].sum()
        # detach：guard 只做诊断，不应进入梯度图（也避免对 requires_grad 张量取 float 的告警）。
        border_frac = float(((psf.sum() - inner) / total).detach())
        if border_frac > 1e-3:
            warnings.warn(
                "[optics] PSF 边界带能量异常（疑似相位欠采样混叠）。"
                "请减小 |W020|（经验上限 ≈ N/8 waves）或增大 pupil_size。",
                RuntimeWarning,
            )

    if crop is not None and crop < psf.shape[-1]:
        N = psf.shape[-1]
        c0 = (N - crop) // 2
        cropped = psf[c0:c0 + crop, c0:c0 + crop]
        if check_sampling:
            import warnings
            clipped_frac = float(((total - cropped.sum()) / total).detach())
            if clipped_frac > 1e-2:
                warnings.warn(
                    "[optics] PSF 裁剪丢失能量 >1%（散景盘大于 crop 窗口）。"
                    "请增大 crop 或减小离焦量，否则重归一化会掩盖形状截断。",
                    RuntimeWarning,
                )
        psf = cropped

    # 能量归一化（加极小量防除零）。
    psf = psf / (psf.sum() + 1e-12)
    return psf


# ------------------------------------------------------------------------------
# 2) 逐通道（RGB）PSF：色差
# ------------------------------------------------------------------------------
# 光谱平均的子波长相位缩放因子与权重（±6% 带宽近似相机色滤片的有效宽度）。
# 设为 ((1.0, 1.0),) 可退回单色行为（调试/对照用）。
SPECTRAL_SAMPLES: tuple = ((0.94, 0.25), (1.0, 0.5), (1.06, 0.25))


def rgb_psf(grid: dict, H: float, coeffs, crop: int | None = None,
            softness: float = 0.02, check_sampling: bool = True,
            spectral: bool = True):
    """生成一组 R/G/B 三通道 PSF（堆叠成 [3,h,w]），实现 LoCA + LaCA 色差。

    - LoCA：在 pupil.complex_pupil(channel=...) 里给各通道 W020 叠加 loca_rgb 偏移。
    - LaCA：对每个通道的 PSF 沿 +x【径向平移】laca_rgb[c]·H 像素（可微 grid_sample）。
      物理依据：横向色差 = 各通道放大率不同 → 离轴处各通道 PSF 中心彼此错开，
      平移量 ∝ 像高 H。沿 +x 是与猫眼相同的字典约定（渲染时随方位角一起旋转）。
      H=0（画面中心）时平移为 0，LaCA 自动消失——符合物理。
    - 【光谱平均（spectral=True，默认）】单色相干仿真的离焦 PSF 盘内有高对比
      菲涅耳干涉环（"能量分布不均"），但真实相机是宽光谱+部分相干+传感器积分，
      散景盘接近均匀。对每通道在 ±6% 带宽内取 3 个子波长（相位 ∝ 1/λ →
      phase_scale 缩放）加权平均：环纹位置随波长移动、彼此抵消，盘面变均匀，
      而盘的尺寸/形状/边缘特征（亮边/猫眼/多边形）保留。成本 3×FFT。

    Returns:
        [3, h, w] 实张量，每个通道已各自能量归一。
    """
    import torch
    samples = SPECTRAL_SAMPLES if spectral else ((1.0, 1.0),)
    chans = []
    for c in range(3):
        acc = None
        for s, wgt in samples:
            P = _pupil.complex_pupil(grid, H, coeffs, channel=c,
                                     softness=softness, phase_scale=s)
            p = pupil_to_psf(P, crop=crop,
                             check_sampling=(check_sampling and s == 1.0))
            acc = p * wgt if acc is None else acc + p * wgt
        psf_c = acc / (acc.sum() + 1e-12)         # 加权平均后重新归一
        shift_px = coeffs.laca_rgb[c] * float(H)
        if abs(shift_px) > 1e-6:
            psf_c = _radial_shift(psf_c, shift_px)
        chans.append(psf_c)
    return torch.stack(chans, dim=0)


def _radial_shift(psf, shift_px: float):
    """把 PSF 沿 +x 平移 shift_px 像素（可微，affine_grid + grid_sample，亚像素精度）。

    用于 LaCA：各通道 PSF 中心的径向错位。平移后重新归一化能量
    （边缘移出的能量极小，因 LaCA 平移仅 1~2px 量级）。
    """
    import torch
    import torch.nn.functional as F
    h, w = psf.shape[-2:]
    inp = psf[None, None]                       # [1,1,h,w]
    # affine_grid 的归一化坐标横跨 [-1,1] 对应宽度 w → shift_px 像素 = 2·shift_px/w。
    # output(x) = input(x + tx)，要让图像内容向 +x 移动需采样 x − s，故 tx = −2s/w。
    tx = -2.0 * shift_px / w
    theta = torch.tensor([[[1.0, 0.0, tx],
                           [0.0, 1.0, 0.0]]], dtype=psf.dtype, device=psf.device)
    flow = F.affine_grid(theta, size=(1, 1, h, w), align_corners=False)
    out = F.grid_sample(inp, flow, mode="bilinear", padding_mode="zeros",
                        align_corners=False)[0, 0]
    return out / (out.sum() + 1e-12)


# ------------------------------------------------------------------------------
# 3) PSF 字典 / 查找表（核心效率手段）
# ------------------------------------------------------------------------------
def build_psf_dictionary(coeffs, H_bins, defocus_bins, pupil_size: int,
                         crop: int, device: str = "cpu", chromatic: bool = True,
                         softness: float = 0.02):
    """离线预计算 PSF 查找表（【推理缓存】用途，见模块头与 DECISIONS D9；
    训练/标定时建议按层离焦值现场算 PSF，绕过插值）。

    Args:
        coeffs: 一组固定的像差系数 a（一支"虚拟镜头"）。
        H_bins: 像高采样值列表，例如 [0.0, 0.33, 0.66, 1.0]。
        defocus_bins: 带符号离焦量(W020, waves)列表，含正负，例如 linspace(-20,20,k)。
        pupil_size: 光瞳/FFT 网格分辨率（如 256）。
        crop: PSF 裁剪尺寸（如 96），散景盘只占中心，裁剪省显存。
        device: 'cpu'/'cuda'。生成可在 cpu 完成。
        chromatic: True → 每个 PSF 含 RGB 三通道（色差）；False → 单通道(用 G 近似)。
        softness: 遮罩软化尺度。

    Returns:
        psf_lut: 张量。
            chromatic=True  → [len(H_bins), len(defocus_bins), 3, crop, crop]
            chromatic=False → [len(H_bins), len(defocus_bins), 1, crop, crop]
        axes: (H_bins, defocus_bins) 便于后续插值定位。
        transmission: [len(H_bins)] 张量——各像高桶的相对透过率 T(H)（口径蚀边角失光，
            见 pupil.relative_transmission）。渲染合成时乘回：亮度 = T(H)·(PSF⊛图像)。

    注意显存：字典大小 = H×D×C×crop×crop×4 字节。
        例 4×16×3×96×96×4B ≈ 68MB，完全可控；增大任一维前先估算。
    """
    import torch
    grid = _pupil.make_pupil_grid(pupil_size, device=device)

    H_list = list(H_bins)
    D_list = list(defocus_bins)
    # 混叠/裁剪 guard 只在最极端离焦档检查一次（其余档位必然更安全），避免循环里重复告警。
    d_extreme = max(D_list, key=abs) if D_list else 0.0
    entries = []
    for H in H_list:
        row = []
        for d in D_list:
            check = (d == d_extreme)
            # 把当前离焦量写入 W020：用 replace 生成临时系数，不污染原 coeffs。
            c_d = coeffs.replace(W020_defocus=coeffs.W020_defocus + float(d))
            if chromatic:
                psf = rgb_psf(grid, H, c_d, crop=crop, softness=softness,
                              check_sampling=check)                             # [3,crop,crop]
            else:
                P = _pupil.complex_pupil(grid, H, c_d, channel=1, softness=softness)  # 用 G(ch=1)
                psf = pupil_to_psf(P, crop=crop, check_sampling=check)[None]    # [1,crop,crop]
            row.append(psf)
        entries.append(torch.stack(row, dim=0))         # [D,C,crop,crop]
    psf_lut = torch.stack(entries, dim=0)               # [H,D,C,crop,crop]

    # 相对透过率 T(H)：口径蚀的"边角失光"分量（PSF 逐个归一化丢掉的那一半物理）。
    transmission = torch.stack(
        [_pupil.relative_transmission(grid, H, coeffs, softness) for H in H_list]
    )                                                    # [H]
    return psf_lut, (H_list, D_list), transmission


# ------------------------------------------------------------------------------
# 4) 从字典插值取用（渲染时）
# ------------------------------------------------------------------------------
def sample_psf(psf_lut, axes, H: float, defocus: float):
    """从 PSF 字典按 (H, 离焦) 做双线性插值取出一个 PSF。对查表插值保持可微。

    【注意】仅适合推理缓存场景且离焦轴足够密：两个不同尺寸的环状 PSF 线性混合
    会产生"双环鬼影"而非中间尺寸的环（见模块头）。训练时请按层现场算 PSF。

    Args:
        psf_lut: build_psf_dictionary 产出的 [H,D,C,h,w] 张量。
        axes: (H_list, D_list) —— 两轴的采样坐标。
        H: 查询像高。
        defocus: 查询离焦量(带符号)。

    Returns:
        [C,h,w] PSF（在两轴上线性插值）。
    """
    import torch

    H_list, D_list = axes

    def _bracket(vals, q):
        # 找到 q 落在的相邻两格 (i0,i1) 及权重 w（线性插值）。
        n = len(vals)
        if q <= vals[0]:
            return 0, 0, 0.0
        if q >= vals[-1]:
            return n - 1, n - 1, 0.0
        for i in range(n - 1):
            if vals[i] <= q <= vals[i + 1]:
                w = (q - vals[i]) / (vals[i + 1] - vals[i] + 1e-12)
                return i, i + 1, float(w)
        return n - 1, n - 1, 0.0

    h0, h1, wh = _bracket(H_list, H)
    d0, d1, wd = _bracket(D_list, defocus)

    # 双线性：先在离焦轴插值，再在像高轴插值。
    def lerp(a, b, t):
        return a * (1.0 - t) + b * t

    p00 = psf_lut[h0, d0]; p01 = psf_lut[h0, d1]
    p10 = psf_lut[h1, d0]; p11 = psf_lut[h1, d1]
    top = lerp(p00, p01, wd)
    bot = lerp(p10, p11, wd)
    psf = lerp(top, bot, wh)
    # 重新归一化各通道能量（插值后可能略偏）。
    psf = psf / (psf.flatten(-2).sum(-1)[..., None, None] + 1e-12)
    return psf
