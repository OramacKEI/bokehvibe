"""
optics/pupil.py
===============
复光瞳函数 P(ρ,θ; H) 的构造 —— 项目技术心脏的前半（后半是 psf.py 的 FFT）。

物理回顾（对照 CLAUDE.md 第 3 节）：
    光瞳坐标 (ρ, θ)，ρ∈[0,1] 归一化光瞳半径，θ 角度。
    H = 归一化像高（像点到画面中心距离），决定彗差/像散/场曲/猫眼的强度。

复光瞳 = 「振幅遮罩」 × 「相位（波前像差）」：

    P(ρ,θ; H) = A(ρ,θ; H) · exp( i · 2π · W(ρ,θ; H) )

    ↑ 注意：W 以「波长倍数(waves)」为单位，故相位 = 2π·W（无需再除以 λ；
      CLAUDE.md 写的 (2π/λ)·W 是当 W 取长度单位时的形式，二者等价）。

    波前像差（相位，单位 waves）：
        W = W020·ρ²  +  W040·ρ⁴  +  W131·H·ρ³·cosθ  +  W222·H²·ρ²·cos²θ  +  W220·H²·ρ²

    光瞳振幅（遮罩，无量纲 [0,1]）：
        A = M_aperture(ρ,θ)          n 边形/圆形通光         → 多边形散景
          · M_vignette(ρ,θ; H)       随像高平移的猫眼截断     → 旋焦 / 单侧亮边
    （洋葱圈周期扰动项已移出研究范围，见 DECISIONS D15）

================================================================================
【可微性（硬要求，CLAUDE.md 第 12 节）】
全程使用 torch 算子。需要"硬边界"的地方（光圈/口径蚀的通光与否）一律用
**sigmoid 软化**代替硬阈值，使遮罩对几何参数可微，也让 PSF 边缘更接近真实光学（有过渡）。

【约定：猫眼沿 +x 轴】
猫眼(口径蚀)本应指向像点的径向方向。为让 PSF 字典只需按"像高 H"一维分桶（而非二维含方位角），
我们【固定让口径蚀沿 +x 方向平移】来计算 PSF；真正渲染时，由 render/ 根据每个像素的方位角
把 PSF 旋转到正确朝向即可。这样把方位角自由度从字典里解耦出去，省显存。

【设备/精度】
所有函数接收一个已建好的坐标网格（make_pupil_grid 产出），其 device/dtype 决定后续计算位置；
因此在 CPU 上即可完成全部物理验证（FFT 也支持 CPU），不依赖 GPU。
"""

from __future__ import annotations


# ------------------------------------------------------------------------------
# 1) 光瞳坐标网格
# ------------------------------------------------------------------------------
def make_pupil_grid(size: int, device: str = "cpu", dtype=None):
    """生成归一化光瞳坐标网格 (ρ, θ) 以及笛卡尔 (x, y)。

    我们在一个边长为 `size` 的正方形上铺 [-1,1]×[-1,1] 网格，内切圆 ρ≤1 即为光瞳。
    （正方形四角 ρ>1，会被 M_aperture 遮掉。）

    Args:
        size: 网格边长（也是后续 FFT 的尺寸，例如 256）。
        device: 'cpu' / 'cuda'。物理验证用 cpu 即可。
        dtype: torch 浮点类型，默认 float32。

    Returns:
        dict，含键 'rho','theta','x','y'，每个是 [size,size] 张量。
        ρ∈[0,√2]（圆外>1），θ∈(-π,π]。
    """
    import torch
    if dtype is None:
        dtype = torch.float32

    # linspace(-1,1,size)：把光瞳直径映射到 [-1,1]。
    lin = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    # meshgrid(indexing='xy')：x 沿列变化、y 沿行变化，得到图像坐标系下的网格。
    x, y = torch.meshgrid(lin, lin, indexing="xy")
    rho = torch.sqrt(x * x + y * y)
    theta = torch.atan2(y, x)
    return {"rho": rho, "theta": theta, "x": x, "y": y}


# ------------------------------------------------------------------------------
# 2) 波前像差 W(ρ,θ; H)
# ------------------------------------------------------------------------------
def wavefront(grid: dict, H: float, coeffs, channel: int | None = None):
    """计算波前像差 W(ρ,θ; H)，单位 waves。对 coeffs 可微。

    Args:
        grid: make_pupil_grid 的输出。
        H: 归一化像高（标量）。中心 H=0 时彗差/像散/场曲项自动消失。
        coeffs: AberrationCoeffs。
        channel: 若指定 0/1/2(R/G/B)，则把该通道的 LoCA 偏移叠加到 W020（纵向色差）。

    Returns:
        W: [size,size] 张量（waves）。
    """
    import torch
    rho = grid["rho"]
    theta = grid["theta"]
    cos_t = torch.cos(theta)

    rho2 = rho * rho
    rho3 = rho2 * rho
    rho4 = rho2 * rho2

    # 离焦基底 + 该通道的 LoCA 偏移（纵向色差：各通道 W020 略不同 → 焦前偏紫/焦后偏绿涌现）
    W020 = coeffs.W020_defocus
    if channel is not None:
        W020 = W020 + coeffs.loca_rgb[channel]

    H = float(H)
    W = (
        W020 * rho2
        + coeffs.W040_spherical * rho4
        + coeffs.W131_coma * H * rho3 * cos_t
        + coeffs.W222_astigmatism * (H ** 2) * rho2 * (cos_t ** 2)
        + coeffs.W220_field_curv * (H ** 2) * rho2
    )
    return W


# ------------------------------------------------------------------------------
# 3) 光瞳振幅 A(ρ,θ; H) 的各遮罩分量
# ------------------------------------------------------------------------------
def _soft_step(value, softness: float):
    """可微软阶跃：sigmoid(value/softness)。value>0 → ~1（通光），value<0 → ~0（遮挡）。

    softness 越小越接近硬边界；太小会让梯度消失、PSF 边缘有振铃。取像素尺度量级即可。
    """
    import torch
    return torch.sigmoid(value / softness)


def aperture_mask(grid: dict, coeffs, softness: float = 0.02):
    """光圈通光遮罩 M_aperture：圆形(n_blades=0) 或 正 n 边形 → 多边形散景。

    正 n 边形（外接半径=1，绕中心旋转 blade_rotation）：
        在极坐标下，多边形边界半径 r_poly(θ) = cos(π/n) / cos( wrap(θ) )，
        其中 wrap(θ) 把角度折叠进单个扇区 [-π/n, π/n]。
        通光条件 ρ ≤ r_poly(θ)；用 _soft_step(r_poly - ρ) 软化。
    """
    import torch
    rho = grid["rho"]

    if coeffs.n_blades == 0:
        # 圆形光圈：ρ ≤ 1。
        return _soft_step(1.0 - rho, softness)

    n = int(coeffs.n_blades)
    theta = grid["theta"] - coeffs.blade_rotation
    sector = 2.0 * torch.pi / n
    # 把 θ 折叠到单个扇区中心：wrap ∈ [-sector/2, sector/2]
    wrapped = torch.remainder(theta, sector) - sector / 2.0
    # 多边形边界半径（外接圆半径取 1）：内切半径 cos(π/n)，沿扇区按 1/cos 张开到顶点。
    import math
    r_poly = math.cos(math.pi / n) / torch.cos(wrapped).clamp(min=1e-3)
    return _soft_step(r_poly - rho, softness)


def vignette_mask(grid: dict, H: float, coeffs, softness: float = 0.02):
    """口径蚀/猫眼遮罩 M_vignette：随像高 H 平移的第二孔径，与主光瞳取交 → 猫眼。

    物理直觉（简化的"双圆相交"模型，与 BokehMe++ K0/K,z_l 的几何同源）：
        机械渐晕相当于在光瞳前/后再加一个孔径；从离轴像点看去，这个孔径相对主光瞳
        发生横向平移，平移量随像高线性增大。两孔径的交集把圆形光瞳切成"猫眼/柠檬"形。

    实现：第二孔径是半径 R_v、圆心在 (s, 0) 的圆（约定沿 +x，见模块说明）：
        s = vignette_strength · H
        通光附加条件：(x - s)² + y² ≤ R_v²
    H=0（画面中心）时 s=0，第二孔径与主光瞳同心，不产生截断。
    """
    import torch
    if coeffs.vignette_strength == 0.0:
        # 无口径蚀：返回全 1（不截断）。用 ones_like 保持张量/设备一致。
        return torch.ones_like(grid["rho"])

    x = grid["x"]
    y = grid["y"]
    s = coeffs.vignette_strength * float(H)
    R_v = coeffs.vignette_radius
    dist2 = (x - s) ** 2 + y ** 2          # 到第二孔径圆心的距离平方
    return _soft_step(R_v ** 2 - dist2, softness)


def amplitude(grid: dict, H: float, coeffs, softness: float = 0.02):
    """光瞳振幅 A = M_aperture · M_vignette。对几何参数可微。"""
    A = aperture_mask(grid, coeffs, softness)
    A = A * vignette_mask(grid, H, coeffs, softness)
    return A


def relative_transmission(grid: dict, H: float, coeffs, softness: float = 0.02):
    """口径蚀导致的【相对透过率】T(H) ∈ (0,1]，对系数可微。

    物理动机：PSF 在 psf.py 里被逐个归一化到 sum=1（作为"能量分配核"），
    这会丢掉口径蚀的另一半效应——猫眼除了改变散景【形状】，还降低了离轴点的
    【总能量】（真实镜头的边角失光/光学渐晕）。渲染时应把本系数乘回去：
        最终亮度 = T(H) · (PSF_归一 ⊛ 图像)
    定义：T(H) = Σ A(含猫眼)² / Σ A(无猫眼)²   —— 能量按 |A|² 计。
    H=0 或 vignette_strength=0 时 T=1。
    """
    import torch
    if coeffs.vignette_strength == 0.0 or H == 0.0:
        # 注意返回张量以保持设备/可微链路一致（标量 1）。
        return torch.ones((), device=grid["rho"].device, dtype=grid["rho"].dtype)
    A_full = amplitude(grid, H, coeffs, softness)
    A_open = aperture_mask(grid, coeffs, softness)
    return (A_full ** 2).sum() / ((A_open ** 2).sum() + 1e-12)


# ------------------------------------------------------------------------------
# 4) 组装复光瞳 P = A · exp(i·2π·W)
# ------------------------------------------------------------------------------
def complex_pupil(grid: dict, H: float, coeffs, channel: int | None = None,
                  softness: float = 0.02, phase_scale: float = 1.0):
    """组装复光瞳，返回 complex 张量。这是 psf.py 做 FFT 的输入。

    Args:
        grid: make_pupil_grid 输出。
        H: 归一化像高。
        coeffs: AberrationCoeffs。
        channel: 0/1/2=R/G/B（用于逐通道色差）；None=不分通道。
        softness: 遮罩边界软化尺度。
        phase_scale: 相位缩放因子（=λ_中心/λ_子波长）。波前 W 以"中心波长的 waves"
            计量，相位 ∝ 1/λ；对同一物理波前换到子波长 λ_j，相位整体乘
            s_j = λ_c/λ_j。psf.rgb_psf 用它做【光谱平均】消除单色相干环纹。

    Returns:
        P: [size,size] complex 张量，P = A·exp(i·2π·s·W)。
    """
    import torch
    A = amplitude(grid, H, coeffs, softness)           # 实振幅 [size,size]
    W = wavefront(grid, H, coeffs, channel=channel)    # 波前(waves) [size,size]
    phase = 2.0 * torch.pi * phase_scale * W            # 相位(弧度)
    # 复光瞳：A·(cosφ + i·sinφ)。用 torch.polar(abs, angle) 一步到位且可微。
    P = torch.polar(A, phase)
    return P
