"""
optics/calibrate.py
===================
【离焦标定】建立 `W020 (waves) ↔ CoC 像素半径 r_px` 的映射 —— 接渲染器的必经桥梁。

为什么需要它（CLAUDE.md 第 2/3 节的"概念断点"）：
    管线上游给出的是【像素半径】 r = K·|D − d_f|（CoC 公式，单位 px），
    而 PSF 生成器的离焦轴是【波前系数】 W020（单位 waves）。
    渲染器要"按 r 取 PSF"，必须知道 W020 多大对应多大的散景盘。

理论预期（几何光学近似，可作肉眼校验基准）：
    光瞳相位 φ(x) = 2π·W020·ρ²。几何光线的像面落点由局部空间频率决定：
        f(ρ) = (1/2π)·dφ/dρ = 2·W020·ρ   （cycles / 单位光瞳坐标）
    光瞳坐标 x∈[-1,1] 用 N 点采样 → FFT 频率分辨率 = 1/2 cycle/单位。
    边缘光线 (ρ=1) 落在 f/(1/2) = 4·W020 像素处，即：
        ★ r_px ≈ 4 · W020 ★   （斜率 4，与 N 无关——N 只决定上限）
    采样上限（Nyquist）：f_max = 2·W020 < N/4  →  W020 < N/8。
        N=256 → W020 上限 ≈ 32 waves → 最大盘半径 ≈ 128px。

测量方法：
    对理想镜头（无像差、圆瞳）扫 W020，对每个 PSF 计算二阶矩半径：
        均匀圆盘半径 R 满足 E[r²] = R²/2  →  R = sqrt(2·Σ psf·r²)
    纯离焦的 PSF 接近均匀圆盘，二阶矩估计稳健。最后线性拟合 r_px = k·W020 + b。

运行：
    python -m optics.calibrate
产物：
    outputs/psf_test/defocus_calibration.png   标定曲线（实测 vs 理论 r=4·W020）
    outputs/psf_test/defocus_calibration.json  拟合结果 {slope, intercept, ...}
        → 渲染器用法：W020 = (r_px - intercept) / slope
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "psf_test"


def measure_psf_radius(psf):
    """用二阶矩估计 PSF 的等效圆盘半径：R = sqrt(2·E[r²])。

    E[r²] = Σ psf(x,y)·((x-cx)² + (y-cy)²)，psf 已归一化 sum=1。
    对均匀圆盘严格成立；对带亮环/柔边的盘是稳健的"能量等效半径"。
    """
    import torch
    h, w = psf.shape[-2:]
    ys = torch.arange(h, dtype=psf.dtype)
    xs = torch.arange(w, dtype=psf.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # 以能量质心为中心（理想镜头 PSF 居中，但这样写对彗差等偏心情况也成立）。
    cx = (psf * xx).sum()
    cy = (psf * yy).sum()
    r2 = (psf * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum()
    return float((2.0 * r2).sqrt())


def calibrate(pupil_size: int = 256, w020_max: float | None = None, n_steps: int = 16):
    """扫 W020 → 测半径 → 线性拟合。返回 (w020 列表, 实测半径列表, slope, intercept)。"""
    import numpy as np
    from . import pupil, psf as psf_mod
    from .aberrations import AberrationCoeffs

    if w020_max is None:
        w020_max = pupil_size / 8 * 0.75   # 留 25% 安全余量，避开 Nyquist 上限 N/8

    grid = pupil.make_pupil_grid(pupil_size, device="cpu")
    w020_values = np.linspace(1.0, w020_max, n_steps)

    radii = []
    for w in w020_values:
        coeffs = AberrationCoeffs(W020_defocus=float(w))
        P = pupil.complex_pupil(grid, 0.0, coeffs, channel=None)
        # 不裁剪（crop=None）：标定要看全图，避免大盘被裁；guard 保持开启。
        psf = psf_mod.pupil_to_psf(P, crop=None)
        radii.append(measure_psf_radius(psf))
    radii = np.array(radii)

    # 线性拟合 r = k·W020 + b（理论 k≈4, b≈0；b 吸收衍射极限/软边的小偏置）。
    k, b = np.polyfit(w020_values, radii, deg=1)

    # 单调性检查：映射必须单调才能反解 W020(r)。
    assert np.all(np.diff(radii) > 0), "标定曲线非单调——请检查混叠告警或缩小 W020 范围"
    return w020_values, radii, float(k), float(b)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pupil_size = 256
    w020, radii, k, b = calibrate(pupil_size=pupil_size)

    # 与理论斜率 4 对比。
    theory = 4.0 * w020
    rel_err = np.abs(radii - theory) / theory

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(w020, radii, "o-", label=f"measured (fit: r = {k:.3f}·W020 + {b:.2f})")
    ax.plot(w020, theory, "--", label="theory: r = 4·W020")
    ax.set_xlabel("W020 (waves)")
    ax.set_ylabel("PSF radius (px, 2nd-moment)")
    ax.set_title(f"Defocus calibration  (pupil_size={pupil_size}, Nyquist limit W020<{pupil_size // 8})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT_DIR / "defocus_calibration.png"
    fig.savefig(str(out_png), dpi=120)
    plt.close(fig)

    result = {
        "pupil_size": pupil_size,
        "slope_px_per_wave": k,
        "intercept_px": b,
        "theory_slope": 4.0,
        "max_rel_err_vs_theory": float(rel_err.max()),
        "w020_nyquist_limit_waves": pupil_size / 8,
        "usage": "W020 = (r_px - intercept_px) / slope_px_per_wave",
    }
    out_json = OUT_DIR / "defocus_calibration.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"[calibrate] 拟合: r_px = {k:.4f}·W020 + {b:.3f}   (理论斜率 4)")
    print(f"[calibrate] 与理论最大相对误差: {rel_err.max() * 100:.2f}%")
    print(f"[calibrate] -> {out_png}")
    print(f"[calibrate] -> {out_json}")


if __name__ == "__main__":
    main()
