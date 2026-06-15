"""
optics/visualize.py
===================
PSF 生成器的【物理验证脚本】—— 对照 CLAUDE.md 第 3 节"效果如何由参数涌现"那张表，
逐个渲染风格预设并扫球差 W040，把 PSF 存成图，肉眼核对焦外风格是否如期涌现。

运行（环境就绪后）：
    python -m optics.visualize
产物：
    outputs/psf_test/presets.png      各风格预设的 PSF（背景/前景两种离焦符号各一行）
    outputs/psf_test/w040_sweep.png   球差 W040 × 离焦符号 双向扫描（验证符号配对定理）

设计要点：
- 默认 device='cpu'：FFT 在 CPU 即可，物理验证完全不依赖 GPU。
- 离焦量固定中等值 |W020|=12（盘半径≈48px），盘内结构看得清又远离混叠上限。
- 两行 = 两种离焦符号：亮边环只出现在 sign(W040)≠sign(W020) 一侧（D13），
  预设命名按背景侧（W020<0）定义。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "psf_test"


def _to_display(psf):
    """把 [C,h,w] PSF 转成可视化用的 numpy 图。

    PSF 能量集中、峰值远高于盘面，直接显示会几乎全黑。用 gamma 压缩(开方)提亮盘面结构，
    再各自归一到 [0,1]。返回 (rgb_HWC, gray_HW)。
    """
    import numpy as np
    arr = psf.detach().cpu().numpy()
    if arr.shape[0] == 3:
        rgb = arr.transpose(1, 2, 0)           # [h,w,3]
        gray = rgb.mean(axis=2)
    else:
        gray = arr[0]
        rgb = np.stack([gray] * 3, axis=2)

    def norm_gamma(a, g=0.5):
        a = np.maximum(a, 0)
        a = a ** g                              # gamma 压缩提亮暗部结构
        m = a.max()
        return a / m if m > 0 else a

    return norm_gamma(rgb), norm_gamma(gray)


def render_presets():
    """渲染所有风格预设的 PSF 并存成一张总图。

    【离焦符号（关键，曾在此踩坑）】本项目约定 W020 = K·(D − d_f)/slope：
    背景(D<d_f) → W020 **负**；前景 → 正。球差的亮边环出现在 W040 与 W020
    **异号**的一侧（边缘光线折返形成焦散环；同号侧则是奶油柔散）。
    soap_bubble/cream_soft 的命名按**背景侧**（经典 Trioplan 场景）定义，
    因此预设验证必须用 **W020=−12**（背景侧）渲染；上下两行分别给出
    背景/前景两侧，顺便展示"前/背景形态反转"这一标志特征。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import pupil, psf as psf_mod
    from .aberrations import PRESETS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pupil_size, crop = 256, 128

    grid = pupil.make_pupil_grid(pupil_size, device="cpu")

    names = list(PRESETS.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4))

    for row, defocus in enumerate([-12.0, +12.0]):       # 上：背景侧；下：前景侧
        for j, name in enumerate(names):
            coeffs = PRESETS[name]
            # 猫眼/单侧亮边类需要在非零像高才显现，给个 H=0.8；其余用 H=0。
            H = 0.8 if coeffs.vignette_strength > 0 else 0.0
            c_d = coeffs.replace(W020_defocus=coeffs.W020_defocus + defocus)
            rgb_psf = psf_mod.rgb_psf(grid, H, c_d, crop=crop)   # [3,crop,crop]
            rgb_img, _ = _to_display(rgb_psf)
            axes[row, j].imshow(rgb_img)
            if row == 0:
                axes[row, j].set_title(f"{name}\n(H={H})", fontsize=8)
            axes[row, j].axis("off")
    axes[0, 0].text(-0.15, 0.5, "background\nW020=-12", transform=axes[0, 0].transAxes,
                    fontsize=9, va="center", ha="right")
    axes[1, 0].text(-0.15, 0.5, "foreground\nW020=+12", transform=axes[1, 0].transAxes,
                    fontsize=9, va="center", ha="right")
    fig.suptitle("PSF presets: BACKGROUND side defines preset names "
                 "(soap_bubble=bright edge @W020<0); foreground row shows inversion",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "presets.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def render_w040_sweep():
    """扫球差 W040 从负到正 × 离焦正负两侧 —— 验证亮边环出现在"W040 与 W020 异号"侧。

    上行 W020=−12（背景侧）：W040>0 → 肥皂泡亮边；W040<0 → 奶油。
    下行 W020=+12（前景侧）：完全反转。这正是"前/背景形态反转"的标志特征。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from . import pupil, psf as psf_mod
    from .aberrations import AberrationCoeffs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pupil_size, crop = 256, 128
    grid = pupil.make_pupil_grid(pupil_size, device="cpu")

    w040_values = np.linspace(-2.0, 2.0, 9)
    fig, axes = plt.subplots(2, len(w040_values),
                             figsize=(2.0 * len(w040_values), 4.8))
    for row, defocus in enumerate([-12.0, +12.0]):
        for j, w in enumerate(w040_values):
            coeffs = AberrationCoeffs(W040_spherical=float(w), W020_defocus=defocus)
            P = pupil.complex_pupil(grid, 0.0, coeffs, channel=1)   # 用 G 通道看强度结构
            psf = psf_mod.pupil_to_psf(P, crop=crop)[None]
            _, gray = _to_display(psf)
            axes[row, j].imshow(gray, cmap="inferno")
            if row == 0:
                axes[row, j].set_title(f"W040={w:+.1f}", fontsize=8)
            axes[row, j].axis("off")
    axes[0, 0].text(-0.15, 0.5, "bg side\nW020=-12", transform=axes[0, 0].transAxes,
                    fontsize=9, va="center", ha="right")
    axes[1, 0].text(-0.15, 0.5, "fg side\nW020=+12", transform=axes[1, 0].transAxes,
                    fontsize=9, va="center", ha="right")
    fig.suptitle("W040 sweep × defocus sign: bright edge ring appears when "
                 "sign(W040) ≠ sign(W020)  (background row defines preset naming)",
                 fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "w040_sweep.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("[optics] 渲染风格预设 ...")
    p1 = render_presets()
    print(f"  -> {p1}")
    print("[optics] 渲染 W040 扫描 ...")
    p2 = render_w040_sweep()
    print(f"  -> {p2}")
    print("[optics] 完成。请肉眼核对：")
    print("  - cream_soft 应中心亮、边缘柔；soap_bubble 应有边缘亮环；")
    print("  - hexagon 应是六边形盘；")
    print("  - swirl_catseye / double_gauss_edge 在 H=0.8 应呈猫眼/单侧亮边。")
