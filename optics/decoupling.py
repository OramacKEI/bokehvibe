"""
optics/decoupling.py
====================
【PSF 层面的解耦度（cross-talk）矩阵】—— RQ1"低维系数能否解耦控制焦外风格"的
第一份可量化证据，不依赖训练，M1 即可产出（CLAUDE.md 第 9 节评测协议第 2 条的前置）。

思路：
    为每个【效应旋钮】定义一个可测的【效应签名】（signature）——一个能从 PSF 上
    直接算出来的标量，专门捕捉该旋钮应当控制的视觉特征：

    | 旋钮 (knob)            | 签名 (signature)         | 物理含义 |
    |------------------------|--------------------------|----------|
    | W040 球差              | ring_ratio：外环/内盘亮度比 | >基准=肥皂泡亮边，<基准=奶油 |
    | W131 彗差              | skewness：x 向三阶矩偏度  | 彗星尾的单侧能量甩出（区别于对称平移） |
    | W222 像散              | elongation：log(σx/σy)    | 像散把盘拉成椭圆（沿 x 拉长，符号为正） |
    | vignette 口径蚀        | transmission：相对透过率 T(H) | 猫眼独有的边角失光（相位像差不改 T） |
    （洋葱圈旋钮及其 ripple 签名已随范围调整移除，见 DECISIONS D15）

    然后逐个扰动旋钮 i、测所有签名 j 的【带符号】变化，得矩阵 M[i,j]（按列归一化）。
    【对角占优 = 解耦良好】；非对角元素如实暴露物理上固有的串扰，
    符号信息保留串扰的"方向"——例如猫眼把盘沿 x 压扁（elongation<0）而像散
    沿 x 拉长（>0），二者虽共享该签名但方向可分。这些耦合真实镜头同样存在
    （猫眼+离焦本来就会轻微移动散景球位置），论文中应如实报告而非掩盖。

运行：
    python -m optics.decoupling
产物：
    outputs/psf_test/decoupling_matrix.png   热力图（行=旋钮，列=签名）
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "psf_test"

# 公共测试条件：中等离焦让盘成型；H=0.7 让视场相关效应（彗差/像散/猫眼）显现。
# 离焦取【负值=背景侧】：风格预设命名按背景侧定义（W040>0 在 W020<0 侧出亮边环，
# 见 optics/visualize.py 的符号说明）——这样 W040 行的 ring_ratio 符号为正，语义直观。
DEFOCUS = -12.0
H_TEST = 0.7
PUPIL_SIZE = 256


# ------------------------------------------------------------------------------
# 签名函数：每个都从单个 [N,N] 归一化 PSF（+透过率）算出一个标量
# ------------------------------------------------------------------------------
def _moments(psf):
    """质心 (cx,cy) 与中心二阶矩 (sxx,syy)。后续签名的公共底料。"""
    import torch
    h, w = psf.shape[-2:]
    ys = torch.arange(h, dtype=psf.dtype)
    xs = torch.arange(w, dtype=psf.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx = (psf * xx).sum()
    cy = (psf * yy).sum()
    sxx = (psf * (xx - cx) ** 2).sum()
    syy = (psf * (yy - cy) ** 2).sum()
    return cx, cy, sxx, syy


def sig_ring_ratio(psf):
    """外环(0.75R~1.05R)与内盘(<0.5R)的平均亮度比。球差的签名。"""
    import torch
    cx, cy, sxx, syy = _moments(psf)
    R = float((2.0 * (sxx + syy) / 2.0).sqrt())          # 等效圆盘半径（各向平均）
    h, w = psf.shape[-2:]
    ys = torch.arange(h, dtype=psf.dtype)
    xs = torch.arange(w, dtype=psf.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
    outer = psf[(r > 0.75 * R) & (r < 1.05 * R)]
    inner = psf[r < 0.5 * R]
    return float(outer.mean() / (inner.mean() + 1e-12))


def sig_skewness(psf):
    """x 向三阶标准化矩（偏度）。彗差的签名——彗星尾是【单侧不对称】的能量甩出，
    偏度捕捉的是形状不对称，而非简单平移（质心移动猫眼也会有，偏度更专属彗差）。"""
    import torch
    cx, _, sxx, _ = _moments(psf)
    h, w = psf.shape[-2:]
    xs = torch.arange(w, dtype=psf.dtype)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=psf.dtype), xs, indexing="ij")
    sigma_x = sxx.sqrt() + 1e-12
    return float((psf * ((xx - cx) / sigma_x) ** 3).sum())


def sig_elongation(psf):
    """log(σx/σy)：x/y 方向二阶矩之比的对数。像散的签名（0=各向同性）。"""
    import math
    _, _, sxx, syy = _moments(psf)
    return float(math.log(float(sxx.sqrt()) / (float(syy.sqrt()) + 1e-12)))


SIGNATURES = {
    "ring_ratio": sig_ring_ratio,
    "skewness": sig_skewness,
    "elongation": sig_elongation,
    "transmission": None,   # 直接取 pupil.relative_transmission，不从 PSF 算
}

# 旋钮及其扰动幅度（从 0 → 典型工作值，对应 PRESETS 量级）。
KNOBS = {
    "W040": {"W040_spherical": 1.5},
    "W131": {"W131_coma": 1.0},
    "W222": {"W222_astigmatism": 1.5},
    "vignette": {"vignette_strength": 0.5, "vignette_radius": 1.2},
}


def compute_signatures(coeffs):
    """对一组系数渲染 PSF（单通道、全图不裁剪）并算全部签名。

    这里走【单色】路径（complex_pupil + pupil_to_psf），不做光谱平均：
    签名全是基准/扰动间的【相对量】，两边同为单色即可公平比较；
    单色 PSF 的特征（环/偏度/椭圆度）更锐利，签名的信噪比更高。
    crop=None 保留全图：彗差/猫眼会把能量甩偏，裁剪可能截掉签名所需的尾部。
    """
    from . import pupil, psf as psf_mod
    grid = pupil.make_pupil_grid(PUPIL_SIZE, device="cpu")
    c_d = coeffs.replace(W020_defocus=coeffs.W020_defocus + DEFOCUS)
    P = pupil.complex_pupil(grid, H_TEST, c_d, channel=None)
    psf = psf_mod.pupil_to_psf(P, crop=None)
    out = {}
    for name, fn in SIGNATURES.items():
        if name == "transmission":
            out[name] = float(pupil.relative_transmission(grid, H_TEST, coeffs))
        else:
            out[name] = fn(psf)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .aberrations import AberrationCoeffs

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = AberrationCoeffs()
    sig_base = compute_signatures(base)
    sig_names = list(SIGNATURES.keys())
    knob_names = list(KNOBS.keys())

    # M[i,j] = sig_j(扰动 i) − sig_j(基准)，【保留符号】（串扰方向也是信息）。
    M = np.zeros((len(knob_names), len(sig_names)))
    for i, kname in enumerate(knob_names):
        sig_k = compute_signatures(base.replace(**KNOBS[kname]))
        for j, sname in enumerate(sig_names):
            M[i, j] = sig_k[sname] - sig_base[sname]

    # 按列归一化到 [-1,1]（每个签名的尺度不同，归一后才可横向比较）。
    M_norm = M / (np.abs(M).max(axis=0, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = ax.imshow(M_norm, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sig_names)), sig_names, rotation=30, ha="right")
    ax.set_yticks(range(len(knob_names)), knob_names)
    ax.set_xlabel("effect signature")
    ax.set_ylabel("perturbed knob")
    ax.set_title(f"PSF-level decoupling matrix (signed)\n(defocus={DEFOCUS}w, H={H_TEST}; diagonal dominance = good)")
    for i in range(len(knob_names)):
        for j in range(len(sig_names)):
            ax.text(j, i, f"{M_norm[i, j]:+.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = OUT_DIR / "decoupling_matrix.png"
    fig.savefig(str(out), dpi=120)
    plt.close(fig)

    # 终端摘要：对角元素在各自列里的占比（理想=每列绝对值最大，即 ±1.00）。
    print("[decoupling] 归一化矩阵（行=旋钮, 列=签名, 带符号）：")
    print("            " + "  ".join(f"{s:>12s}" for s in sig_names))
    for i, kname in enumerate(knob_names):
        print(f"{kname:>10s}  " + "  ".join(f"{M_norm[i, j]:+12.3f}" for j in range(len(sig_names))))
    diag = [M_norm[i, i] for i in range(min(len(knob_names), len(sig_names)))]
    print(f"[decoupling] 对角元素: {['%+.2f' % d for d in diag]}（理想=|·|全 1.00）")
    print(f"[decoupling] -> {out}")


if __name__ == "__main__":
    main()
