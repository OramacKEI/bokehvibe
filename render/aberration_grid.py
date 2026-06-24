"""
render/aberration_grid.py
=========================
逐像差「7×7 点光源阵列」验证图 —— 肉眼核对当前渲染器对【每一种像差】在
【背景离焦 / 焦平面 / 前景离焦】三种工况下的散景 PSF 形态是否正确涌现。

================================ 场景设计 ================================
纯黑背景上摆 7×7 个白色小圆点光源（半径 ~2px），**全部置于同一视差 D=0.5**。
三种工况只移动【对焦视差 d_f】（灯不动），于是背景/前景两侧 |离焦| 严格相等：

    工况            d_f     灯相对焦面    带符号 CoC r      说明
    ───────────────────────────────────────────────────────────────────
    背景离焦 bg     1.0     比焦面远      r<0 (W020<0)     预设命名口径侧（如
                                                          soap 出亮边环、cream 亮心）
    焦平面   focus  0.5     就在焦面      r=0              锐利点；离焦类像差消失，
                                                          但彗差/像散/场曲等【视场
                                                          像差】在离轴处仍残留（物理正确）
    前景离焦 fg     0.0     比焦面近      r>0 (W020>0)     球差类形态相对背景侧【翻转】
                                                          （本项目标志特征）

两侧 |r| 相等（灯固定在中点 D=0.5）→ 离焦【基线】盘同尺寸；但像差会在此基线上叠加形变，
故不同像差/前后焦的盘尺寸【本就不该相等】（这是物理，不是 bug）：
- 球差(W040≠0)使前/后焦盘【尺寸与形态都不对称】——球差的定义性标志：边缘光线在 W020/W040
  【异号】侧折返、堆成小盘+亮边环；【同号】侧铺开成大盘。实测 W040=+1.5：背景盘≈27px、
  前景盘≈51px（离焦基线 38px）。前后焦弥散圆不对称正是球差区别于纯离焦的关键，不归一化。
- 欠矫正(under)盘偏大、过矫正(over)盘偏小、变迹(apod)边缘渐暗看着偏小——均为物理正确。

【对焦到最小弥散圆】本验证【开启】球差焦移补偿 balance_spherical_focus（=渲染器/demo 默认，D46）：
给所有层加 −1.3·W040 的恒定移焦，模拟摄影师把焦对到【最小弥散圆】而非【傍轴焦点】。
于是球差类的【焦平面】只残留轻微弥散（90% 能量半径≈3px，正是 soap/Trioplan 镜头焦内的
轻 glow——真实球差镜头合焦本就略软、无法对成完美点），而非傍轴焦点处的大糊盘（≈10px，
那不是真实拍摄观感；早期关闭 balance 时焦平面发糊即源于此）。无球差像差(W040≈0)偏移≈0、
焦平面仍是纯锐点。

================================ 7×7 的意义 ================================
中心灯 H≈0（无视场效应），角灯 H≈0.9（视场效应最强）→ 一张图里同时看到
彗差 / 像散 / 场曲 / 猫眼 / LaCA 随像高的连续变化（视场相关像差走 tile 路径）。
视场无关的像差（球差 / 多边形 / 变迹 / LoCA / 球色差）则 49 个盘形态一致（走全局快速路径）。

================================ 产物 ================================
outputs/aberration_grid/
    input_scene.png      输入点光源阵列（参考）
    <name>_strip.png     每个像差一张三联：background | in-focus | foreground（全分辨率，可放大读结构）
    montage.png          总览：行=像差，列=三工况
    aperture_starburst_panel.png
                         光圈↔星芒【联动】演示：同一 n_blades 同时驱动散景盘边数与星芒芒数
                         （方阑→方盘+十字、五边→五边盘+10芒…），blade_curvature 同时软化盘角与 spike。
                         证明星芒是多边形光圈的衍射表现(非独立预设、非彗差)，见 render_aperture_starburst_panel

用法（bokeh env）：
    ~/anaconda3/envs/bokeh/bin/python -m render.aberration_grid
"""

from __future__ import annotations

import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "aberration_grid"

# 画幅与阵列参数
IMG_SIZE = 1024          # 正方形边长
N_LIGHTS = 7             # 每行/列灯数（7×7）
GRID_FRAC = (0.085, 0.915)  # 阵列在画幅内的占比范围（留边，角灯 H≈0.9 又不贴边）
LIGHT_RADIUS = 2         # 单个点光源半径(px)（≈5px 直径的小圆，单像素易丢失故取 2）
APERTURE_K = 76.0        # 光圈增益：|r| = K·|D−d_f| = 76·0.5 = 38px 散景盘（盘大、结构清、49 个不重叠）


def build_aberrations():
    """返回待验证的像差列表：(name, coeffs, field_varying, 英文标签)。

    field_varying=True 的像差【依赖像高】，必须走 tile 路径 + 离轴灯才显现
    （彗差/像散/场曲/猫眼/LaCA）；False 的像差视场无关，走全局快速路径即可。
    系数取值偏「夸张」以便肉眼读形态（验证用，非训练分布）。
    """
    from optics.aberrations import AberrationCoeffs as AC
    return [
        # ---- 波前像差（相位项）----
        ("ideal",           AC(),                                              False, "ideal (pure defocus)"),
        ("spherical_under", AC(W040_spherical=-1.5),                           False, "spherical W040<0 (under)"),
        ("spherical_over",  AC(W040_spherical=+1.5),                           False, "spherical W040>0 (over)"),
        ("spherical_high",  AC(W040_spherical=-1.0, W060_spherical2=+1.3),     False, "high-order W060 (nisen)"),
        ("coma",            AC(W131_coma=2.5),                                 True,  "coma W131 (field)"),
        ("astigmatism",     AC(W222_astigmatism=2.5),                          True,  "astigmatism W222 (field)"),
        ("field_curvature", AC(W220_field_curv=2.5),                           True,  "field curvature W220 (field)"),
        # ---- 光瞳振幅（遮罩项）----
        ("polygon",         AC(n_blades=6, blade_curvature=0.25),              False, "polygon aperture (6 blades)"),
        ("catseye_swirl",   AC(vignette_strength=0.9, vignette_radius=1.05),   True,  "cat's-eye / swirl (field)"),
        ("apodization",     AC(apodization=2.5),                               False, "apodization / STF"),
        # ---- 色差（逐通道）----
        ("loca",            AC(loca_rgb=(0.6, -0.6, 0.6)),                     False, "LoCA (axial color, green-magenta)"),
        ("laca",            AC(laca_rgb=(6.0, 0.0, -6.0)),                     True,  "LaCA (lateral color, field)"),
        ("spherochrom",     AC(W040_spherical=+0.8,
                               spherochrom_rgb=(0.6, 0.0, -0.6)),              False, "spherochromatism"),
    ]


# 三种工况：灯固定 D=0.5，只移 d_f（见模块头表格）
CONFIGS = [
    ("background", 1.0, "background defocus (r<0)"),
    ("infocus",    0.5, "in-focus (r=0)"),
    ("foreground", 0.0, "foreground defocus (r>0)"),
]


def make_light_grid(device, dtype):
    """合成 7×7 白色点光源阵列（黑背景），并返回 (image[3,S,S], disparity[S,S])。

    灯全部置于视差 D=0.5（同一深度），三种工况靠移动 d_f 实现 bg/focus/fg。
    """
    import torch
    S = IMG_SIZE
    img = torch.zeros(3, S, S, device=device, dtype=dtype)
    lo, hi = GRID_FRAC
    coords = [int(round((lo + (hi - lo) * i / (N_LIGHTS - 1)) * S)) for i in range(N_LIGHTS)]
    R = LIGHT_RADIUS
    for cy in coords:
        for cx in coords:
            # 画一个半径 R 的实心小圆（比单像素更稳健，避免被分层/重采样吞掉）。
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    if dy * dy + dx * dx <= R * R:
                        img[:, cy + dy, cx + dx] = 1.0
    disp = torch.full((S, S), 0.5, device=device, dtype=dtype)
    return img, disp


def normalize_for_display(out_np):
    """逐图归一化以凸显散景盘结构：除以【全图最大值】，裁 [0,1]。

    黑背景上灯很暗（点光源能量被散景盘摊薄是物理正确，见 render.demo D45 注释），
    故展示时统一提亮。各图独立归一 → 比的是【形态】不是绝对亮度。

    【为何用 max 而非百分位】焦平面工况下灯是【锐点】，只有 ~0.06% 的像素是亮的，
    其余 99.9%+ 是黑背景（含渲染器 out/acc 近零相除留下的 ~1e-4 数值底噪——绝对值
    远低于 8-bit 量化阶 1/255，正常曝光下不可见）。若按 99.7 百分位归一，该百分位会
    落进【底噪】而非亮点里 → 除以 ~5e-4 把底噪放大成可见灰噪点（之前 infocus 图的
    "奇怪噪声"）。改用 max：焦点像素值≈1.0 → 底噪/1.0≈1e-4 不可见；散景盘工况实测
    峰/内比仅 ~1.5（无单像素尖点），max 归一不会压暗盘结构。等价于 render_lights_panel
    的 o/o.max() 做法。"""
    import numpy as np
    ref = max(float(out_np.max()), 1e-6)
    return np.clip(out_np / ref, 0.0, 1.0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from render.renderer import RenderControl, render

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grid] device={device}, image={IMG_SIZE}², lights={N_LIGHTS}×{N_LIGHTS}, K={APERTURE_K}")

    image, disparity = make_light_grid(device, torch.float32)
    plt.imsave(str(OUT_DIR / "input_scene.png"),
               image.cpu().numpy().transpose(1, 2, 0))

    aberrations = build_aberrations()
    # results[name] = {config_name: normalized HxWx3}
    results: dict[str, dict[str, "np.ndarray"]] = {}

    t_start = time.time()
    for name, coeffs, field_varying, label in aberrations:
        results[name] = {}
        for cfg_name, d_f, _ in CONFIGS:
            ctrl = RenderControl(
                focus_disparity=d_f,
                aperture_K=APERTURE_K,
                focus_tolerance=0.0,           # 阵列同深度，无需容差；focus 工况即纯 r=0
                coeffs=coeffs,
                balance_spherical_focus=True,   # 开焦移补偿=对焦到【最小弥散圆】(D46)：球差合焦只残留
                                                # 轻微弥散(真实 soap 镜头焦内 glow)，而非傍轴焦点的大糊盘；
                                                # 无球差像差(W040≈0)偏移≈0、合焦仍是纯点（见模块头）
                highlight_gain=0.0,             # 不做 HDR 放大，只看 PSF 形态（归一化提亮）
                n_layers=24,
            )
            t0 = time.time()
            with torch.no_grad():
                out = render(image, disparity, ctrl, field_varying=field_varying)
            dt = time.time() - t0
            out_np = out.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            results[name][cfg_name] = normalize_for_display(out_np)
            mode = "tile" if field_varying else "global"
            print(f"[grid] {name:16s} {cfg_name:10s} ({mode:6s}) {dt:5.2f}s")

        # 每个像差的全分辨率三联条带（background | in-focus | foreground），可放大读结构。
        sep = np.ones((IMG_SIZE, 6, 3), dtype=np.float32)  # 白色分隔条
        strip = np.concatenate([
            results[name]["background"], sep,
            results[name]["infocus"], sep,
            results[name]["foreground"],
        ], axis=1)
        plt.imsave(str(OUT_DIR / f"{name}_strip.png"), strip)

    # ----------------- 总览 montage：行=像差，列=三工况 -----------------
    n_rows = len(aberrations)
    n_cols = len(CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    for r, (name, coeffs, field_varying, label) in enumerate(aberrations):
        for c, (cfg_name, d_f, cfg_label) in enumerate(CONFIGS):
            ax = axes[r, c]
            ax.imshow(results[name][cfg_name])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cfg_label, fontsize=12)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle(
        "Aberration verification: 7x7 point-light grid  "
        "(lights at D=0.5; columns move focus plane d_f; balance OFF)\n"
        "background col = preset-naming side (W020<0)   |   "
        "field aberrations vary across the grid (center H~0 -> corner H~0.9)",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    montage_path = OUT_DIR / "montage.png"
    fig.savefig(str(montage_path), dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"[grid] 总耗时 {time.time() - t_start:.1f}s")
    print(f"[grid] 总览 -> {montage_path}")
    print(f"[grid] 全分辨率三联 -> {OUT_DIR}/<name>_strip.png  ({n_rows} 个像差)")
    print("[grid] 肉眼核对要点：")
    print("  · spherical_over: background 列出锐亮边环、foreground 列翻转为亮心/柔散（形态反转）")
    print("  · spherical_under: 与 over 相反；high-order W060: 盘内带状双线(nisen)")
    print("  · coma: 角灯单侧彗尾、朝向随方位角；astigmatism: 角灯拉成椭圆/线；")
    print("  · field_curvature: 中心锐、角灯离焦(焦面是曲面)；catseye: 角灯切成朝心的眼形")
    print("  · polygon: 六边形亮斑；apodization: 盘边柔化无硬边；")
    print("  · LoCA: 盘心紫/绿偏移；LaCA: 角灯径向红蓝分离(中心灯无色差)；spherochrom: 盘边彩色镶边")


# 联动演示扫描的叶片数（偶数边→N 芒、奇数边→2N 芒）。
APERTURE_BLADES = [4, 5, 6, 8]


def render_aperture_starburst_panel():
    """光圈↔星芒【联动】演示：散景盘边数与星芒芒数由【同一个 n_blades】驱动。

    星芒不是独立效果，而是多边形光圈（polygon 像差）的【衍射表现】，与散景盘【同源】：
      · 大离焦 → 多边形【散景盘】（N 边，geom 硬边）；
      · 近合焦小离焦 → 【星芒】（偶数边 N→N 芒、奇数边→2N 芒，wave 衍射）；
      · 二者由【同一 n_blades】决定；blade_curvature 同时软化【盘角】与【spike】
        （真实镜头叶片微凸 → 盘角变圆 + 星芒变弱，解释为何现代镜头星芒不锐）。
    故框架【不单列星芒预设】——任何多边形光圈切到 psf_mode='wave' + 小离焦即自动涌现星芒。

    本面板对每个 n_blades 并排画 [散景盘 | 星芒]，并对比直边(curv=0)/弧边(curv=0.3)，
    用单个点光源放大看清形态。盘走 geom（硬边多边形），星芒走 wave + 硬边光瞳(softness=0.01)——
    主管线默认 softness=0.04 是为抑制散景盘相干边环设的(D28)、会削弱 spike，故星芒单独指定。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from optics.aberrations import AberrationCoeffs
    from optics import psf as psfmod, pupil as pupilmod
    from render.renderer import fft_conv2d

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = 320                                          # 单点放大画幅
    img = torch.zeros(3, S, S, device=device)
    img[:, S // 2, S // 2] = 1.0
    grid = pupilmod.make_pupil_grid(256, device=device, dtype=img.dtype)

    def disk(nb, curv):                              # 多边形散景盘（geom 大离焦硬边）
        c = AberrationCoeffs(W020_defocus=9.6, n_blades=nb, blade_curvature=curv)
        psf = psfmod.rgb_geometric_psf(0.0, c, crop=130, samples=512, device=device)
        o = fft_conv2d(img, psf).clamp(min=0.0).cpu().numpy().transpose(1, 2, 0)
        return np.clip(o / max(o.max(), 1e-9), 0, 1)

    def star(nb, curv):                              # 星芒（wave 小离焦 + 硬边光瞳）
        c = AberrationCoeffs(W020_defocus=2.5, n_blades=nb, blade_curvature=curv)
        psf = psfmod.rgb_psf(grid, 0.0, c, crop=240, softness=0.01, check_sampling=False)
        o = fft_conv2d(img, psf).clamp(min=0.0).cpu().numpy().transpose(1, 2, 0)
        o = o / max(float(np.percentile(o, 99.95)), 1e-9)
        return np.clip(o, 0, 1) ** 0.4               # gamma<1 提亮暗 spike

    cols = [("disk (curv=0)", lambda nb: disk(nb, 0.0)),
            ("starburst (curv=0)", lambda nb: star(nb, 0.0)),
            ("disk (curv=0.3)", lambda nb: disk(nb, 0.3)),
            ("starburst (curv=0.3)", lambda nb: star(nb, 0.3))]
    fig, axes = plt.subplots(len(APERTURE_BLADES), 4,
                             figsize=(15, 3.6 * len(APERTURE_BLADES)))
    for r, nb in enumerate(APERTURE_BLADES):
        rays = nb if nb % 2 == 0 else 2 * nb         # 偶数边→N 芒、奇数边→2N 芒
        for c, (lab, fn) in enumerate(cols):
            ax = axes[r, c]
            with torch.no_grad():
                ax.imshow(fn(nb))
            ax.axis("off")
            if r == 0:
                ax.set_title(lab, fontsize=11)
            if c == 0:
                ax.text(-0.10, 0.5, f"n_blades={nb}\n{nb}-gon disk / {rays}-ray star",
                        transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=10)
    fig.suptitle("aperture <-> starburst LINKAGE: one n_blades drives BOTH the bokeh-disk edges "
                 "and the spike count;\nblade_curvature softens BOTH the disk corners and the spikes "
                 "(=> starburst is NOT a separate preset)",
                 fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / "aperture_starburst_panel.png"
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] 光圈↔星芒联动演示 -> {path}")


if __name__ == "__main__":
    main()
    render_aperture_starburst_panel()
