"""
render/demo.py
==============
M2 渲染器的【首张散景图】demo：真实照片 → DA V2 视差 → 像差化分层渲染。

跑通后产出（outputs/render_test/）：
    demo_input.png         输入图（缩放后）
    demo_disparity.png     DA V2 视差灰度图（肉眼核对：近处更亮）
    demo_<preset>.png      各风格预设的散景结果
    demo_grid.png          总对比图（输入 + 视差 + 各预设并排）

用法（bokeh 或 ultralytics env）：
    python -m render.demo                          # 默认 BokehMe demo 图
    python -m render.demo path/to/your.jpg         # 自己的图
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "render_test"
DEFAULT_IMG = PROJECT_ROOT / "third_party" / "BokehMe" / "inputs" / "21.jpg"

# demo 用的镜头预设：名字 → (像差预设名, 是否走 tile 视场相关路径)
# 视场无关的效果（球差/光圈形状）走全局快速路径即可；
# 猫眼/旋焦必须走 tile 路径才会显现。
DEMO_PRESETS = [
    ("ideal", False),
    ("soap_bubble", False),
    ("cream_soft", False),
    ("hexagon", False),
    ("swirl_catseye", True),
]


def load_image(path: Path, max_width: int = 1024):
    """读图 → RGB float [0,1]，限制宽度（demo 速度/显存友好）。返回 numpy (H,W,3)。"""
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[1] > max_width:
        scale = max_width / bgr.shape[1]
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr[:, :, ::-1].astype("float32") / 255.0


def auto_focus_params(disparity):
    """从视差图自动定 (对焦 d_f, 容差 tol, 光圈 K)——demo 与端到端 demo 共用。

    对焦：画面中心大窗口（假设主体居中）的视差中位数；
    容差 δ0：主体自身视差跨度（中心窗口 p15~p85 半宽）——DA V2 的相对视差会把
    主体拉出 0.1~0.3 跨度，没有容差主体内部会被虚掉（DECISIONS D14）；
    光圈 K：背景（5 百分位视差）有效 CoC ≈ 30px（散景球明显、接近真实大光圈），
    全图最大有效 CoC 压在 120px 内。

    【D45 背景虚化不足修复】旧版背景目标 22px + 最大 CoC 上限 60px：当画面有【极近前景】
    （如贴镜松枝，视差≈0.8）时，max_dev 很大 → "最大 CoC 60px" 上限把 K 压到很小
    （实测 21.jpg K 被压到 79）→ 背景 CoC 只剩 ~10px、虚化严重不足（用户反馈）。根因：
    60px 上限是为【旧 wave 模式的 FFT 采样上限 W020<N/8】设的，而默认 geom 模式（D40）
    【无采样上限】、可渲任意大盘 → 上限可放宽。物理上前景极近物体本就该糊成一片（CoC 无上限），
    不该为压前景而牺牲背景。改：背景目标 22→30px、最大 CoC 60→120px（geom 友好）。
    注意：若手动切 psf_mode='wave'，前景大盘可能触发采样上限告警，届时需调小 K 或增大 pupil_size。
    """
    import torch
    h, w = disparity.shape
    center = disparity[h // 2 - 100:h // 2 + 100, w // 2 - 80:w // 2 + 80].flatten()
    d_f = float(center.median())
    # 容差 tol 收紧（D39）：过宽的焦带会一直延伸到背景附近 → 背景刚出焦带、CoC 很小 → 背景
    # 看着仍清晰（P2 根因之一）。旧 (p15~p85)/2、上限 0.15 被对焦窗里的背景撑大到 ~0.12；
    # 改 (p20~p80)/2、上限 0.10，焦带只罩主体自身跨度，背景更早进入虚化。
    tol = float((torch.quantile(center, 0.80) - torch.quantile(center, 0.20)) / 2)
    tol = min(max(tol, 0.03), 0.10)
    d_far = float(torch.quantile(disparity.flatten(), 0.05))
    K = 26.0 / max(abs(d_f - d_far) - tol, 0.05)         # 背景 CoC 目标 26px（散景球明显但不过度）
    max_dev = max(d_f, 1.0 - d_f) - tol
    K = min(K, 100.0 / max(max_dev, 0.1))                # 最大 CoC 100px（geom 无采样上限；D46 由 120
                                                          # 回调到 100：120 在极近前景下 K 过大→背景过度
                                                          # 虚化、杂乱背景成"太多散景盘"，100 更平衡）
    return d_f, tol, K


def remap_disparity(disparity, gamma: float = 1.0):
    """视差去压缩重映射（D39）——对抗 DA V2 的【远/中景非线性压缩】。

    物理上 CoC 对【视差(1/物距)】是线性的（CoC 对【物距】才非线性），而 DA V2 输出的就是
    视差，故我们的 r=K|D−d_f| 已是正确物理，**不需要为物理补非线性**。但 DA V2 的"仿射不变
    逆深度"在实践中并非严格线性于 1/Z：它把中/远景压缩，使主体（人）的视差被拉得离远背景太近
    （21.jpg：人 0.18、建筑 0.0，间隔仅 0.18）→ 焦内主体一动背景就跟着清晰。本函数用幂律
    D' = D^γ（γ<1）把低-中视差段【展开】、近景段压缩，**校正深度模型的压缩偏差**（不是补物理
    非线性）。γ=1 关闭；γ≈0.6~0.8 适合"近前景+中景主体+远背景"被压扁的场景。

    Args:
        disparity: [H,W] 归一化视差（[0,1]，近大远小）。
        gamma: 幂律指数。<1 拉开远-中分离（背景更虚）；=1 不变。
    Returns:
        [H,W] 重映射后的视差（仍在 [0,1]）。
    """
    if abs(gamma - 1.0) < 1e-6:
        return disparity
    return disparity.clamp(0.0, 1.0) ** gamma


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from depth.estimator import load_depth_backend, save_disparity_visualization
    from optics.aberrations import PRESETS
    from render.renderer import RenderControl, render

    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo] device={device}, image={img_path}")

    # ---- 1) 深度：DA V2-Small 推理视差（冻结，仅推理）----
    rgb = load_image(img_path)
    backend = load_depth_backend("depth_anything_v2", device=device)
    disp_np = backend.infer_disparity((rgb * 255).astype("uint8"))
    save_disparity_visualization(disp_np, OUT_DIR / "demo_disparity.png")
    print(f"[demo] 视差 ok: shape={disp_np.shape}")

    image = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)        # [3,H,W]
    disparity = torch.from_numpy(disp_np).to(device)                    # [H,W]

    # 视差边缘吸附：扫平 DA V2 在物体边界的"软坡带"，否则主体周围一圈半虚化轮廓。
    from render.renderer import snap_disparity_edges
    disparity = snap_disparity_edges(disparity)
    # 去压缩重映射（D39）：**默认关闭**（γ=1）。好深度（vitl@770）下主体本就与背景分开，施 γ<1
    # 反而把 d_f 推高、虚掉主体（实测女生脸变糊）。仅当确认 DA V2 压扁了某场景时手动开（sys.argv[2]）。
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    disparity = remap_disparity(disparity, gamma)
    if abs(gamma - 1.0) > 1e-6:
        print(f"[demo] 视差去压缩 γ={gamma}")
    save_disparity_visualization(disparity.cpu().numpy(),
                                 OUT_DIR / "demo_disparity_snapped.png")

    # 高光核心深度统一（D43）：消除小灯珠内部深度跳变造成的"嵌套同心环/外圈淡盘"。
    # 真实图里 DA V2 对小过曝灯珠的深度估计不可靠（同一灯珠像素被判不同视差→分到不同
    # CoC 层→大小不一的盘叠加）。把每个真饱和高光核心统一为其最亮像素的深度→单灯珠单盘。
    from render.renderer import unify_highlight_core_depth
    disparity, n_uni = unify_highlight_core_depth(image, disparity)
    if n_uni > 0:
        print(f"[demo] 高光核心深度统一: {n_uni} 个灯珠核心 → 消嵌套环")

    # ---- 2) 对焦与光圈（逻辑见 auto_focus_params）----
    d_f, tol, K = auto_focus_params(disparity)
    d_far = float(torch.quantile(disparity.flatten(), 0.05))
    max_dev = max(d_f, 1.0 - d_f) - tol
    print(f"[demo] 对焦 d_f={d_f:.3f}±{tol:.3f}, 背景 d_far={d_far:.3f}, K={K:.1f} "
          f"(背景 CoC≈{K * max(abs(d_f - d_far) - tol, 0):.0f}px, "
          f"最大≈{K * max_dev:.0f}px)")

    # ---- 3) 各镜头预设渲染 ----
    results = []
    for name, field_varying in DEMO_PRESETS:
        ctrl = RenderControl(focus_disparity=d_f, aperture_K=K,
                             focus_tolerance=tol,
                             coeffs=PRESETS[name], n_layers=24,
                             # 【D45 gamma 回归标准 2.2】D44 用 γ=4 提亮散景球，但 `**(1/4)` 把球差盘内
                             # 对比【压平】（线性域亮边 6.5× → 显示仅 1.6×）→ soap 亮边/cream 亮心消失、
                             # 且整体"像素响应太强/不柔和"（用户反馈）。回归标准 sRGB γ=2.2（RenderControl
                             # 默认）：球差对比保留（显示 2.3×）、更柔和、物理正确显示。散景球偏暗是 LDR 输入
                             # 固有限制（点光源真实 HDR 亮度被 8-bit 截断）——纯夜景可选开温和 highlight_gain，
                             # 但【不用非物理高 γ 压对比来补偿】。详见 DECISIONS D45。
                             highlight_gain=0.0)
        t0 = time.time()
        with torch.no_grad():                            # demo 不需要梯度，省显存
            out = render(image, disparity, ctrl, field_varying=field_varying)
        dt = time.time() - t0
        out_np = out.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        plt.imsave(str(OUT_DIR / f"demo_{name}.png"), out_np)
        mode = "tile" if field_varying else "global"
        print(f"[demo] {name:18s} ({mode:6s}) {dt:6.2f}s -> demo_{name}.png")
        results.append((name, out_np))

    # ---- 4) 总对比图 ----
    plt.imsave(str(OUT_DIR / "demo_input.png"), rgb)
    n = len(results) + 2
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    panels = [("input", rgb), ("disparity", None)] + results
    for ax, (name, img_p) in zip(axes.flat, panels):
        if name == "disparity":
            ax.imshow(disp_np, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img_p)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("M2 first-light: aberration-controllable bokeh rendering", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "demo_grid.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] 总对比图 -> {OUT_DIR / 'demo_grid.png'}")
    print("[demo] 请肉眼核对：soap_bubble 焦外应有亮边、cream 应柔、hexagon 六边形亮斑、")
    print("       swirl_catseye 画面边缘应现猫眼/旋焦。")


def render_lights_panel():
    """合成"夜景点光源"场景逐预设渲染 —— 盘内结构（六边形/肥皂泡/猫眼）的
    决定性展示。真实照片里灯点often太暗太小，看不清结构；这个面板没有歧义。

    场景：黑背景 + 随机彩色点光源（全部在背景视差 0 处），对焦在近处(d_f=0.85)
    → 所有灯点获得 ~35px 的散景盘。swirl 走 tile 路径（四角应呈切向排列的猫眼）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from optics.aberrations import PRESETS
    from render.renderer import RenderControl, render

    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = 512
    rng = np.random.default_rng(7)
    img = torch.zeros(3, S, S, device=device)
    for _ in range(16):                                   # 16 个灯点：减少重叠让单盘结构可见
        y, x = int(rng.integers(8, S - 8)), int(rng.integers(8, S - 8))
        color = torch.tensor(rng.uniform(0.6, 1.0, 3), device=device,
                             dtype=img.dtype)
        img[:, y - 1:y + 2, x - 1:x + 2] = color[:, None, None]
    disp = torch.zeros(S, S, device=device)               # 灯全在远景

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
    for ax in axes.flat:
        ax.axis("off")                                    # 先全关，有内容的再画
    for ax, (name, field_varying) in zip(axes.flat, DEMO_PRESETS):
        # n_layers 用默认（自适应）：旧值 8 在 K=40 下相邻层 CoC 步长达 ~5px，
        # 单个灯点被 tent 拆到两层 → 弥散圆"中心亮点"伪影（D31）。
        # K=75（盘半径≈64px）：小盘(K=40,34px)处于近焦过渡区、边缘本就软（物理正确，
        # 非 bug，见 outputs/render_test/disc_size_edge.png）；大盘进入几何区 → 平顶+锐边，
        # 盘内的肥皂泡/猫眼/多边形结构也看得更清。
        ctrl = RenderControl(focus_disparity=0.85, aperture_K=75.0,
                             coeffs=PRESETS[name],
                             # D45：gamma 回归标准 2.2（默认），保留球差盘内对比（γ=4 会压平亮边/亮心）；
                             # 下方 o/o.max() 归一化提亮看形态，故纯夜景盘偏暗不影响形态展示。
                             highlight_gain=0.0)
        with torch.no_grad():
            out = render(img, disp, ctrl, field_varying=field_varying)
        o = out.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        ax.imshow((o / max(o.max(), 1e-6)).clip(0, 1))    # 归一化提亮便于读结构
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle("night-lights showcase: per-preset bokeh disk structure\n"
                 "(lights at far background, W020<0 side = preset-naming side)",
                 fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / "demo_lights_grid.png"
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] 夜景灯点展示 -> {out_path}")


if __name__ == "__main__":
    main()
    render_lights_panel()
