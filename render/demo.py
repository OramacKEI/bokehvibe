"""
render/demo.py
==============
M2 渲染器的【首张散景图】demo：真实照片 → DA V2 视差 → 像差化分层渲染。

跑通后产出（outputs/render_test/）：
    demo_input.png         输入图（缩放后）
    demo_disparity.png     DA V2 视差伪彩图（肉眼核对：近处更亮）
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
    光圈 K：背景（5 百分位视差）有效 CoC ≈ 22px（盘结构可见、不至于糊成一片），
    全图最大有效 CoC 压在 60px 内。
    """
    import torch
    h, w = disparity.shape
    center = disparity[h // 2 - 100:h // 2 + 100, w // 2 - 80:w // 2 + 80].flatten()
    d_f = float(center.median())
    tol = float((torch.quantile(center, 0.85) - torch.quantile(center, 0.15)) / 2)
    tol = min(max(tol, 0.04), 0.15)
    d_far = float(torch.quantile(disparity.flatten(), 0.05))
    K = 22.0 / max(abs(d_f - d_far) - tol, 0.05)
    max_dev = max(d_f, 1.0 - d_f) - tol
    K = min(K, 60.0 / max(max_dev, 0.1))
    return d_f, tol, K


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
    save_disparity_visualization(disparity.cpu().numpy(),
                                 OUT_DIR / "demo_disparity_snapped.png")

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
                             highlight_gain=8.0,         # 高光提升：让灯点出亮斑散景
                             highlight_thresh=0.82)
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
            ax.imshow(disp_np, cmap="Spectral_r", vmin=0, vmax=1)
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
    for _ in range(60):                                   # 随机彩色灯点
        y, x = int(rng.integers(8, S - 8)), int(rng.integers(8, S - 8))
        color = torch.tensor(rng.uniform(0.6, 1.0, 3), device=device,
                             dtype=img.dtype)
        img[:, y - 1:y + 2, x - 1:x + 2] = color[:, None, None]
    disp = torch.zeros(S, S, device=device)               # 灯全在远景

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
    for ax in axes.flat:
        ax.axis("off")                                    # 先全关，有内容的再画
    for ax, (name, field_varying) in zip(axes.flat, DEMO_PRESETS):
        ctrl = RenderControl(focus_disparity=0.85, aperture_K=40.0,
                             coeffs=PRESETS[name], n_layers=8,
                             highlight_gain=10.0, highlight_thresh=0.5)
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
