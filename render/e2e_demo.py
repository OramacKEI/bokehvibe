"""
render/e2e_demo.py
==================
M2 收官件：【真实照片】的完整管线端到端 demo。

    真实照片 → DA V2 视差（冻结推理）→ 像差化物理渲染（tile 视场相关）
             → 边界细化网（训练好的 checkpoint）→ 融合输出
    并与"纯物理渲染"并排对照——细化网的净收益应肉眼可见地体现在深度边界
    （主体轮廓的渗色/硬边/半透明圈变干净），平坦区域则应几乎不动
    （误差图融合的设计目标：不洗掉物理层的像差风格）。

【细化网怎么用到整张图上（关键实现策略）】
细化网是在 512² 裁块上训练的，且每个训练样本整块共享一个视场位置
(H_field, azimuth)（synth 的"单 PSF 视场近似"，DECISIONS D8 一脉相承）。
直接整图一次前向会让 FiLM 拿到错误的视场条件（整图只有一个 H/azimuth，
而真实画幅上各处不同）。因此推理时同样按 512² 滑窗跑：

    512px 窗口、256px 步长（50% 重叠）滑过整图
    → 每窗用【窗口中心】的 (H, azimuth) 拼 13 维 ctrl_vec（与训练分布一致，
      像高/方位角的几何约定与 render_tiled 完全相同）
    → 各窗输出用 Hann 窗加权融合（重叠区平滑过渡、无接缝）

网络本身全卷积，512 窗口与训练裁块同尺寸 → 输入分布最贴近训练。
~0.7M 参数的网络一窗前向 <10ms，整图 ~35 窗的开销可忽略。

【ctrl_vec 的两个易错点】（接口约定见 data/synth.py make_sample / refine/network.py）
① azimuth 约定：synth 采样 [0, 2π) 后存 azimuth/π ∈ [0, 2)；math.atan2 返回
   [-π, π]，必须先 wrap 到 [0, 2π) 再除 π，否则负角超出训练分布。
② K 的语义：CoC 增益（像素/单位视差），ctrl_vec 存 K/100——与训练侧
   make_sample 完全同口径（细化网用 [0][1] 两维现场算离焦提示通道）。

用法（bokeh env，需训练好的 checkpoint）：
    python -m render.e2e_demo                              # 默认图 + 最新权重
    python -m render.e2e_demo path/to/photo.jpg
    python -m render.e2e_demo --ckpt outputs/train_run1/ckpt_latest.pth

产出（outputs/e2e_test/）：
    e2e_<preset>_{phys,refined,mask}.png   各预设：纯渲染 / 细化后 / 误差图
    e2e_grid.png                           总对比面板（含边界放大窗）
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "e2e_test"
DEFAULT_IMG = PROJECT_ROOT / "third_party" / "BokehMe" / "inputs" / "21.jpg"

# 端到端展示用预设：覆盖"理想镜头 / 视场无关像差 / 视场相关像差"三类。
E2E_PRESETS = ["ideal", "soap_bubble", "swirl_catseye"]


def find_default_ckpt() -> Path:
    """默认 checkpoint：优先 P1 训练 run3，其次 v1 run2，再次验证训练 run1。"""
    for run in ["train_run3", "train_run2", "train_run1"]:
        p = PROJECT_ROOT / "outputs" / run / "ckpt_latest.pth"
        if p.exists():
            return p
    raise FileNotFoundError("找不到训练 checkpoint，请先跑 train.train 或用 --ckpt 指定")


# ==============================================================================
# 1) 滑窗细化：512² 窗口 + 窗心视场条件 + Hann 加权融合
# ==============================================================================
def _window_positions(total: int, patch: int, stride: int) -> list[int]:
    """一维滑窗起点：步长 stride 覆盖全长，最后一窗对齐末端（保证全覆盖）。"""
    if total <= patch:
        return [0]
    pos = list(range(0, total - patch + 1, stride))
    if pos[-1] != total - patch:
        pos.append(total - patch)
    return pos


def refine_full_image(net, image, disparity, bokeh_phys, base_ctrl: list,
                      patch: int = 512, stride: int = 256):
    """滑窗跑细化网并 Hann 融合成整图（实现策略见文件头）。

    Args:
        net: 已加载权重的 RefineNet（eval 模式）。
        image/bokeh_phys: [3,H,W]；disparity: [H,W]（与训练同口径，sRGB/[0,1]）。
        base_ctrl: 13 维 ctrl_vec 的 Python list，[11](H_field)/[12](azimuth/π)
                   两维占位即可——每窗按窗心几何现场覆写。
    Returns:
        (refined [3,H,W], mask [1,H,W])  融合输出与误差图（误差图仅供可视化）。
    """
    import torch

    dev, dt = image.device, image.dtype
    _, H, W = image.shape
    # 图比窗口还小（罕见）：退到整图单窗，尺寸取 4 的倍数（网络下采样×4 的要求）。
    patch = min(patch, H // 4 * 4, W // 4 * 4)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    half_diag = math.sqrt(cy ** 2 + cx ** 2)

    # Hann² 融合窗：重叠区平滑过渡。边缘值取下限 1e-3 而不是 0——
    # 图像四角只有一个窗覆盖，若权重恰为 0 会出现 0/0。
    w1 = torch.hann_window(patch, periodic=False, device=dev, dtype=dt)
    w2d = (w1[:, None] * w1[None, :]).clamp(min=1e-3)        # [P,P]

    num = torch.zeros_like(image)                            # 加权和
    den = torch.zeros(1, H, W, device=dev, dtype=dt)         # 权重和
    mask_num = torch.zeros(1, H, W, device=dev, dtype=dt)

    cvec = torch.tensor(base_ctrl, device=dev, dtype=dt)
    with torch.no_grad():
        for y0 in _window_positions(H, patch, stride):
            for x0 in _window_positions(W, patch, stride):
                y1, x1 = y0 + patch, x0 + patch
                # 窗心 → 像高/方位角（几何约定与 render_tiled 一致：y 向下）。
                pcy, pcx = (y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0
                H_field = math.sqrt((pcy - cy) ** 2 + (pcx - cx) ** 2) / half_diag
                azim = math.atan2(pcy - cy, pcx - cx) % (2 * math.pi)  # wrap 到 [0,2π)
                cvec[11], cvec[12] = H_field, azim / math.pi
                out = net(image[None, :, y0:y1, x0:x1],
                          disparity[None, y0:y1, x0:x1],
                          bokeh_phys[None, :, y0:y1, x0:x1], cvec[None])
                num[:, y0:y1, x0:x1] += w2d * out["bokeh"][0]
                mask_num[:, y0:y1, x0:x1] += w2d * out["error_mask"][0]
                den[:, y0:y1, x0:x1] += w2d
    den = den.clamp(min=1e-8)
    return (num / den).clamp(0.0, 1.0), mask_num / den


def refine_full_image_v2(net, image, disparity, bokeh_phys, base_ctrl: list,
                         ctrl, use_gate: bool, H_bins: int = 6):
    """P1 整图【单次前向】细化（替代 v1 滑窗，NETWORK_DESIGN §4.1 的收益③）。

    训练时网络见过逐像素 (H, 方位角, PSF 描述子) 条件图 → 推理直接按真实画幅
    几何构建同款条件图，整图一次前向：滑窗/Hann 融合/azimuth wrap 全部退役。
    ctrl_vec 的 [11][12]（块级 H/azimuth）在整图下退化为均值占位——空间信息
    已由条件图逐像素携带，FiLM 只负责全局"镜头是什么"。

    Args:
        net: RefineNet（extra_cond_ch>0 的 v2 权重）。
        ctrl: RenderControl（构建描述子表/边界带用，与渲染同一份）。
        use_gate: 训练时是否开了边界带门控（ckpt 记录），推理同口径。
    Returns:
        (refined [3,H,W], mask [1,H,W])
    """
    import torch
    import torch.nn.functional as F
    from refine.conditioning import (boundary_band, condition_maps,
                                     field_geometry)

    _, H, W = image.shape
    # 网络下采样×4 要求边长为 4 的倍数：replicate 补齐，前向后裁回。
    ph, pw = (-H) % 4, (-W) % 4
    pad = (0, pw, 0, ph)
    img_p = F.pad(image[None], pad, mode="replicate")[0]
    disp_p = F.pad(disparity[None, None], pad, mode="replicate")[0, 0]
    phys_p = F.pad(bokeh_phys[None], pad, mode="replicate")[0]

    # 逐像素几何（真实画幅）+ 全幅 H 桶描述子表（与 render_tiled 的 H_bins 同款分桶）。
    H_map, az_map = field_geometry(disp_p.shape, disp_p.device, disp_p.dtype)
    H_centers = torch.linspace(0.0, 1.0, H_bins).tolist()
    with torch.no_grad():
        cond = condition_maps(disp_p, ctrl, H_map, az_map, H_centers,
                              with_descriptors=(net.extra_cond_ch == 9))
        band = boundary_band(disp_p, ctrl) if use_gate else None
        cvec = torch.tensor(base_ctrl, device=image.device, dtype=image.dtype)
        cvec[11], cvec[12] = float(H_map.mean()), 1.0
        out = net(img_p[None], disp_p[None], phys_p[None], cvec[None],
                  cond_maps=cond[None],
                  mask_gate=None if band is None else band[None])
    refined = out["bokeh"][0, :, :H, :W].clamp(0.0, 1.0)
    return refined, out["error_mask"][0, :, :H, :W]


def refine_full_image_matte(net, image, disparity, bokeh_phys, base_ctrl: list,
                            ctrl, use_gate: bool, H_bins: int = 6):
    """Plan B（D29）matte 选择式重渲的整图推理。

    与训练 refine_forward 的 matte 分支同口径：
        B_fg = bokeh_phys（标准渲染）；band = boundary_band；
        B_bg = render_tiled(image, push_band_to_background(disp))（带内压到背景重虚化）；
        网络出 raw matte α（整图单前向，逐像素条件图）→
        B = band·(α·B_fg + (1−α)·B_bg) + (1−band)·B_fg
    网络从未接触颜色通道 → 真背景（树枝）处 α→0 取 B_bg(虚) → 找补结构上不可能。

    Returns: (refined [3,H,W], matte [1,H,W])
    """
    import torch
    import torch.nn.functional as F
    from refine.conditioning import boundary_band, condition_maps, field_geometry
    from render.renderer import push_band_to_background, render_tiled

    _, H, W = image.shape
    band = boundary_band(disparity, ctrl)                  # [1,H,W]
    with torch.no_grad():
        disp_bg = push_band_to_background(disparity, ctrl, band=band)
        B_bg = render_tiled(image, disp_bg, ctrl).clamp(0.0, 1.0)

    # 网络下采样×4 要求边长为 4 的倍数：replicate 补齐，前向后裁回。
    ph, pw = (-H) % 4, (-W) % 4
    pad = (0, pw, 0, ph)
    img_p = F.pad(image[None], pad, mode="replicate")[0]
    disp_p = F.pad(disparity[None, None], pad, mode="replicate")[0, 0]
    phys_p = F.pad(bokeh_phys[None], pad, mode="replicate")[0]

    H_map, az_map = field_geometry(disp_p.shape, disp_p.device, disp_p.dtype)
    H_centers = torch.linspace(0.0, 1.0, H_bins).tolist()
    with torch.no_grad():
        cond = condition_maps(disp_p, ctrl, H_map, az_map, H_centers,
                              with_descriptors=(net.extra_cond_ch == 9))
        cvec = torch.tensor(base_ctrl, device=image.device, dtype=image.dtype)
        cvec[11], cvec[12] = float(H_map.mean()), 1.0
        # matte 模式：mask_gate=None（门控由 band splice 承担，与训练一致）。
        out = net(img_p[None], disp_p[None], phys_p[None], cvec[None],
                  cond_maps=cond[None], mask_gate=None)
    alpha = out["matte"][0, :, :H, :W]                     # [1,H,W] raw α
    B_band = alpha * bokeh_phys + (1.0 - alpha) * B_bg
    refined = (band * B_band + (1.0 - band) * bokeh_phys).clamp(0.0, 1.0)
    return refined, alpha


def find_boundary_zoom(mask, size: int = 192):
    """选误差图响应最强的 size² 窗口（细化网最活跃处=最值得放大核对的边界）。"""
    import torch.nn.functional as F
    score = F.avg_pool2d(mask[None], kernel_size=size, stride=size // 4)[0, 0]
    iy, ix = divmod(int(score.argmax()), score.shape[1])
    y0 = min(iy * (size // 4), mask.shape[1] - size)
    x0 = min(ix * (size // 4), mask.shape[2] - size)
    return y0, x0, size


# ==============================================================================
# 2) 主流程
# ==============================================================================
def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from depth.estimator import load_depth_backend, save_disparity_visualization
    from optics.aberrations import PRESETS
    from refine.network import RefineNet
    from render.demo import auto_focus_params, load_image
    from render.renderer import RenderControl, render_tiled, snap_disparity_edges

    ap = argparse.ArgumentParser(description="真实照片端到端 demo（M2 收官件）")
    ap.add_argument("image", nargs="?", default=str(DEFAULT_IMG))
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--presets", type=str, nargs="+", default=E2E_PRESETS)
    ap.add_argument("--max-width", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="视差去压缩幂律(D39)；<1 拉开远景虚化(慎用,会虚主体)，=1 关闭(默认)")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt) if args.ckpt else find_default_ckpt()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[e2e] device={device}  image={args.image}  ckpt={ckpt_path}")

    # ---- 1) 细化网：加载训练好的权重，冻结推理 ----
    # P1 架构标记（train.py 存入 ckpt）：v2(extra_cond_ch>0)→整图单前向；
    # 旧 v1 ckpt（无该字段）→ 滑窗路径，保持可复现。
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    extra_ch = int(ck.get("extra_cond_ch", 0))
    use_gate = bool(ck.get("mask_edge_gate", False))
    matte_mode = bool(ck.get("matte_mode", False))
    # P2a（D32）：按 checkpoint 记录的 block/宽度重建（旧 ckpt 无此键 → 回落 v1 默认）。
    block_type = str(ck.get("block_type", "film_res"))
    ar_mid = int(ck.get("ar_mid", 96))
    iu_mid = int(ck.get("iu_mid", 48))
    net = RefineNet(extra_cond_ch=extra_ch, matte_mode=matte_mode,
                    block_type=block_type, ar_mid=ar_mid, iu_mid=iu_mid).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    arch = ("Plan B matte 选择式重渲" if matte_mode
            else "v2 整图单前向" if extra_ch else "v1 滑窗")
    print(f"[e2e] 细化网就绪（训练 {ck['iter']} 步，{ck['n_params']:,} 参数，"
          f"{arch}，门控={use_gate}）")

    # ---- 2) 深度：DA V2-Small → 视差 → 边缘吸附（与训练输入侧同路径）----
    rgb = load_image(Path(args.image), max_width=args.max_width)
    backend = load_depth_backend("depth_anything_v2", device=device)
    disp_np = backend.infer_disparity((rgb * 255).astype("uint8"))
    image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device)
    disparity = snap_disparity_edges(torch.from_numpy(disp_np).to(device))
    # 去压缩重映射（D39）：**默认关闭**（γ=1）；好深度下施 γ<1 反而虚掉主体。仅手动开。
    from render.demo import remap_disparity
    disparity = remap_disparity(disparity, getattr(args, "gamma", 1.0))
    # 高光核心深度统一（D43）：消除小灯珠深度跳变造成的嵌套同心环（与 render.demo 一致）。
    from render.renderer import unify_highlight_core_depth
    disparity, n_uni = unify_highlight_core_depth(image, disparity)
    if n_uni > 0:
        print(f"[e2e] 高光核心深度统一: {n_uni} 个灯珠核心 → 消嵌套环")
    save_disparity_visualization(disparity.cpu().numpy(),
                                 OUT_DIR / "e2e_disparity.png")

    # ---- 3) 对焦/光圈自动估计（与 render.demo 同逻辑）----
    d_f, tol, K = auto_focus_params(disparity)
    print(f"[e2e] 对焦 d_f={d_f:.3f}±{tol:.3f}  K={K:.1f}")

    # ---- 4) 各预设：物理渲染 → 滑窗细化 → 对照 ----
    rows = []
    for name in args.presets:
        coeffs = PRESETS[name]
        ctrl = RenderControl(focus_disparity=d_f, aperture_K=K,
                             focus_tolerance=tol, coeffs=coeffs,
                             # D45：gamma 回归标准 2.2（默认），保留球差盘内对比（γ=4 压平亮边/亮心）；
                             # 不用 gain 放大（从 LDR 猜测放大致溢出+非点光源冒假弥散圆）。
                             highlight_gain=0.0)
        t0 = time.time()
        with torch.no_grad():
            phys = render_tiled(image, disparity, ctrl).clamp(0.0, 1.0)
        t_phys = time.time() - t0

        # 13 维 ctrl_vec（顺序=make_sample 接口约定；末两维每窗覆写）。
        base_ctrl = [d_f, K / 100.0, coeffs.W040_spherical, coeffs.W131_coma,
                     coeffs.W222_astigmatism, coeffs.W220_field_curv,
                     coeffs.n_blades / 10.0, coeffs.vignette_strength,
                     coeffs.vignette_radius, coeffs.loca_rgb[0],
                     coeffs.laca_rgb[0], 0.0, 0.0]
        t0 = time.time()
        if matte_mode:
            refined, mask = refine_full_image_matte(net, image, disparity, phys,
                                                    base_ctrl, ctrl, use_gate)
        elif extra_ch > 0:
            refined, mask = refine_full_image_v2(net, image, disparity, phys,
                                                 base_ctrl, ctrl, use_gate)
        else:
            refined, mask = refine_full_image(net, image, disparity, phys,
                                              base_ctrl)
        t_ref = time.time() - t0
        print(f"[e2e] {name:14s} 渲染 {t_phys:5.2f}s + 细化 {t_ref:5.2f}s  "
              f"|refined−phys| 均值={float((refined - phys).abs().mean()):.4f}  "
              f"mask 均值={float(mask.mean()):.3f}")

        for tag, t in [("phys", phys), ("refined", refined), ("mask", mask[0])]:
            arr = t.cpu().numpy()
            arr = arr.transpose(1, 2, 0) if arr.ndim == 3 else arr
            plt.imsave(str(OUT_DIR / f"e2e_{name}_{tag}.png"), arr.clip(0, 1),
                       **({} if arr.ndim == 3 else
                          dict(cmap="Spectral_r", vmin=0, vmax=1)))
        rows.append((name, phys, refined, mask))

    # ---- 5) 总对比面板：整图 + 误差图最活跃处的放大窗 ----
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(22, 3.6 * n), squeeze=False)
    for i, (name, phys, refined, mask) in enumerate(rows):
        y0, x0, sz = find_boundary_zoom(mask)
        panels = [
            ("input", image), (f"{name}: B_phys", phys),
            (f"{name}: refined", refined), ("error mask m", mask[0]),
            ("zoom B_phys", phys[:, y0:y0 + sz, x0:x0 + sz]),
            ("zoom refined", refined[:, y0:y0 + sz, x0:x0 + sz]),
        ]
        for ax, (title, t) in zip(axes[i], panels):
            arr = t.cpu().numpy()
            if arr.ndim == 3:
                ax.imshow(arr.transpose(1, 2, 0).clip(0, 1))
            else:
                ax.imshow(arr.clip(0, 1), cmap="Spectral_r", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle(f"end-to-end: DA V2 -> physical render -> refine "
                 f"(ckpt @ {ck['iter']} iters)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "e2e_grid.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[e2e] 总对比面板 -> {OUT_DIR / 'e2e_grid.png'}")
    print("[e2e] 请肉眼核对：①zoom 窗里主体边界的渗色/硬边应比 B_phys 干净；")
    print("      ②mask 应集中在深度边界；③平坦区/散景盘风格两列应一致（不被洗掉）。")


if __name__ == "__main__":
    main()
