"""
refine/conditioning.py
======================
细化网 v2 (P1) 的【物理条件构建】—— 三件事，全部是"物理可计算量"（设计红线
NETWORK_DESIGN §1：信息流向只许 physics → network）：

1. **空间物理条件图（P1a）**：逐像素的 [像高 H, sin方位角, cos方位角] 三通道，
   替代 v1 里"整块一个标量走 FiLM"的近似 → 训练=推理同口径，整图单次前向，
   D20 的滑窗/Hann 融合退役。

2. **PSF 描述子图（P1b，论文级创新点）**：渲染器是非盲的——每层 PSF 自己算的，
   顺手提取低维解析描述子（等效半径/拉伸/取向/亮边环比/透过率），再按每个像素的
   (视差→离焦, 像高) 在描述子表上插值成 6 通道逐像素图。网络由此"看见"
   此处散景核长什么样，边界修复策略可以按核形状分化。
   ⚠️ 与 D9"双环鬼影"的区别：这里插值的是【标量统计量】（随离焦/像高平滑单调，
   插值安全），绝不插值 PSF 图像本身。
   描述子与 optics/decoupling.py 的解耦签名同一套物理词汇表（"签名三位一体"，
   NETWORK_DESIGN §7 创新点④）。

3. **边界带门控（找补对策 b，见 PROJECT_STATUS §6 [2026-06-12]）**：
   误差图 m 的活动范围被物理限制在"视差边缘带"内——边带 = 视差跳变处向外扩张
   最大 CoC 半径（渗色的物理传播距离上限）。50k 核图发现网络在真实照片的
   非边界纹理区大面积点亮 mask、把虚化前景"找补"回锐利纹理；门控从结构上
   禁止这件事（红线检查：边带是纯物理量，network 拿不到任何"画风格"的通道）。

通道布局（拼到细化网 guide 的末尾，refine/network.py 的 extra_cond_ch）：
    [0] H_map      逐像素归一化像高（0=画面中心，1=半对角线）
    [1] sin_az     逐像素方位角 sin（sin/cos 编码避免 2π 跳变、旋转连续）
    [2] cos_az     逐像素方位角 cos
    [3] r_eq       PSF 等效半径 / 128（≈ 本地 CoC 半径，与 v1 的 r̃ 提示同尺度）
    [4] elong      log(σ_major/σ_minor) —— 猫眼/像散的拉伸度（旋转不变量）
    [5] aniso_c    各向异性张量 cos2φ 分量（已转到该像素的实际方位）
    [6] aniso_s    各向异性张量 sin2φ 分量（同上；幅度=拉伸强度，盘各向同性时→0，
                   天然解决"圆盘取向无定义"的噪声问题——比裸 sin2φ/cos2φ 干净）
    [7] ring       外环/内盘亮度比 ring/(1+ring) ∈ [0,1)（肥皂泡/奶油形态）
    [8] T          relative_transmission(H)（猫眼边角失光）

运行自检（描述子表单调性 + 条件图可视化）：
    python -m refine.conditioning      # → outputs/refine_test/conditioning.png
"""

from __future__ import annotations

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "refine_test"


# ==============================================================================
# 1) 单个 PSF 的解析描述子（与 optics/decoupling.py 的签名同一套统计量）
# ==============================================================================
def psf_stats(psf, eps: float = 1e-12):
    """从单个归一化 PSF [N,N]（sum=1）提取 5 个解析描述子。

    Returns (全是 python float):
        r_eq     等效半径(px)：均匀圆盘半径口径，r_eq = √(2·(sxx+syy)/2) ≈ CoC 半径的
                 0.7 倍量级；上层除以 128 归一化（标定上限，见 CLAUDE.md 第 3 节）。
        elong    log(σ_major/σ_minor) ≥ 0：二阶矩张量特征值之比的半对数（拉伸强度）。
        aniso_c  (sxx−syy)/(sxx+syy)：各向异性张量的 cos2φ 分量（+x 约定坐标系）。
        aniso_s  2·sxy/(sxx+syy)：sin2φ 分量。
                 (aniso_c, aniso_s) = 拉伸强度 × (cos2φ, sin2φ)——取向以 2φ 编码
                 （拉伸轴是 mod π 的"轴"不是"向量"），且各向同性时自动→0，
                 不会像裸角度那样在圆盘上输出随机噪声。
        ring     外环(0.75R~1.05R)/内盘(<0.5R) 平均亮度比（decoupling 同款定义）。
    """
    import torch
    h, w = psf.shape[-2:]
    ys = torch.arange(h, device=psf.device, dtype=psf.dtype)
    xs = torch.arange(w, device=psf.device, dtype=psf.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx = (psf * xx).sum()
    cy = (psf * yy).sum()
    sxx = (psf * (xx - cx) ** 2).sum()
    syy = (psf * (yy - cy) ** 2).sum()
    sxy = (psf * (xx - cx) * (yy - cy)).sum()

    trace = sxx + syy
    r_eq = float(trace.sqrt())                       # = √(sxx+syy)，decoupling 的 R 同款
    # 特征值 λ± = trace/2 ± √((sxx−syy)²/4 + sxy²) → 拉伸强度与取向。
    lam_half = ((sxx - syy) ** 2 / 4.0 + sxy ** 2).sqrt()
    lam_p, lam_m = trace / 2.0 + lam_half, trace / 2.0 - lam_half
    elong = 0.5 * math.log(float(lam_p / (lam_m + eps)) + eps) if float(trace) > eps else 0.0
    aniso_c = float((sxx - syy) / (trace + eps))
    aniso_s = float(2.0 * sxy / (trace + eps))

    # 亮边环比：近似 delta 的小 PSF（合焦层）没有"环"可言，取中性值 1.0 防止
    # 空索引/除零把噪声当信号。
    if r_eq < 3.0:
        ring = 1.0
    else:
        r = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
        outer = psf[(r > 0.75 * r_eq) & (r < 1.05 * r_eq)]
        inner = psf[r < 0.5 * r_eq]
        ring = float(outer.mean() / (inner.mean() + eps)) if outer.numel() > 0 else 1.0
    return r_eq, elong, aniso_c, aniso_s, ring


# ==============================================================================
# 2) 描述子表：(离焦层 × 像高桶) → 6 维描述子（+x 约定）
# ==============================================================================
def descriptor_table(ctrl, H_centers, device, dtype=None):
    """对渲染器同款的离焦分层 × 给定像高桶，现场算 PSF 并提取描述子表。

    PSF 走【单色】路径（complex_pupil + pupil_to_psf，不做光谱平均/RGB）：
    描述子只是条件词汇，不是精确测量；单色 1 次 FFT/项（vs rgb_psf 的 9 次），
    且与 optics/decoupling.py 的签名口径一致（D11 同样用单色）。
    成本 ≈ n_layers × len(H_centers) 个 256² FFT，训练时逐样本算也只有几毫秒级。

    与实际渲染的一致性：离焦量经同一套标定换算（coc_to_w020），并应用同样的
    balance_spherical_focus 焦移补偿——描述子描述的是"真的会被渲染出来的核"。

    Args:
        ctrl: RenderControl（取 d_f/K/tol/coeffs/n_layers/pupil_size/balance）。
        H_centers: 像高桶中心列表（必须均匀间隔，插值用 tent 权重）。
    Returns:
        [n_layers, len(H_centers), 6] 张量，最后一维 =
        [r_eq/128, elong, aniso_c, aniso_s, ring/(1+ring), T(H)]。
    """
    import torch
    from optics import pupil as pupil_mod
    from optics.psf import pupil_to_psf
    from render.renderer import (coc_to_w020, load_defocus_calibration,
                                 signed_coc)

    slope, intercept = load_defocus_calibration()
    grid = pupil_mod.make_pupil_grid(ctrl.pupil_size, device=str(device),
                                     dtype=dtype)
    centers = torch.linspace(0.0, 1.0, ctrl.effective_n_layers())     # 与 layer_weights 同款层中心
    table = torch.zeros(ctrl.effective_n_layers(), len(H_centers), 6,
                        device=device, dtype=dtype or torch.float32)
    with torch.no_grad():
        # T(H) 与离焦无关，每个 H 桶只算一次。
        T_h = [float(pupil_mod.relative_transmission(grid, float(h), ctrl.coeffs))
               for h in H_centers]
        for l in range(ctrl.effective_n_layers()):
            r_l = signed_coc(float(centers[l]), float(ctrl.focus_disparity),
                             float(ctrl.aperture_K), float(ctrl.focus_tolerance))
            w020 = coc_to_w020(r_l, slope, intercept, ctrl.pupil_size)
            if ctrl.balance_spherical_focus:
                w020 = w020 - ctrl.coeffs.W040_spherical   # 与 _layer_psf 同款焦移补偿
            c_l = ctrl.coeffs.replace(
                W020_defocus=ctrl.coeffs.W020_defocus + w020)
            for j, h in enumerate(H_centers):
                P = pupil_mod.complex_pupil(grid, float(h), c_l, channel=None)
                psf = pupil_to_psf(P, crop=None, check_sampling=False)
                r_eq, elong, a_c, a_s, ring = psf_stats(psf)
                table[l, j, 0] = r_eq / 128.0
                table[l, j, 1] = elong
                table[l, j, 2] = a_c
                table[l, j, 3] = a_s
                table[l, j, 4] = ring / (1.0 + ring)
                table[l, j, 5] = T_h[j]
    return table


def _tent_weights(value_map, centers):
    """对【均匀间隔】中心列表算逐像素 tent 插值权重 [n, H, W]（Σ_n = 1）。

    与 renderer.layer_weights 同款的线性 tent；n=1 时退化为全 1（无插值）。
    """
    import torch
    n = len(centers)
    v = value_map[None]                                   # [1,H,W]
    if n == 1:
        return torch.ones_like(v)
    c = torch.as_tensor(centers, device=value_map.device, dtype=value_map.dtype)
    delta = float(c[1] - c[0])
    w = (1.0 - (v - c[:, None, None]).abs() / max(delta, 1e-12)).clamp(min=0.0)
    return w / (w.sum(dim=0, keepdim=True) + 1e-12)


# ==============================================================================
# 3) 逐像素条件图（P1a + P1b 的主入口）
# ==============================================================================
def condition_maps(disparity, ctrl, H_map, az_map, H_centers=None,
                   with_descriptors: bool = True):
    """构建 9 通道（或关掉描述子时 3 通道）逐像素物理条件图。

    描述子的逐像素化 = 双线性插值：
      离焦轴：像素视差 → renderer.layer_weights 同款 tent 权重（n_layers 桶）
              ——与物理渲染分层【完全同一套】权重，条件和渲染口径严格一致；
      像高轴：H_map → H_centers 上的 tent 权重。
    取向通道 (aniso_c, aniso_s) 表里存的是 +x 约定，按像素方位角旋转 2·az 到
    实际朝向（拉伸轴 mod π → 用 2φ 编码旋转，跨画幅中心的 ±π 翻转自动无感）。

    Args:
        disparity: [H,W] 网络输入侧视差（扰动/估计，与喂给细化网的同一张）。
        ctrl: RenderControl。
        H_map / az_map: [H,W] 逐像素像高与方位角（训练=虚拟画幅几何，推理=真实几何）。
        H_centers: 像高桶中心（None → 单桶取 H_map 均值；必须均匀间隔）。
        with_descriptors: False 时只输出 P1a 的 3 个空间通道（消融用）。
    Returns:
        [9,H,W]（或 [3,H,W]）条件图，dtype/device 同 disparity。
    """
    import torch
    from render.renderer import layer_weights

    spatial = torch.stack([H_map, torch.sin(az_map), torch.cos(az_map)])
    if not with_descriptors:
        return spatial

    if H_centers is None:
        H_centers = [float(H_map.mean())]
    table = descriptor_table(ctrl, H_centers, device=disparity.device,
                             dtype=disparity.dtype)        # [L,nH,6]
    w_l, _ = layer_weights(disparity, ctrl.effective_n_layers())       # [L,H,W] 离焦轴 tent
    w_h = _tent_weights(H_map, H_centers)                  # [nH,H,W] 像高轴 tent
    # 双轴插值：desc[c] = Σ_l Σ_h table[l,h,c]·w_l·w_h。按 H 桶循环避免大中间张量。
    desc = torch.zeros(6, *disparity.shape, device=disparity.device,
                       dtype=disparity.dtype)
    for j in range(len(H_centers)):
        # [L,6]ᵀ·[L,HW] → [6,HW]：单个 H 桶的离焦插值，再乘该桶的像高权重。
        d_j = torch.einsum("lc,lyx->cyx", table[:, j], w_l)
        desc = desc + d_j * w_h[j:j + 1]

    # 取向旋转：(aniso_c, aniso_s) 按 2·az 旋转（轴量 mod π 的标准变换）。
    c2, s2 = torch.cos(2.0 * az_map), torch.sin(2.0 * az_map)
    a_c, a_s = desc[2].clone(), desc[3].clone()
    desc[2] = a_c * c2 - a_s * s2
    desc[3] = a_c * s2 + a_s * c2
    return torch.cat([spatial, desc], dim=0)


def field_geometry(shape, device, dtype=None, H_field: float = None,
                   azimuth: float = None, half_diag: float = None):
    """生成逐像素 (H_map, az_map)。

    两种用法：
    - 推理整图：只传 shape → 以图像自身中心/半对角线为准（与 render_tiled 同约定）。
    - 训练裁块：传 (H_field, azimuth, half_diag) → 把裁块放进"虚拟画幅"：
      裁块中心位于虚拟画幅中心沿 azimuth 方向、距离 H_field·half_diag 处，
      逐像素 H/方位角按【精确几何】算（不是沿径向的线性近似）——跨画幅中心的
      裁块（H_map 在块内过 0）也是对的。

    Returns:
        (H_map [H,W] ∈[0,1]，az_map [H,W] 弧度（图像坐标 y 向下，与 render_tiled 一致）)
    """
    import torch
    Hh, Ww = shape
    ys = torch.arange(Hh, device=device, dtype=dtype or torch.float32)
    xs = torch.arange(Ww, device=device, dtype=dtype or torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    if H_field is None:
        # 推理整图：画幅 = 图像本身。
        cy, cx = (Hh - 1) / 2.0, (Ww - 1) / 2.0
        hd = math.sqrt(cy ** 2 + cx ** 2)
    else:
        # 训练裁块：虚拟画幅中心 = 裁块中心 − H_field·half_diag·(cos az, sin az)。
        hd = float(half_diag)
        pcy, pcx = (Hh - 1) / 2.0, (Ww - 1) / 2.0
        cy = pcy - H_field * hd * math.sin(azimuth)        # y 向下，sin 对 y
        cx = pcx - H_field * hd * math.cos(azimuth)
    dy, dx = yy - cy, xx - cx
    H_map = ((dy ** 2 + dx ** 2).sqrt() / hd).clamp(0.0, 1.0)
    az_map = torch.atan2(dy, dx)
    return H_map, az_map


# ==============================================================================
# 4) 边界带门控（找补对策 b）
# ==============================================================================
def boundary_band(disparity, ctrl, edge_thresh: float = 0.04,
                  pad_px: int = 8, soften: int = 7,
                  focus_gate: bool = True, focus_frac: float = 0.25):
    """视差边缘带 ∈ [0,1]：误差图 m 的物理活动范围（m ← m·band）。

    物理依据：分层渲染的边界瑕疵（渗色/硬边/半透明圈）只可能出现在
    "视差跳变处 ± 本图最大 CoC 半径"内——CoC 半径就是模糊的传播距离上限。
    带外强制 m=0 → 网络在平坦/纹理区无法启用神经分支，从结构上杜绝
    "把虚化纹理找补回锐利"的红线违规（PROJECT_STATUS §6 [2026-06-12] 的根因②）。

    ⚠️ 焦邻近门（D23，找补对策 b 的强化）：只看"视差跳变"对【密集碎边缘的远背景】
    （成片树枝/草丛）失效——那里处处跳变，边带≈全图，门形同虚设，网络照样把
    虚化的背景纹理找补回锐利（train_run3 真实图核图实测）。真正需要神经修复的
    遮挡边界，至少一侧应贴近焦区（清晰/浅虚化的主体）；而找补的害处全在
    【远离焦区、|CoC| 很大】的深背景碎边缘。故叠加一道焦邻近门：边缘还要落在
    "邻域内存在清晰内容（|CoC| < focus_frac·最大|CoC|）"处才放行。远背景碎边缘
    （两侧都重度虚化）→ 焦门=0 → 强制回落 B_phys（虚化），从结构上掐断背景找补。

    实现：5×5 窗口的视差极差 > edge_thresh 判为边缘（snap_disparity_edges 同款
    判据）→ 1/4 分辨率上 max-pool 扩张（半径 = 最大|CoC| + pad，省算力且带边
    精度无关紧要——这是宽松的门，不是精确分割）→ 焦邻近门同样扩张 rad 后相乘
    → 上采样 + 均值滤波软化（避免硬门在融合输出上印出接缝）。

    Args:
        disparity: [H,W] 网络输入侧视差。
        ctrl: RenderControl（算最大 |CoC| 用 d_f/K/tol）。
        focus_gate: 是否叠加焦邻近门（默认开；关掉=退回纯边缘带的旧行为）。
        focus_frac: 清晰判据阈值 = focus_frac × 图内最大|CoC|（越小越严，越压背景）。
    Returns:
        [1,H,W] 软门控图（边带≈1，远离边界→0）。
    """
    import torch
    import torch.nn.functional as F
    from render.renderer import signed_coc

    d = disparity[None, None]                              # [1,1,H,W]
    mx = F.max_pool2d(d, 5, stride=1, padding=2)
    mn = -F.max_pool2d(-d, 5, stride=1, padding=2)
    edge = ((mx - mn) > edge_thresh).to(d.dtype)

    # 扩张半径 = 图内实际出现的最大 |CoC|（≤ 标定上限 128px）+ 余量。
    r_map = signed_coc(disparity, float(ctrl.focus_disparity),
                       float(ctrl.aperture_K), float(ctrl.focus_tolerance))
    rad_max = min(float(r_map.abs().max()), 128.0)
    rad = int(rad_max) + pad_px
    k4 = 2 * math.ceil(rad / 4) + 1

    # 1/4 分辨率扩张：max-pool 下采样保边缘 → 大核 max-pool → 最近邻还原。
    e4 = F.max_pool2d(edge, 4, stride=4)
    e4 = F.max_pool2d(e4, k4, stride=1, padding=k4 // 2)
    band = F.interpolate(e4, size=d.shape[-2:], mode="nearest")

    if focus_gate:
        # 焦邻近门：清晰区（|CoC| 小）同样扩张 rad（清晰物体的渗色传播范围），
        # 与边缘带相乘 → 只保留"贴近清晰内容"的那部分边缘。
        focus_px = max(float(pad_px), focus_frac * rad_max)
        focus = (r_map.abs() < focus_px).to(d.dtype)[None, None]
        f4 = F.max_pool2d(focus, 4, stride=4)
        f4 = F.max_pool2d(f4, k4, stride=1, padding=k4 // 2)
        focus_band = F.interpolate(f4, size=d.shape[-2:], mode="nearest")
        band = band * focus_band

    band = F.avg_pool2d(band, soften, stride=1, padding=soften // 2)
    return band[0].clamp(0.0, 1.0)                         # [1,H,W]


# ==============================================================================
# 5) 自检：描述子物理合理性 + 条件图可视化
# ==============================================================================
def _smoke_test():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from optics.aberrations import PRESETS
    from render.renderer import RenderControl

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ① 描述子表物理合理性（soap_bubble + 猫眼预设，H 两桶）。
    ctrl = RenderControl(focus_disparity=0.8, aperture_K=40.0,
                         coeffs=PRESETS["swirl_catseye"])
    table = descriptor_table(ctrl, [0.0, 0.8], device=device)
    r0 = table[:, 0, 0]                                    # H=0 桶的 r_eq 序列
    # r_eq 应随 |离焦| 单调：层中心离 d_f 越远半径越大（两侧分别检查）。
    import numpy as np
    centers = np.linspace(0, 1, ctrl.effective_n_layers())
    far = r0[centers < 0.8 - 0.05]                          # 背景侧（离焦增大方向为远离）
    assert all(far[i] >= far[i + 1] - 1e-3 for i in range(len(far) - 1)), \
        "r_eq 未随背景侧离焦单调"
    # 猫眼：H=0.8 桶的拉伸强度应明显大于 H=0 桶（离焦大的层）。
    el0, el1 = float(table[2, 0, 1]), float(table[2, 1, 1])
    print(f"[cond] r_eq(H=0) 背景侧单调 ✓   elong: H=0 {el0:.3f} vs H=0.8 {el1:.3f}"
          f"（猫眼应使后者更大）")
    assert el1 > el0, "猫眼拉伸未在高像高桶显现"
    # T(H) 单调降。
    assert float(table[0, 1, 5]) <= float(table[0, 0, 5]) + 1e-6, "T(H) 未随像高下降"

    # ② 条件图 + 边界带可视化（合成样本）。
    from data.synth import SynthBokehDataset, SynthConfig
    ds = SynthBokehDataset(SynthConfig(device=device, seed=7))
    s = ds[0]
    m = s["meta"]
    H_map, az_map = m["H_map"], m["az_map"]
    cm = condition_maps(s["disparity"], m["ctrl"], H_map, az_map,
                        H_centers=m["H_centers"])
    band = boundary_band(s["disparity"], m["ctrl"])
    names = ["H_map", "sin_az", "cos_az", "r_eq/128", "elong",
             "aniso_c", "aniso_s", "ring_n", "T(H)"]
    fig, axes = plt.subplots(2, 6, figsize=(20, 6.5))
    axes = axes.ravel()
    axes[0].imshow(s["image"].cpu().numpy().transpose(1, 2, 0))
    axes[0].set_title("input", fontsize=9)
    axes[1].imshow(s["disparity"].cpu().numpy(), cmap="Spectral_r")
    axes[1].set_title("disparity (perturbed)", fontsize=9)
    for i in range(9):
        im = axes[2 + i].imshow(cm[i].cpu().numpy(), cmap="viridis")
        axes[2 + i].set_title(names[i], fontsize=9)
        plt.colorbar(im, ax=axes[2 + i], shrink=0.7)
    axes[11].imshow(band[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[11].set_title("boundary band (mask gate)", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("refine/conditioning.py smoke — spatial + PSF descriptor maps "
                 "+ boundary band", fontsize=12)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "conditioning.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[cond] 条件图形状 {tuple(cm.shape)} ✓  可视化 -> {out}")
    print("[cond] 请肉眼核对：r_eq 图应≈|CoC| 分布（对焦层暗、离焦层亮）；"
          "边界带应贴着前景轮廓一圈、平坦区为 0。")


if __name__ == "__main__":
    _smoke_test()
