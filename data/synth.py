"""
data/synth.py
=============
【在线合成流水线】训练数据的来源（CLAUDE.md 第 7 节）。

每个样本 = 透明前景图层（程序化生成）+ 全焦背景（真实照片裁块），
按【随机采样的像差向量 a】现场渲染成 (全焦图, 散景 GT) 配对：

    ┌ 已知量（合成时精确掌握）──────────────────────────────┐
    │  每层 (内容, alpha, 视差)  +  控制向量 c = (d_f, K, a)  │
    │  + 裁块视场位置 (H_field, azimuth)                      │
    └─────────────────────────────────────────────────┘
         │                                  │
         ▼ 精确分层合成（renderer.            ▼ alpha 直接叠出
           composite_blurred_layers）          全焦图 + 真视差图
         │                                  │
       散景 GT B*                       网络输入 I + 【加扰动的】视差 D̃
                                        （扰动模拟 DA V2 的估计误差！）

【GT 与输入分离 —— 本文件最重要的设计（CLAUDE.md 第 7 节，BokehMe 的关键 trick）】
GT 用已知 alpha 的精确分层合成；网络输入侧的视差故意加扰动（边界形变/模糊/低频噪声）。
不扰动 → 细化网在真实深度误差下的边界必崩（sim-to-real gap）。

【素材策略（v1）】
- 前景：程序化生成（低频傅里叶轮廓的随机闭合形状 + 噪声纹理 + 软边 alpha）。
  好处：无需外部抠图数据集、形状边界无穷多样；以后可无缝换成真实抠图素材
  （接口只要求 (rgb, alpha)）。
- 背景：真实照片随机裁块（默认池子=DA V2 示例图 + BokehMe 输入图，**占位**，
  正式训练应换成大相册目录——改 SynthConfig.bg_dirs 即可）。
- HDR 高光：随机贴小亮斑（接近过曝），配合渲染器的 highlight_gain 涌现亮斑散景。

【视场相关训练的省钱技巧（与 renderer.composite_blurred_layers 配套）】
训练裁块小（512px），块内像高 H 近似常数 → 每个样本采样一个 (H_field, azimuth)
当作"该裁块在画幅上的位置"，用单个旋转后的 PSF 渲染整块。与 tile 近似（D8）物理
一致，让猫眼/彗差/像散/LaCA 也拿到廉价监督；(H, azimuth) 同时写进控制向量喂给细化网。

运行可视化自检：
    python -m data.synth          # 生成 4 个样本 → outputs/synth_test/samples.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "synth_test"

# 占位背景池：正式训练换成大相册目录（SynthConfig.bg_dirs）。
_DEFAULT_BG_DIRS = [
    PROJECT_ROOT / "third_party" / "Depth-Anything-V2" / "assets" / "examples",
    PROJECT_ROOT / "third_party" / "BokehMe" / "inputs",
]


# ==============================================================================
# 0) 配置
# ==============================================================================
@dataclass
class SynthConfig:
    crop: int = 512                  # 训练裁块边长（CLAUDE.md 第 5 节：512 起步）
    n_fg_range: tuple = (1, 3)       # 前景层数范围
    bg_dirs: list = field(default_factory=lambda: [str(p) for p in _DEFAULT_BG_DIRS])
    # 视差分配：背景远(小)、前景近(大)，两区间隔开避免层间歧义。
    bg_disp_range: tuple = (0.05, 0.35)
    fg_disp_range: tuple = (0.45, 0.95)
    # 光圈：最大 |CoC| 像素半径的采样范围（按对焦位置反推 K）。
    max_coc_range: tuple = (12.0, 48.0)
    # 对焦：约半数样本对到某前景层、其余对到背景（前/背景虚化都要见过）。
    p_focus_fg: float = 0.5
    # HDR 高光贴片
    p_highlight: float = 0.7         # 样本带高光点的概率
    n_highlight_range: tuple = (3, 12)
    highlight_gain: float = 4.0      # 与渲染器 ctrl.highlight_gain 一致
    # 细薄前景（找补对策 a，PROJECT_STATUS §6 [2026-06-12]）：真实照片的树枝/
    # 栅栏/发丝对网络而言"处处是边界"，v1 合成前景全是光滑 blob、从未见过
    # 这种结构 → 网络按边界策略把虚化细枝抄回锐利输入。按此概率把前景层
    # 换成程序化细薄结构，让"细薄物保持虚化"拿到精确 GT 监督。
    p_thin_fg: float = 0.5           # 每个前景层为细薄结构的概率
    # 背景细薄结构（找补【根因】修正，D23）：上面的 p_thin_fg 只把细薄结构放在
    # 近前景，对焦前景时它们 GT 清晰 → 网络学成"细薄=该锐化"，真实图里背景树枝
    # 于是被找补回锐利（train_run3 实测）。这里在背景侧（视差 < bg_disp，更远）
    # 插入细薄层并强制对焦前景 → 它们重度虚化、GT 也虚化，纠正"细薄→锐化"的
    # 泛化偏差，教会网络"背景深处的细薄结构应保持虚化"。
    p_thin_bg: float = 0.4           # 样本含背景细薄层的概率
    n_bg_thin_range: tuple = (1, 2)  # 背景细薄层数（含端点）
    # 虚拟画幅几何（P1a 第二步，NETWORK_DESIGN §4.1）：裁块放进虚拟画幅，
    # half_diag = crop × factor（log-uniform 采样）→ 块内像高跨度
    # span ≈ (crop/2)/half_diag ∈ [0.06, 0.5]，覆盖"整图小窗"到"近乎全幅"。
    half_diag_factor: tuple = (1.0, 8.0)
    # 块内 H 跨度 < t1 → 单 H 渲染（视场近似恒定，与 v1 一致）；
    # ∈[t1,t2) → 2 子带；≥t2 → 3 子带（渲染成本 ∝ 子带数，按需付费）。
    subband_span_thresh: tuple = (0.12, 0.30)
    # 视差扰动强度（模拟 DA V2 误差）
    perturb_blur_range: tuple = (3, 15)     # 高斯模糊核（奇数范围）
    perturb_morph_range: tuple = (0, 5)     # 边界膨胀/腐蚀像素
    perturb_noise_amp: float = 0.04         # 低频噪声幅度
    device: str = "cuda"
    seed: int | None = None


# ==============================================================================
# 1) 程序化前景：随机闭合形状 + 纹理 + 软 alpha
# ==============================================================================
def _random_blob_alpha(size: int, rng, device):
    """低频傅里叶轮廓的随机闭合形状 → [1,size,size] 软边 alpha。

    极坐标半径函数 R(θ) = r0·(1 + Σ_k a_k·cos(kθ+φ_k))，k=2..5 只取低频
    → 形状光滑闭合不自交；alpha = sigmoid((R(θ)−r)/2)，约 2px 软边（抗锯齿）。
    """
    import torch
    cx = rng.uniform(0.25, 0.75) * size
    cy = rng.uniform(0.25, 0.75) * size
    r0 = rng.uniform(0.12, 0.30) * size
    ys = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    dx, dy = xx - cx, yy - cy
    r = (dx * dx + dy * dy).sqrt()
    theta = torch.atan2(dy, dx)
    R = torch.ones_like(theta)
    for k in range(2, 6):
        a_k = rng.uniform(0.0, 0.25 / k)
        phi = rng.uniform(0, 2 * math.pi)
        R = R + a_k * torch.cos(k * theta + phi)
    R = r0 * R
    return torch.sigmoid((R - r) / 2.0)[None]            # [1,S,S]


def _random_thin_alpha(size: int, rng, device):
    """细薄多边界结构 alpha [1,size,size]：树枝 / 草丛发丝 / 栅栏（三选一）。

    设计目标（找补对策 a）：制造"宽度仅 1~6px、边界密集"的前景——它们被虚化后
    在全焦输入里仍清晰可见，网络必须学会【不】把它们从输入里抄回来
    （GT 是精确分层合成的，虚化才是正确答案）。

    branch/grass 用 cv2 折线绘制（LINE_AA 自带 ~1px 软边），fence 用坐标场
    的软方波（无需外部素材，参数随机 → 形态无穷多样）。
    """
    import cv2
    import numpy as np
    import torch
    kind = ["branch", "grass", "fence"][int(rng.integers(0, 3))]

    if kind == "fence":
        # 平行条带：方向近竖直 ±45°，周期/占空比随机 → 软方波 alpha。
        theta = math.pi / 2 + rng.uniform(-math.pi / 4, math.pi / 4)
        period = float(rng.uniform(14, 48))
        width = period * float(rng.uniform(0.12, 0.4))
        ys = torch.arange(size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, ys, indexing="ij")
        proj = xx * math.cos(theta) + yy * math.sin(theta) + rng.uniform(0, period)
        frac = torch.remainder(proj, period)                  # [0,period)
        dist = (frac - period / 2).abs()                      # 到条带中心的距离(px)
        return torch.sigmoid((width / 2 - dist) / 1.0)[None]  # ~2px 软边

    canvas = np.zeros((size, size), np.float32)
    if kind == "branch":
        # 2~4 条主枝（从随机边缘出发的随机折线），宽度递减 + 1~2 层分叉 + 末端小叶。
        tips = []

        def draw_branch(p, ang, length, width, depth):
            steps = int(rng.integers(3, 6))
            for _ in range(steps):
                q = p + length / steps * np.array([math.cos(ang), math.sin(ang)])
                cv2.line(canvas, tuple(p.astype(int)), tuple(q.astype(int)),
                         1.0, max(int(round(width)), 1), cv2.LINE_AA)
                p, ang = q, ang + rng.uniform(-0.4, 0.4)
                if depth < 2 and rng.uniform() < 0.45:        # 分叉
                    draw_branch(p.copy(), ang + rng.uniform(-1.0, 1.0) * 0.9,
                                length * 0.6, width * 0.6, depth + 1)
            tips.append(p)

        for _ in range(int(rng.integers(2, 5))):
            edge = int(rng.integers(0, 4))                    # 出发边：上下左右
            t = rng.uniform(0, size)
            p0 = np.array([[t, 0.0], [t, size - 1.0], [0.0, t], [size - 1.0, t]][edge])
            ang0 = [math.pi / 2, -math.pi / 2, 0.0, math.pi][edge] + rng.uniform(-0.6, 0.6)
            draw_branch(p0, ang0, size * rng.uniform(0.4, 0.9),
                        rng.uniform(2.5, 6.0), 0)
        for p in tips:                                        # 末端小叶（椭圆簇）
            for _ in range(int(rng.integers(0, 4))):
                c = (p + rng.normal(0, 6, 2)).astype(int)
                cv2.ellipse(canvas, tuple(np.clip(c, 0, size - 1)),
                            (int(rng.integers(2, 7)), int(rng.integers(1, 4))),
                            float(rng.uniform(0, 180)), 0, 360, 1.0, -1, cv2.LINE_AA)
    else:  # grass：底边出发的一把细弯曲线（1~2px，发丝/草叶）
        x0s = rng.uniform(0, size, size=int(rng.integers(10, 28)))
        for x0 in x0s:
            n_pts = 8
            ts = np.linspace(0, 1, n_pts)
            bend = rng.uniform(-0.5, 0.5) * size * 0.4
            length = size * rng.uniform(0.4, 1.0)
            xs = x0 + bend * ts ** 2 + rng.normal(0, 1.5, n_pts).cumsum()
            ys = size - 1 - length * ts
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, 1.0,
                          int(rng.integers(1, 3)), cv2.LINE_AA)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.7)            # 统一 ~1px 软边（抗锯齿）
    return torch.tensor(canvas.clip(0.0, 1.0), device=device)[None]


def _random_texture(size: int, rng, device):
    """随机纹理 [3,size,size]：2~3 个随机颜色按低频噪声场混合（多样且零外部依赖）。"""
    import torch
    import torch.nn.functional as F
    n_colors = int(rng.integers(2, 4))
    colors = torch.tensor(rng.uniform(0.05, 0.95, size=(n_colors, 3)),
                          device=device, dtype=torch.float32)
    # 低频噪声场：低分辨率随机数 → bicubic 上采样 → softmax 当混合权重。
    low = int(rng.integers(4, 12))
    w = torch.tensor(rng.normal(0, 1, size=(n_colors, 1, low, low)),
                     device=device, dtype=torch.float32)
    w = F.interpolate(w, size=(size, size), mode="bicubic", align_corners=False)
    w = torch.softmax(w * 3.0, dim=0)                    # [C,1,S,S]
    tex = (w * colors[:, :, None, None]).sum(0)          # [3,S,S]
    return tex.clamp(0.0, 1.0)


def _paste_highlights(rgb, rng, n: int):
    """随机贴近过曝的小高斯亮斑（模拟点光源）；渲染时经 highlight_gain 变亮斑散景。"""
    import torch
    _, H, W = rgb.shape
    out = rgb.clone()
    ys = torch.arange(H, device=rgb.device, dtype=rgb.dtype)
    xs = torch.arange(W, device=rgb.device, dtype=rgb.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for _ in range(n):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        rad = rng.uniform(1.5, 4.0)
        tint = torch.tensor(rng.uniform(0.85, 1.0, size=3),
                            device=rgb.device, dtype=rgb.dtype)
        spot = torch.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
        out = torch.maximum(out, spot[None] * tint[:, None, None])
    return out


# ==============================================================================
# 2) 背景池
# ==============================================================================
class BackgroundPool:
    """从图片目录池随机裁背景块。初始化时扫描路径，按需读图（cv2）。"""

    def __init__(self, dirs: list, crop: int):
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.paths = [p for d in dirs for p in sorted(Path(d).glob("*"))
                      if p.suffix.lower() in exts]
        if not self.paths:
            raise FileNotFoundError(f"背景池为空，请检查目录：{dirs}")
        self.crop = crop

    def sample(self, rng, device):
        import cv2
        import torch
        # 大相册池里偶有损坏/不可读文件，重抽几次而不是让整夜训练崩掉。
        for _ in range(10):
            path = self.paths[int(rng.integers(0, len(self.paths)))]
            bgr = cv2.imread(str(path))
            if bgr is not None:
                break
        else:
            raise RuntimeError(f"背景池连续 10 次读图失败，最后一张：{path}")
        h, w = bgr.shape[:2]
        # 图太小先放大到能裁出 crop。
        scale = max(self.crop / h, self.crop / w, 1.0)
        if scale > 1.0:
            bgr = cv2.resize(bgr, (int(w * scale) + 1, int(h * scale) + 1),
                             interpolation=cv2.INTER_CUBIC)
            h, w = bgr.shape[:2]
        y0 = int(rng.integers(0, h - self.crop + 1))
        x0 = int(rng.integers(0, w - self.crop + 1))
        patch = bgr[y0:y0 + self.crop, x0:x0 + self.crop, ::-1]   # BGR→RGB
        return torch.tensor(patch.copy(), device=device,
                            dtype=torch.float32).permute(2, 0, 1) / 255.0


# ==============================================================================
# 3) 视差扰动（模拟 DA V2 的估计误差 —— 防 sim-to-real 边界崩坏的关键）
# ==============================================================================
def perturb_disparity(disp, rng, cfg: SynthConfig):
    """对真视差做三连扰动：边界形变(膨胀/腐蚀) → 高斯模糊 → 低频噪声。

    动机：DA V2 的误差形态主要是 ①物体边界不贴（差几个像素）②边缘过渡带发糊
    ③区域内低频漂移。三个扰动逐一对应。细化网必须在这种"不完美深度"上学会
    修边界，否则真实推理时必崩（CLAUDE.md 第 7 节）。
    """
    import torch
    import torch.nn.functional as F
    d = disp[None, None]                                  # [1,1,H,W]

    # ① 边界形变：随机膨胀或腐蚀（max-pool 实现）。
    m = int(rng.integers(cfg.perturb_morph_range[0], cfg.perturb_morph_range[1] + 1))
    if m > 0:
        k = 2 * m + 1
        if rng.uniform() < 0.5:
            d = F.max_pool2d(d, k, stride=1, padding=m)           # 膨胀（近景外扩）
        else:
            d = -F.max_pool2d(-d, k, stride=1, padding=m)         # 腐蚀（近景内缩）

    # ② 高斯模糊：边缘过渡带发糊（可分离卷积，横竖各一次）。
    kb = int(rng.integers(cfg.perturb_blur_range[0] // 2,
                          cfg.perturb_blur_range[1] // 2 + 1)) * 2 + 1
    sigma = kb / 4.0
    xs = torch.arange(kb, device=d.device, dtype=d.dtype) - kb // 2
    g1 = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    d = F.conv2d(d, g1[None, None, :, None], padding=(kb // 2, 0))
    d = F.conv2d(d, g1[None, None, None, :], padding=(0, kb // 2))

    # ③ 低频噪声漂移。
    low = int(rng.integers(3, 8))
    noise = torch.tensor(rng.normal(0, 1, size=(1, 1, low, low)),
                         device=d.device, dtype=d.dtype)
    noise = F.interpolate(noise, size=d.shape[-2:], mode="bicubic",
                          align_corners=False)
    d = d + cfg.perturb_noise_amp * noise
    return d[0, 0].clamp(0.0, 1.0)


# ==============================================================================
# 4) 样本生成主函数 + Dataset
# ==============================================================================
def make_sample(cfg: SynthConfig, bg_pool: BackgroundPool, rng):
    """生成一个训练样本（全部张量在 cfg.device 上）。

    Returns dict：
        image        [3,S,S]  全焦输入图（sRGB）
        disparity    [S,S]    【加扰动】视差（网络输入侧用这个！）
        disparity_gt [S,S]    真视差（调试/消融用，训练不喂网）
        bokeh_gt     [3,S,S]  散景 GT（已知 alpha 精确分层合成，sRGB）
        ctrl_vec     [13]     控制向量数值表示（FiLM 条件注入细化网，D15 后 13 维）
        meta         dict     采样到的对象（ctrl/H_field/azimuth/层视差）
    """
    import torch
    from optics.aberrations import sample_random
    from render.renderer import (RenderControl, composite_blurred_layers,
                                 linear_to_srgb, srgb_to_linear)

    S, dev = cfg.crop, cfg.device

    # ---- 图层组装：背景(全图 alpha=1) + 1~3 个程序化前景 ----
    bg_rgb = bg_pool.sample(rng, dev)
    if rng.uniform() < cfg.p_highlight:
        bg_rgb = _paste_highlights(bg_rgb, rng,
                                   int(rng.integers(*cfg.n_highlight_range)))
    bg_disp = float(rng.uniform(*cfg.bg_disp_range))
    layers = [(bg_rgb, torch.ones(1, S, S, device=dev), bg_disp)]

    n_fg = int(rng.integers(cfg.n_fg_range[0], cfg.n_fg_range[1] + 1))
    fg_disps = sorted(rng.uniform(*cfg.fg_disp_range, size=n_fg).tolist())
    for d_fg in fg_disps:
        # 细薄结构（枝/草/栅栏）与光滑 blob 混采（找补对策 a，见 SynthConfig）。
        alpha = (_random_thin_alpha(S, rng, dev) if rng.uniform() < cfg.p_thin_fg
                 else _random_blob_alpha(S, rng, dev))
        layers.append((_random_texture(S, rng, dev), alpha, float(d_fg)))

    # ---- 控制向量采样：各系数独立采样 = 解耦监督（CLAUDE.md 第 7/8 节）----
    coeffs = sample_random(rng)
    focus_on_fg = rng.uniform() < cfg.p_focus_fg and n_fg > 0
    d_f = float(fg_disps[int(rng.integers(0, n_fg))]) if focus_on_fg else bg_disp
    # K 由"最大 |CoC|"反推：保证盘半径落在采样范围、不超标定上限（~120px）。
    max_dev = max(abs(1.0 - d_f), abs(d_f))
    K = float(rng.uniform(*cfg.max_coc_range)) / max(max_dev, 1e-3)
    ctrl = RenderControl(focus_disparity=d_f, aperture_K=K, coeffs=coeffs,
                         highlight_gain=cfg.highlight_gain)
    # 裁块视场位置：H 按面积均匀采样（√U），方位角均匀。
    H_field = math.sqrt(rng.uniform(0.0, 1.0))
    azimuth = float(rng.uniform(0, 2 * math.pi))

    # ---- 虚拟画幅几何 + H 子带（P1a 第二步，NETWORK_DESIGN §4.1）----
    # 把裁块放进 half_diag = S×factor 的虚拟画幅，逐像素像高/方位角按精确几何算；
    # 块内 H 跨度大时 GT 分 2~3 个 H 子带分别渲染再 tent 混合，让"条件图说 H 在变、
    # GT 也真的在变"——防止自相矛盾（NETWORK_DESIGN §10 风险表第一条）。
    from refine.conditioning import _tent_weights, field_geometry
    factor = math.exp(rng.uniform(math.log(cfg.half_diag_factor[0]),
                                  math.log(cfg.half_diag_factor[1])))
    H_map, az_map = field_geometry((S, S), dev, dtype=torch.float32,
                                   H_field=H_field, azimuth=azimuth,
                                   half_diag=S * factor)
    span = float(H_map.max() - H_map.min())
    t1, t2 = cfg.subband_span_thresh
    if span < t1:
        # 跨度小：单 H 渲染（与 v1 同口径）。条件图同步退化为常数——
        # 物理一致性优先：GT 是按恒定 H 渲的，条件图就不能声称 H 在变。
        H_centers = [float(H_map.mean())]
        H_weights = None
        H_map = torch.full_like(H_map, H_centers[0])
        az_map = torch.full_like(az_map, azimuth)
    else:
        n_sub = 2 if span < t2 else 3
        H_centers = torch.linspace(float(H_map.min()), float(H_map.max()),
                                   n_sub).tolist()
        H_weights = _tent_weights(H_map, H_centers)

    # ---- GT：线性域精确分层合成 → sRGB ----
    layers_lin = [(srgb_to_linear(rgb, ctrl), a, d) for rgb, a, d in layers]
    with torch.no_grad():
        bokeh_lin = composite_blurred_layers(layers_lin, ctrl,
                                             H_field=H_field, azimuth=azimuth,
                                             H_centers=H_centers,
                                             H_weights=H_weights)
    bokeh_gt = linear_to_srgb(bokeh_lin, ctrl).clamp(0.0, 1.0)

    # ---- 网络输入：全焦合成图 + 真视差 → 扰动视差 ----
    image = layers[0][0]
    disp_gt = torch.full((S, S), bg_disp, device=dev)
    for rgb, a, d in layers[1:]:
        image = rgb * a + image * (1 - a)
        disp_gt = torch.where(a[0] > 0.5,
                              torch.as_tensor(d, device=dev, dtype=disp_gt.dtype),
                              disp_gt)
    disp_in = perturb_disparity(disp_gt, rng, cfg)

    # 控制向量的数值表示（细化网 FiLM 条件；顺序固定 —— 改这里必须同步改细化网）：
    # [d_f, K/100, W040, W131, W222, W220, n_blades/10, vignette_s, vignette_R,
    #  loca_R, laca_R, H_field, azimuth/π]   —— 共 13 维（洋葱圈已移除，DECISIONS D15）
    c = coeffs
    ctrl_vec = torch.tensor([d_f, K / 100.0, c.W040_spherical, c.W131_coma,
                             c.W222_astigmatism, c.W220_field_curv,
                             c.n_blades / 10.0, c.vignette_strength,
                             c.vignette_radius,
                             c.loca_rgb[0], c.laca_rgb[0],
                             H_field, azimuth / math.pi],
                            device=dev, dtype=torch.float32)

    return {"image": image, "disparity": disp_in, "disparity_gt": disp_gt,
            "bokeh_gt": bokeh_gt, "ctrl_vec": ctrl_vec,
            "meta": {"ctrl": ctrl, "H_field": H_field, "azimuth": azimuth,
                     "layer_disps": [bg_disp] + fg_disps,
                     # P1 条件构建与输入侧渲染所需（与 GT 严格同一组几何/子带）：
                     "H_map": H_map, "az_map": az_map,
                     "H_centers": H_centers, "H_weights": H_weights}}


class SynthBokehDataset:
    """torch 风格的在线合成数据集（每次 __getitem__ 现场生成全新样本）。

    注意：样本默认在 GPU 上生成（FFT 渲染快）。DataLoader 多 worker 场景需
    改 device='cpu' 或在主进程生成 —— M2 训练接入时按实测吞吐再定。
    """

    def __init__(self, cfg: SynthConfig | None = None, length: int = 10000):
        import numpy as np
        self.cfg = cfg or SynthConfig()
        self.length = length
        self.bg_pool = BackgroundPool(self.cfg.bg_dirs, self.cfg.crop)
        self._rng = np.random.default_rng(self.cfg.seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return make_sample(self.cfg, self.bg_pool, self._rng)


# 兼容旧名（refine/train 的 stub 可能引用）。
OnlineSynthDataset = SynthBokehDataset


# ==============================================================================
# 5) 可视化自检
# ==============================================================================
def main():
    import time

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SynthBokehDataset(SynthConfig(device=device, seed=0))
    n = 4
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.6 * n))
    t0 = time.time()
    for i in range(n):
        s = ds[i]
        panels = [("input (all-in-focus)", s["image"]),
                  ("disparity (perturbed)", s["disparity"]),
                  ("disparity GT", s["disparity_gt"]),
                  ("bokeh GT", s["bokeh_gt"])]
        for ax, (name, t) in zip(axes[i], panels):
            arr = t.cpu().numpy()
            if arr.ndim == 3:
                ax.imshow(arr.transpose(1, 2, 0))
            else:
                ax.imshow(arr, cmap="Spectral_r", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(name, fontsize=10)
            ax.axis("off")
        m = s["meta"]
        c = m["ctrl"]
        axes[i, 0].text(-0.04, 0.5,
                        f"d_f={float(c.focus_disparity):.2f} K={float(c.aperture_K):.0f}\n"
                        f"W040={c.coeffs.W040_spherical:+.1f} blades={c.coeffs.n_blades}\n"
                        f"H={m['H_field']:.2f}",
                        transform=axes[i, 0].transAxes, fontsize=8,
                        va="center", ha="right")
    dt = (time.time() - t0) / n
    fig.suptitle("data/synth.py online synthesis check "
                 f"({dt * 1000:.0f} ms/sample on {device})", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "samples.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[synth] 4 个样本（{dt * 1000:.0f} ms/样本）-> {out}")
    print("[synth] 请肉眼核对：①'扰动视差'的前景边界应可见地不贴/发糊（对照真视差），")
    print("        ②散景 GT 与真视差一致（对焦层锐利）、亮斑应呈光圈形状/带像差风格。")


if __name__ == "__main__":
    main()
