"""
refine/network.py
=================
边界细化网络 —— 全项目【唯一可训练】的部分（CLAUDE.md 第 2 节，~1–2M 参数上限）。

⚠️ 轻量铁律（CLAUDE.md 第 5 节）：可训练参数 ≤ ~2M。严禁引入大骨干（如 SDXL）。
   本实现实测约 0.7M（运行 `python -m refine.network` 打印精确值并断言 < 2M）。

【职责】物理渲染器（render/renderer.py）从"估计/扰动视差"出发渲染，深度边界处
会出现渗色、硬边、半透明圈等瑕疵（深度误差 + 分层近似的固有代价）；本网络
对照全焦图修复这些瑕疵，并通过误差图把神经结果与物理结果融合：

      输入：全焦图 I[3] ‖ 扰动视差 D̃[1] ‖ 物理渲染 B_phys[3] ‖ 离焦提示 r̃[1]
            （r̃ = K·(D̃ − d_f)/128，由 ctrl_vec 前两维现场算出 —— 物理一致的免费特征）
        │
        ▼  下采样 ×2（ARNet 永远在低分辨率工作 → 高分辨率推理时计算量可控）
      ┌─────────────── ARNet（自适应渲染网，BokehMe 轻量版）───────────────┐
      │ PixelUnshuffle(2) → 96ch × 3 个 FiLM 残差块 → PixelShuffle(2)       │
      │ 输出：粗神经散景（= B_phys 半分辨率 + 残差）+ 误差图 logit m        │
      └──────────────────────────────────────────────────────────────────┘
        │ 上采样回全分辨率
        ▼
      ┌─────────────── IUNet（迭代上采样网，BokehMe 轻量版）────────────────┐
      │ [全分辨率引导 8ch ‖ 上采样粗散景 3ch] → 48ch × 2 个 FiLM 残差块     │
      │ 输出：全分辨率神经散景 B_neural（= 上采样粗散景 + 残差）            │
      └──────────────────────────────────────────────────────────────────┘
        ▼
      融合（CLAUDE.md 第 2 节"误差图融合"）：B = m·B_neural + (1−m)·B_phys
      —— m 只在物理渲染出错的地方（深度边界）才升高，平坦区域保留物理结果，
         物理层的像差风格（肥皂泡/猫眼/色差…）不会被网络"洗掉"。

【控制向量 c 的一致性注入】（CLAUDE.md 第 2 节，关键设计）：
    c 同时进入 (1) 渲染器（物理塑形 PSF）和 (2) 本网络（FiLM 条件注入）。
    FiLM = Feature-wise Linear Modulation：把 c 编码成每个残差块的逐通道仿射
    (γ, β)，h ← h·(1+γ)+β。网络由此"知道"当前是什么镜头/多大虚化，
    可以对不同像差风格采取不同的边界修复策略。

【接口约定】ctrl_vec 为 data/synth.py make_sample 定义的 13 维向量
    [d_f, K/100, W040, W131, W222, W220, n_blades/10, vignette_s, vignette_R,
     loca_R, laca_R, H_field, azimuth/π]
    本文件依赖其中 [0]=d_f、[1]=K/100 两个索引来现场计算离焦提示通道 ——
    synth 改顺序必须同步改这里（两处都有醒目注释互相指向）。

【零初始化的稳定起点】两个残差头 + 误差图头都零初始化 →
    初始时 B_neural ≈ B_phys（仅差一次下/上采样的轻微软化）、m = 0.5，
    网络从"几乎透传物理渲染"出发学习，训练初期不会破坏物理结果。

运行自检（参数量 / 前向形状 / 可微性 / 真样本可视化）：
    python -m refine.network        # → outputs/refine_test/smoke.png
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "refine_test"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # torch 尚未安装时仍可导入本文件做静态检查
    _TORCH = False
    nn = None


if _TORCH:

    # ==========================================================================
    # 1) FiLM 条件：控制向量 c → 共享特征 → 每块的逐通道仿射 (γ, β)
    # ==========================================================================
    class CondEncoder(nn.Module):
        """把 13 维控制向量编码成共享条件特征（各 FiLM 块再各自线性映射成 γ/β）。

        两层 MLP 足够：c 本身就是低维、物理可解释的量，不需要深编码器。
        """

        def __init__(self, cond_dim: int, feat_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cond_dim, feat_dim), nn.ELU(inplace=True),
                nn.Linear(feat_dim, feat_dim), nn.ELU(inplace=True),
            )

        def forward(self, c):                      # [B,cond_dim] → [B,feat_dim]
            return self.net(c)

    class FiLMResBlock(nn.Module):
        """残差块 + FiLM 调制：x + FiLM(conv(act(conv(x))))，再激活。

        γ 用 (1+γ̂) 形式：线性层初始输出≈0 时调制≈恒等，训练起点稳定。
        """

        def __init__(self, ch: int, cond_feat: int = 64):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
            self.film = nn.Linear(cond_feat, 2 * ch)
            self.act = nn.ELU(inplace=True)

        def forward(self, x, cond_feat):
            h = self.conv2(self.act(self.conv1(x)))
            gamma, beta = self.film(cond_feat)[:, :, None, None].chunk(2, dim=1)
            h = h * (1.0 + gamma) + beta
            return self.act(x + h)

    # ==========================================================================
    # 2) ARNet：低分辨率的"自适应渲染"——预测粗神经散景 + 误差图
    # ==========================================================================
    class ARNet(nn.Module):
        """BokehMe ARNet 的轻量 FiLM 版。

        输入已是半分辨率（caller 负责下采样）；内部再 PixelUnshuffle(2) 把空间
        信息折进通道（无损下采样，BokehMe 的 Space2Depth 同款）→ 实际卷积在
        1/4 输入分辨率上跑，计算量很小。输出 PixelShuffle 回半分辨率。
        """

        def __init__(self, in_ch: int = 8, mid: int = 96, n_blocks: int = 3,
                     cond_feat: int = 64):
            super().__init__()
            self.down = nn.PixelUnshuffle(2)
            self.conv0 = nn.Conv2d(in_ch * 4, mid, 3, padding=1)
            self.blocks = nn.ModuleList(
                [FiLMResBlock(mid, cond_feat) for _ in range(n_blocks)])
            # 头输出 (3 残差 + 1 误差 logit) × 4，PixelShuffle 还原空间分辨率。
            self.head = nn.Conv2d(mid, 4 * 4, 3, padding=1)
            self.up = nn.PixelShuffle(2)
            self.act = nn.ELU(inplace=True)
            # 近零初始化（不能严格置零：W=0 会把流向 FiLM/cond 的梯度全部阻断，
            # 条件分支永远学不动）。std=1e-4 → 起点仍≈透传物理渲染。
            nn.init.normal_(self.head.weight, std=1e-4)
            nn.init.zeros_(self.head.bias)

        def forward(self, x_half, cond_feat):
            h = self.act(self.conv0(self.down(x_half)))
            for blk in self.blocks:
                h = blk(h, cond_feat)
            out = self.up(self.head(h))            # [B,4,H/2,W/2]
            return out[:, :3], out[:, 3:]          # (散景残差, 误差图 logit)

    # ==========================================================================
    # 3) IUNet：全分辨率精化——把粗神经散景按全分辨率引导信息锐化
    # ==========================================================================
    class IUNet(nn.Module):
        """BokehMe IUNet 的轻量 FiLM 版（单级 ×2；更高分辨率推理时可迭代调用）。"""

        def __init__(self, guide_ch: int = 8, mid: int = 48, n_blocks: int = 2,
                     cond_feat: int = 64):
            super().__init__()
            self.conv0 = nn.Conv2d(guide_ch + 3, mid, 3, padding=1)
            self.blocks = nn.ModuleList(
                [FiLMResBlock(mid, cond_feat) for _ in range(n_blocks)])
            self.head = nn.Conv2d(mid, 3, 3, padding=1)
            self.act = nn.ELU(inplace=True)
            nn.init.normal_(self.head.weight, std=1e-4)   # 近零，理由同 ARNet.head
            nn.init.zeros_(self.head.bias)

        def forward(self, guide_full, coarse_up, cond_feat):
            h = self.act(self.conv0(torch.cat([guide_full, coarse_up], dim=1)))
            for blk in self.blocks:
                h = blk(h, cond_feat)
            return coarse_up + self.head(h)        # 残差精化

    # ==========================================================================
    # 4) RefineNet：组装 + 误差图融合
    # ==========================================================================
    class RefineNet(nn.Module):
        """边界细化网（ARNet + IUNet + FiLM），见文件头说明。

        forward 输入（全部 [B,...]，sRGB [0,1]，H/W 需为 4 的倍数）：
            image      [B,3,H,W]  全焦图
            disparity  [B,H,W] 或 [B,1,H,W]  扰动/估计视差（近=1）
            bokeh_phys [B,3,H,W]  物理渲染器输出
            ctrl_vec   [B,13]     控制向量（见文件头接口约定）
            cond_maps  [B,extra_cond_ch,H,W] 可选：逐像素物理条件图（P1a/P1b，
                       refine/conditioning.condition_maps 产出，拼在 guide 末尾；
                       extra_cond_ch=0 时不传 = v1 行为）
            mask_gate  [B,1,H,W] 可选：误差图门控（边界带，conditioning.boundary_band；
                       m ← m·gate，把神经分支的活动范围物理限制在视差边缘带内——
                       找补对策 b，见 PROJECT_STATUS §6 [2026-06-12]）
        返回 dict：
            bokeh        [B,3,H,W] 最终融合输出 B
            bokeh_neural [B,3,H,W] 神经分支（调试/可视化）
            error_mask   [B,1,H,W] 误差图 m ∈ (0,1)（已乘门控；调试/可视化）
        """

        def __init__(self, cond_dim: int = 13, cond_feat: int = 64,
                     ar_mid: int = 96, ar_blocks: int = 3,
                     iu_mid: int = 48, iu_blocks: int = 2,
                     extra_cond_ch: int = 0):
            super().__init__()
            self.cond_dim = cond_dim
            self.extra_cond_ch = extra_cond_ch              # 0=v1；3=P1a；9=P1a+P1b
            guide_ch = 8 + extra_cond_ch
            self.cond_enc = CondEncoder(cond_dim, cond_feat)
            self.arnet = ARNet(in_ch=guide_ch, mid=ar_mid, n_blocks=ar_blocks,
                               cond_feat=cond_feat)
            self.iunet = IUNet(guide_ch=guide_ch, mid=iu_mid, n_blocks=iu_blocks,
                               cond_feat=cond_feat)

        @staticmethod
        def _defocus_hint(disparity, ctrl_vec):
            """离焦提示通道 r̃ = K·(D̃ − d_f)/128：物理一致的免费特征。

            ⚠️ 依赖 ctrl_vec 索引 [0]=d_f、[1]=K/100（data/synth.py make_sample
            定义，改顺序必须同步）。/128 把像素 CoC 压到 ~[-1,1]（标定上限 128px）。
            """
            d_f = ctrl_vec[:, 0][:, None, None, None]
            K = ctrl_vec[:, 1][:, None, None, None] * 100.0
            return K * (disparity - d_f) / 128.0

        def forward(self, image, disparity, bokeh_phys, ctrl_vec,
                    cond_maps=None, mask_gate=None):
            if disparity.dim() == 3:
                disparity = disparity[:, None]      # [B,H,W] → [B,1,H,W]
            cond = self.cond_enc(ctrl_vec)

            # 全分辨率引导：I ‖ D̃ ‖ B_phys ‖ 离焦提示（8ch）‖ 物理条件图（P1）。
            # 条件图拼在末尾 → 前 8 通道的索引约定（如下方 guide_half[:,4:7]）不变。
            hint = self._defocus_hint(disparity, ctrl_vec)
            parts = [image, disparity, bokeh_phys, hint]
            if self.extra_cond_ch > 0:
                assert cond_maps is not None and cond_maps.shape[1] == self.extra_cond_ch, \
                    f"需要 [B,{self.extra_cond_ch},H,W] 的 cond_maps（P1 条件图）"
                parts.append(cond_maps)
            guide = torch.cat(parts, dim=1)

            # ---- ARNet @ 半分辨率：粗神经散景 + 误差图 ----
            guide_half = F.interpolate(guide, scale_factor=0.5, mode="bilinear",
                                       align_corners=False)
            res_half, mask_logit_half = self.arnet(guide_half, cond)
            coarse = guide_half[:, 4:7] + res_half  # B_phys(半分) + 残差

            # ---- IUNet @ 全分辨率：上采样 + 引导精化 ----
            coarse_up = F.interpolate(coarse, size=image.shape[-2:],
                                      mode="bilinear", align_corners=False)
            bokeh_neural = self.iunet(guide, coarse_up, cond)

            # ---- 误差图融合：只在物理渲染出错处使用神经结果 ----
            mask = torch.sigmoid(
                F.interpolate(mask_logit_half, size=image.shape[-2:],
                              mode="bilinear", align_corners=False))
            if mask_gate is not None:
                # 边界带门控：带外 m≡0（神经分支结构性失效），从根上杜绝
                # 在平坦/纹理区"找补"锐利纹理的红线违规。
                mask = mask * mask_gate
            bokeh = mask * bokeh_neural + (1.0 - mask) * bokeh_phys
            return {"bokeh": bokeh, "bokeh_neural": bokeh_neural,
                    "error_mask": mask}


def count_parameters(module) -> int:
    """可训练参数总数（轻量铁律的检查口径）。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ==============================================================================
# 5) 自检：参数量 / 前向形状 / 可微性 / 真样本可视化
# ==============================================================================
def _smoke_test():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # P1 默认配置：9 通道物理条件图（P1a 空间 3 + P1b 描述子 6）。
    net = RefineNet(extra_cond_ch=9).to(device)

    # ① 参数量：必须 < 2M（CLAUDE.md 第 5 节硬约束）。
    n_params = count_parameters(net)
    n_v1 = count_parameters(RefineNet(extra_cond_ch=0))
    assert n_params < 2_000_000, f"参数量 {n_params:,} 超出 2M 轻量铁律！"
    print(f"[refine] 可训练参数：{n_params:,}（< 2M ✓，v1 基线 {n_v1:,}）  "
          f"ARNet={count_parameters(net.arnet):,} IUNet={count_parameters(net.iunet):,} "
          f"Cond={count_parameters(net.cond_enc):,}")

    # ② 前向形状 + ③ 可微性（对网络参数与 ctrl_vec 都要有梯度——FiLM 链路）。
    B, S = 2, 128
    img = torch.rand(B, 3, S, S, device=device)
    disp = torch.rand(B, S, S, device=device)
    phys = torch.rand(B, 3, S, S, device=device)
    cvec = torch.rand(B, 13, device=device, requires_grad=True)
    cmaps = torch.rand(B, 9, S, S, device=device)
    gate = torch.rand(B, 1, S, S, device=device)
    out = net(img, disp, phys, cvec, cond_maps=cmaps, mask_gate=gate)
    assert out["bokeh"].shape == (B, 3, S, S) and out["error_mask"].shape == (B, 1, S, S)
    out["bokeh"].mean().backward()
    g_net = sum(float(p.grad.abs().sum()) for p in net.parameters()
                if p.grad is not None)
    g_c = float(cvec.grad.abs().sum())
    assert g_net > 0 and g_c > 0, f"梯度断链：net={g_net}, ctrl_vec={g_c}"
    print(f"[refine] 前向形状 ✓；梯度：net |∑|={g_net:.3e}, ctrl_vec |∑|={g_c:.3e} ✓")

    # ④ 真样本可视化：零初始化起点应≈透传物理渲染（人工核对融合接线正确）。
    from data.synth import SynthBokehDataset, SynthConfig
    from refine.conditioning import boundary_band, condition_maps
    from render.renderer import render_field_patch, snap_disparity_edges
    ds = SynthBokehDataset(SynthConfig(device=device, seed=3))
    s = ds[0]
    m = s["meta"]
    with torch.no_grad():
        disp = snap_disparity_edges(s["disparity"])
        phys = render_field_patch(s["image"], disp, m["ctrl"],
                                  m["H_field"], m["azimuth"],
                                  H_centers=m["H_centers"], H_weights=m["H_weights"])
        cm = condition_maps(disp, m["ctrl"], m["H_map"], m["az_map"],
                            H_centers=m["H_centers"])
        gate = boundary_band(disp, m["ctrl"])
        out = net(s["image"][None], disp[None], phys[None],
                  s["ctrl_vec"][None], cond_maps=cm[None], mask_gate=gate[None])
    passthrough_err = float((out["bokeh"][0] - phys).abs().mean())
    print(f"[refine] 零初始化透传误差 |B−B_phys| = {passthrough_err:.4f}"
          "（应很小：仅差一次下/上采样软化的一半权重）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = [("input (all-in-focus)", s["image"]),
              ("disparity (perturbed)", s["disparity"]),
              ("B_phys (renderer)", phys),
              ("B fused (untrained)", out["bokeh"][0]),
              ("error mask m", out["error_mask"][0, 0]),
              ("bokeh GT", s["bokeh_gt"])]
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    for ax, (name, t) in zip(axes, panels):
        arr = t.detach().cpu().numpy()
        if arr.ndim == 3:
            ax.imshow(arr.transpose(1, 2, 0).clip(0, 1))
        else:
            ax.imshow(arr, cmap="Spectral_r", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"refine/network.py smoke test — {n_params:,} params (untrained, "
                 "fused≈B_phys expected)", fontsize=11)
    fig.tight_layout()
    out_png = OUT_DIR / "smoke.png"
    fig.savefig(str(out_png), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[refine] 可视化 -> {out_png}")


if __name__ == "__main__":
    _smoke_test()
