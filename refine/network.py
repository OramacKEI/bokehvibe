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

    # --------------------------------------------------------------------------
    # P2a：NAFNet 块（NETWORK_DESIGN §5.1，DECISIONS D32）—— FiLMResBlock 的替代
    # --------------------------------------------------------------------------
    class LayerNorm2d(nn.Module):
        """通道维 LayerNorm：对 [B,C,H,W] 每个空间位置在 C 维上归一化（NAFNet 用）。

        与 nn.LayerNorm（归一化最后若干维）不同——这里归一化【通道】维，
        保持空间结构。可学习逐通道仿射 (weight, bias)。
        """

        def __init__(self, channels: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.eps = eps

        def forward(self, x):
            mu = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            x = (x - mu) / torch.sqrt(var + self.eps)
            return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

    class NAFFiLMBlock(nn.Module):
        """NAFNet 块 + FiLM 条件（P2a，NETWORK_DESIGN §5.1）。

        结构（两子块各带残差缩放）：
          空间子块：LN → FiLM(γ,β) → 1×1↑2C → 3×3 深度可分 → SimpleGate(→C)
                    → SCA(简化通道注意力) → 1×1→C，  x ← x + β_s·(·)
          通道子块：LN → 1×1↑2C → SimpleGate(→C) → 1×1→C，  x ← x + γ_s·(·)

        关键设计（依据 NAFNet, ECCV22；NAFBET 已在散景变换任务验证）：
        - **SimpleGate** 把特征对半拆开逐元素相乘（2C→C），用乘性门【替代激活函数】
          （无 ReLU/GELU）——这是 NAFNet 参数效率的核心。
        - **SCA**（Simplified Channel Attention）：全局平均池化→1×1→逐通道缩放，
          比 SE 注意力更省，给每通道一个全局自适应增益。
        - **深度可分卷积**：3×3 只在通道内做空间混合，1×1 做通道混合 → 同感受野下
          参数远少于 plain conv（故 NAF 块在同 ch/块数下比 FiLMResBlock 更省参）。
        - **FiLM 挂在空间子块 LN 之后**（条件注入点，NETWORK_DESIGN §5.1）：网络仍按
          镜头/像差系数分化边界修复策略，与 FiLMResBlock 同口径。
        - 残差缩放 β_s/γ_s（NAFNet 的可学习 LayerScale）初始化 =1（标准残差）：
          块从首步即做实变换、梯度正常流（不取 NAFNet 的 0 初始化——那会让块首步
          梯度只到缩放系数、暂时阻断卷积/FiLM；本项目靠近零的 head 已保证输出稳定起点，
          见 ARNet.head 注释，故这里无需再靠 0 残差缩放求稳）。
        """

        def __init__(self, ch: int, cond_feat: int = 64,
                     dw_expand: int = 2, ffn_expand: int = 2):
            super().__init__()
            dw_ch = ch * dw_expand
            ffn_ch = ch * ffn_expand
            # 空间子块
            self.norm1 = LayerNorm2d(ch)
            self.film = nn.Linear(cond_feat, 2 * ch)
            self.conv1 = nn.Conv2d(ch, dw_ch, 1)                       # 1×1 升维
            self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1,
                                   groups=dw_ch)                        # 3×3 深度可分
            # SimpleGate: dw_ch → dw_ch//2 = ch（设 dw_expand=2）
            self.sca = nn.Sequential(                                   # 简化通道注意力
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dw_ch // 2, dw_ch // 2, 1))
            self.conv3 = nn.Conv2d(dw_ch // 2, ch, 1)                  # 1×1 降维
            # 通道子块（FFN）
            self.norm2 = LayerNorm2d(ch)
            self.conv4 = nn.Conv2d(ch, ffn_ch, 1)
            # SimpleGate: ffn_ch → ffn_ch//2 = ch
            self.conv5 = nn.Conv2d(ffn_ch // 2, ch, 1)
            # 可学习残差缩放（LayerScale，初始化 1 = 标准残差，理由见 docstring）。
            self.beta = nn.Parameter(torch.ones(1, ch, 1, 1))
            self.gamma = nn.Parameter(torch.ones(1, ch, 1, 1))

        @staticmethod
        def _simple_gate(x):
            a, b = x.chunk(2, dim=1)
            return a * b

        def forward(self, x, cond_feat):
            # --- 空间子块 ---
            h = self.norm1(x)
            gamma, beta = self.film(cond_feat)[:, :, None, None].chunk(2, dim=1)
            h = h * (1.0 + gamma) + beta                                # FiLM 挂在 LN 之后
            h = self.conv1(h)
            h = self.conv2(h)
            h = self._simple_gate(h)                                    # 2C → C
            h = h * self.sca(h)                                         # 通道注意力
            h = self.conv3(h)
            x = x + h * self.beta
            # --- 通道子块（FFN）---
            h = self.norm2(x)
            h = self.conv4(h)
            h = self._simple_gate(h)                                    # 2C → C
            h = self.conv5(h)
            x = x + h * self.gamma
            return x

    def _make_block(block_type: str, ch: int, cond_feat: int):
        """残差块工厂：'film_res'（v1 plain conv×2）/ 'naf'（P2a NAFNet 块）。
        两者 forward 签名一致 (x, cond_feat)，可在 ARNet/IUNet 里无缝替换。"""
        if block_type == "naf":
            return NAFFiLMBlock(ch, cond_feat)
        return FiLMResBlock(ch, cond_feat)

    def _run_block(blk, h, cond_feat, use_checkpoint: bool):
        """运行一个残差块；use_checkpoint=True 时用梯度检查点（反向重算激活、
        不缓存中间张量）。NAF 块在全分辨率 IUNet 的激活显存是 film_res 的数倍
        （2 子块 ×2 通道扩张 + depthwise），开检查点可在同 batch/crop 下放进 12GB，
        代价 ~25% 慢（D32）。非重入模式正确处理 (h, cond_feat) 多输入与 FiLM 链路。"""
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(blk, h, cond_feat, use_reentrant=False)
        return blk(h, cond_feat)

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
                     cond_feat: int = 64, out_ch: int = 4,
                     block_type: str = "film_res", grad_checkpoint: bool = False):
            super().__init__()
            self.down = nn.PixelUnshuffle(2)
            self.conv0 = nn.Conv2d(in_ch * 4, mid, 3, padding=1)
            self.blocks = nn.ModuleList(
                [_make_block(block_type, mid, cond_feat) for _ in range(n_blocks)])
            self.grad_checkpoint = grad_checkpoint
            # 头输出 out_ch × 4，PixelShuffle 还原空间分辨率。
            # v1 融合式：out_ch=4（3 散景残差 + 1 误差 logit）；
            # Plan B matte 式：out_ch=1（粗 matte logit，D29）。
            self.head = nn.Conv2d(mid, out_ch * 4, 3, padding=1)
            self.up = nn.PixelShuffle(2)
            self.act = nn.ELU(inplace=True)
            # 近零初始化（不能严格置零：W=0 会把流向 FiLM/cond 的梯度全部阻断，
            # 条件分支永远学不动）。std=1e-4 → 起点稳定。
            nn.init.normal_(self.head.weight, std=1e-4)
            nn.init.zeros_(self.head.bias)

        def forward(self, x_half, cond_feat):
            h = self.act(self.conv0(self.down(x_half)))
            for blk in self.blocks:
                h = _run_block(blk, h, cond_feat, self.grad_checkpoint and self.training)
            return self.up(self.head(h))            # [B,out_ch,H/2,W/2]（caller 负责拆分）

    # ==========================================================================
    # 3) IUNet：全分辨率精化——把粗神经散景按全分辨率引导信息锐化
    # ==========================================================================
    class IUNet(nn.Module):
        """BokehMe IUNet 的轻量 FiLM 版（单级 ×2；更高分辨率推理时可迭代调用）。"""

        def __init__(self, guide_ch: int = 8, mid: int = 48, n_blocks: int = 2,
                     cond_feat: int = 64, coarse_ch: int = 3, out_ch: int = 3,
                     block_type: str = "film_res", grad_checkpoint: bool = False):
            super().__init__()
            # coarse_ch / out_ch：v1 散景式=3/3；Plan B matte 式=1/1（粗→精 matte logit）。
            self.conv0 = nn.Conv2d(guide_ch + coarse_ch, mid, 3, padding=1)
            self.blocks = nn.ModuleList(
                [_make_block(block_type, mid, cond_feat) for _ in range(n_blocks)])
            self.grad_checkpoint = grad_checkpoint
            self.head = nn.Conv2d(mid, out_ch, 3, padding=1)
            self.act = nn.ELU(inplace=True)
            nn.init.normal_(self.head.weight, std=1e-4)   # 近零，理由同 ARNet.head
            nn.init.zeros_(self.head.bias)

        def forward(self, guide_full, coarse_up, cond_feat):
            h = self.act(self.conv0(torch.cat([guide_full, coarse_up], dim=1)))
            for blk in self.blocks:
                h = _run_block(blk, h, cond_feat, self.grad_checkpoint and self.training)
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
                     extra_cond_ch: int = 0, matte_mode: bool = False,
                     block_type: str = "film_res", grad_checkpoint=None):
            super().__init__()
            self.cond_dim = cond_dim
            self.extra_cond_ch = extra_cond_ch              # 0=v1；3=P1a；9=P1a+P1b
            self.block_type = block_type                    # 'film_res'(v1) / 'naf'(P2a)
            # Plan B（D29）：matte_mode=True → 网络只输出 1ch 几何 matte α，不画颜色；
            # 由 train/e2e 用 α 在两张物理渲染 B_fg/B_bg 间逐像素选择（找补结构上不可能）。
            self.matte_mode = matte_mode
            # 梯度检查点：None=自动（naf 开、film_res 关——naf 全分辨率激活显存数倍，D32）。
            # 只挂在【IUNet】（全分辨率，激活显存大头）；ARNet 在 1/4 分辨率激活很小，
            # 不必检查点（省去其反向重算时间）。
            if grad_checkpoint is None:
                grad_checkpoint = (block_type == "naf")
            self.grad_checkpoint = grad_checkpoint
            guide_ch = 8 + extra_cond_ch
            self.cond_enc = CondEncoder(cond_dim, cond_feat)
            head_ch = 1 if matte_mode else 4                # matte: 仅 1ch；v1: 3 残差+1 mask
            coarse_ch = 1 if matte_mode else 3
            self.arnet = ARNet(in_ch=guide_ch, mid=ar_mid, n_blocks=ar_blocks,
                               cond_feat=cond_feat, out_ch=head_ch,
                               block_type=block_type, grad_checkpoint=False)
            self.iunet = IUNet(guide_ch=guide_ch, mid=iu_mid, n_blocks=iu_blocks,
                               cond_feat=cond_feat, coarse_ch=coarse_ch,
                               out_ch=coarse_ch, block_type=block_type,
                               grad_checkpoint=grad_checkpoint)

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

            guide_half = F.interpolate(guide, scale_factor=0.5, mode="bilinear",
                                       align_corners=False)
            ar_out = self.arnet(guide_half, cond)         # [B,head_ch,H/2,W/2]

            # ============ Plan B：matte 式（D29）—— 网络只输出 1ch 几何 matte ============
            if self.matte_mode:
                # ARNet 出粗 matte logit，IUNet 全分辨率精化（粗→精，1ch）。
                coarse = ar_out                            # [B,1,H/2,W/2] 粗 matte logit
                coarse_up = F.interpolate(coarse, size=image.shape[-2:],
                                          mode="bilinear", align_corners=False)
                matte_logit = self.iunet(guide, coarse_up, cond)   # 残差精化（仍 logit）
                matte = torch.sigmoid(matte_logit)
                if mask_gate is not None:
                    # 边界带门控：带外 α≡0 → 选择式重渲在带外恒取 B_phys（见 train/e2e
                    # 的 splice）。把网络作用从结构上限制在边界带内（D22/D23 门控沿用）。
                    matte = matte * mask_gate
                # error_mask 键复用 matte（供 e2e 的 find_boundary_zoom/可视化沿用同接口）。
                return {"matte": matte, "matte_logit": matte_logit,
                        "error_mask": matte}

            # ============ v1：误差图融合式（B_neural 直接出颜色，保留作消融基线）============
            res_half, mask_logit_half = ar_out[:, :3], ar_out[:, 3:]
            coarse = guide_half[:, 4:7] + res_half         # B_phys(半分) + 残差
            coarse_up = F.interpolate(coarse, size=image.shape[-2:],
                                      mode="bilinear", align_corners=False)
            bokeh_neural = self.iunet(guide, coarse_up, cond)
            mask = torch.sigmoid(
                F.interpolate(mask_logit_half, size=image.shape[-2:],
                              mode="bilinear", align_corners=False))
            if mask_gate is not None:
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
    # Plan B 默认配置：matte 式 + 9 通道物理条件图（P1a 空间 3 + P1b 描述子 6）。
    net = RefineNet(extra_cond_ch=9, matte_mode=True).to(device)

    # ① 参数量：必须 < 2M（CLAUDE.md 第 5 节硬约束）。matte 头比 v1 更省（少 3ch 颜色输出）。
    n_params = count_parameters(net)
    n_v1 = count_parameters(RefineNet(extra_cond_ch=9, matte_mode=False))
    assert n_params < 2_000_000, f"参数量 {n_params:,} 超出 2M 轻量铁律！"
    print(f"[refine] matte 式可训练参数：{n_params:,}（< 2M ✓，v1 融合式同条件 {n_v1:,}）  "
          f"ARNet={count_parameters(net.arnet):,} IUNet={count_parameters(net.iunet):,} "
          f"Cond={count_parameters(net.cond_enc):,}")
    # P2a（D32）：NAFNet 块的参数对照。同 ch(96/48) 约半参；加宽到 144/72 ≈ 同预算。
    n_naf = count_parameters(RefineNet(extra_cond_ch=9, matte_mode=True, block_type="naf"))
    n_naf_w = count_parameters(RefineNet(extra_cond_ch=9, matte_mode=True,
                                         block_type="naf", ar_mid=144, iu_mid=72))
    print(f"[refine] P2a NAFNet 块：同 ch(96/48)={n_naf:,}（≈半参）；"
          f"加宽(144/72)={n_naf_w:,}（≈同预算 film_res）")

    # ② 前向形状 + ③ 可微性（对网络参数与 ctrl_vec 都要有梯度——FiLM 链路）。
    B, S = 2, 128
    img = torch.rand(B, 3, S, S, device=device)
    disp = torch.rand(B, S, S, device=device)
    phys = torch.rand(B, 3, S, S, device=device)
    cvec = torch.rand(B, 13, device=device, requires_grad=True)
    cmaps = torch.rand(B, 9, S, S, device=device)
    gate = torch.rand(B, 1, S, S, device=device)
    out = net(img, disp, phys, cvec, cond_maps=cmaps, mask_gate=gate)
    assert out["matte"].shape == (B, 1, S, S), out["matte"].shape
    assert (out["matte"] >= 0).all() and (out["matte"] <= 1).all(), "matte 应 ∈[0,1]"
    out["matte"].mean().backward()
    g_net = sum(float(p.grad.abs().sum()) for p in net.parameters()
                if p.grad is not None)
    g_c = float(cvec.grad.abs().sum())
    assert g_net > 0 and g_c > 0, f"梯度断链：net={g_net}, ctrl_vec={g_c}"
    print(f"[refine] matte 前向形状 ✓；梯度：net |∑|={g_net:.3e}, ctrl_vec |∑|={g_c:.3e} ✓")

    # ④ 真样本可视化：B = α·B_fg + (1−α)·B_bg，带外回落 B_phys（Plan B 选择式重渲，D29）。
    #    未训练时 α≈0.5（近零初始化），输出≈带内两渲染均值；训练后 α 应学成真前景占比。
    from data.synth import SynthBokehDataset, SynthConfig
    from refine.conditioning import boundary_band, condition_maps
    from render.renderer import (push_band_to_background, render_field_patch,
                                 snap_disparity_edges)
    ds = SynthBokehDataset(SynthConfig(device=device, seed=3))
    s = ds[0]
    m = s["meta"]
    with torch.no_grad():
        disp = snap_disparity_edges(s["disparity"])
        band = boundary_band(disp, m["ctrl"])
        B_fg = render_field_patch(s["image"], disp, m["ctrl"],
                                  m["H_field"], m["azimuth"],
                                  H_centers=m["H_centers"], H_weights=m["H_weights"])
        disp_bg = push_band_to_background(disp, m["ctrl"], band=band)
        B_bg = render_field_patch(s["image"], disp_bg, m["ctrl"],
                                  m["H_field"], m["azimuth"],
                                  H_centers=m["H_centers"], H_weights=m["H_weights"])
        cm = condition_maps(disp, m["ctrl"], m["H_map"], m["az_map"],
                            H_centers=m["H_centers"])
        out = net(s["image"][None], disp[None], B_fg[None],
                  s["ctrl_vec"][None], cond_maps=cm[None], mask_gate=band[None])
        alpha = out["matte"][0]                            # [1,H,W]
        B_band = alpha * B_fg + (1.0 - alpha) * B_bg
        B = band * B_band + (1.0 - band) * B_fg            # 带外 = B_phys(=B_fg)
    print(f"[refine] matte 均值={float(alpha.mean()):.3f}（未训练≈0.5×band）；"
          f"|B−B_fg| 带内={float((B - B_fg).abs()[band.expand_as(B) > 0.5].mean()):.4f} "
          f"带外={float((B - B_fg).abs()[band.expand_as(B) <= 0.5].mean()):.4f}（带外应≈0）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = [("input (all-in-focus)", s["image"]),
              ("B_fg = B_phys", B_fg),
              ("B_bg (band→background)", B_bg),
              ("matte alpha (untrained)", alpha[0]),
              ("B = matte select", B),
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
    fig.suptitle(f"refine/network.py Plan B matte smoke — {n_params:,} params "
                 "(untrained; B=α·B_fg+(1−α)·B_bg, 找补结构上不可能)", fontsize=11)
    fig.tight_layout()
    out_png = OUT_DIR / "smoke_matte.png"
    fig.savefig(str(out_png), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[refine] 可视化 -> {out_png}")


if __name__ == "__main__":
    _smoke_test()
