"""
train/train.py
==============
边界细化网训练循环（M2 后半，CLAUDE.md 第 8 节损失 + 第 5 节硬约束）。

只训练【边界细化网】（refine/network.py，~0.7M）；深度网冻结（训练用合成真视差
加扰动，根本不跑 DA V2）、渲染器无权重（物理层，no_grad 下当数据增强用）。

每个训练步的数据流（全部在线生成，无离线数据集）：

    data/synth.py make_sample
        ├─ image      全焦图                       ┐
        ├─ disparity  扰动视差 → snap_edges(可选)  ├─► render_field_patch ─► B_phys
        ├─ bokeh_gt   精确分层 GT（监督目标）      ┘   （no_grad，物理渲染）
        └─ ctrl_vec   13 维控制向量
                │
                ▼
    RefineNet(image, disparity, B_phys, ctrl_vec) ─► B（融合输出）
                │
                ▼
    loss = L1(B, GT) + λp·LPIPS(B, GT)            # 融合输出（主目标）
         + λn·L1(B_neural, GT)                     # 神经分支直接监督（防门控死区）
         + λm·L1(m, m_gt)                          # 误差图显式监督（防塌缩）
    其中 m_gt = clamp(k·|B_phys − GT|, 0, 1)       # 物理渲染的真实误差图

【为什么这样训练就能学到"修边界"】GT 由已知 alpha 精确合成（无深度误差），
B_phys 由扰动视差渲染（带真实形态的深度误差）→ 两者的差集中在深度边界，
网络的容量小（~0.7M），只够学这个局部修复，学不动全图重绘——这正是想要的。

【后两项损失为什么必须有（实测踩坑，见 DECISIONS D19）】只用融合损失训练会
发生"误差图塌缩"：神经分支初始略糊 → mask>0 处损失更大 → 梯度把 mask 压到 0
→ mask=0 后神经分支梯度 ∂L/∂B_neural·m ≈ 0 永远学不动 → 网络退化成纯透传。
BokehMe 的对策正是给误差图配显式 GT（物理渲染的真实误差）、给神经分支配
不经门控的直接重建损失——两条腿先各自站稳，融合才有意义。

【关键诊断指标】L1(B_phys, GT) vs L1(B, GT)：
    后者 < 前者 = 细化网在物理渲染之上有净收益（训练日志每行都打印）。

硬约束（12GB 单卡）：crop 512 / batch 4 起步（configs/base.yaml train 节）。

运行：
    python -m train.train --smoke          # 30 步快速验证（loss 应下降）
    python -m train.train                  # 正式训练（默认 20k 步）
    python -m train.train --iters 5000 --out outputs/train_run2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML = PROJECT_ROOT / "configs" / "base.yaml"


def load_config() -> dict:
    """读 configs/base.yaml（train/refine/depth 三节是本文件用到的）。"""
    import yaml
    with open(CONFIG_YAML) as f:
        return yaml.safe_load(f)


# ==============================================================================
# 1) 批数据：在线合成 + 输入侧物理渲染（no_grad）
# ==============================================================================
def make_batch(ds, batch_size: int, snap_edges: bool,
               extra_cond_ch: int = 0, mask_gate: bool = False,
               matte_mode: bool = False):
    """取 batch_size 个在线样本，逐个做输入侧物理渲染，堆成批张量。

    输入侧管线与真实推理严格一致：扰动视差 →（可选）边缘吸附 → 物理渲染。
    渲染在 no_grad 下进行——训练时控制向量 c 是给定值而非优化对象，
    不需要梯度流过渲染器（指纹标定才需要，那是另一条链路）。

    P1 扩展：extra_cond_ch>0 时附带逐像素物理条件图（3=仅空间 P1a，9=+描述子
    P1b）；mask_gate=True 时附带视差边缘带门控图。两者都从【吸附后的输入视差】
    与样本自带的虚拟画幅几何算出——与推理端 e2e 的口径完全一致。

    Plan B（matte_mode=True，D29）：额外渲染【B_bg】——把边界带视差压到局部背景后
    重渲（push_band_to_background），该处按背景重度虚化。训练时网络 matte α 在
    B_fg(=phys) / B_bg 间逐像素选择。matte_mode 下总是计算 band（splice 与 push 都需要），
    并附带样本的 matte_gt（前景占比真值）。

    Returns:
        (image, disparity, phys, gt, ctrl_vec, cond_maps|None, band|None,
         bg|None, matte_gt|None)   —— 后两项仅 matte_mode 非 None。
    """
    import torch
    from refine.conditioning import boundary_band, condition_maps
    from render.renderer import (push_band_to_background, render_field_patch,
                                 snap_disparity_edges)

    imgs, disps, physs, gts, cvecs = [], [], [], [], []
    conds, bands, bgs, mattes = [], [], [], []
    need_band = mask_gate or matte_mode               # matte 的 splice/push 也要 band
    for _ in range(batch_size):
        s = ds[0]                                   # 在线数据集：每次调用都是新样本
        m = s["meta"]
        disp = s["disparity"]
        if snap_edges:
            disp = snap_disparity_edges(disp)
        with torch.no_grad():
            phys = render_field_patch(s["image"], disp, m["ctrl"],
                                      m["H_field"], m["azimuth"],
                                      H_centers=m["H_centers"],
                                      H_weights=m["H_weights"])
            if extra_cond_ch > 0:
                conds.append(condition_maps(
                    disp, m["ctrl"], m["H_map"], m["az_map"],
                    H_centers=m["H_centers"],
                    with_descriptors=(extra_cond_ch == 9)))
            band = boundary_band(disp, m["ctrl"]) if need_band else None
            if need_band:
                bands.append(band)
            if matte_mode:
                disp_bg = push_band_to_background(disp, m["ctrl"], band=band)
                bg = render_field_patch(s["image"], disp_bg, m["ctrl"],
                                        m["H_field"], m["azimuth"],
                                        H_centers=m["H_centers"],
                                        H_weights=m["H_weights"])
                bgs.append(bg.clamp(0.0, 1.0))
                mattes.append(s["matte_gt"])
        imgs.append(s["image"])
        disps.append(disp)
        physs.append(phys.clamp(0.0, 1.0))
        gts.append(s["bokeh_gt"])
        cvecs.append(s["ctrl_vec"])
    return (torch.stack(imgs), torch.stack(disps), torch.stack(physs),
            torch.stack(gts), torch.stack(cvecs),
            torch.stack(conds) if conds else None,
            torch.stack(bands) if bands else None,
            torch.stack(bgs) if bgs else None,
            torch.stack(mattes) if mattes else None)


# ==============================================================================
# 1.5) 细化网前向 + 输出装配（v1 融合式 / Plan B matte 选择式，统一入口）
# ==============================================================================
def refine_forward(net, img, disp, phys, cvec, cond, band, bg, matte_mode: bool):
    """跑细化网并装配最终散景 B；返回 (B, aux)。

    v1 融合式：B = m·B_neural + (1−m)·B_phys，aux={"map": m}（误差图）。
    Plan B matte 式（D29）：网络出 raw matte α（不经门控，门控由 splice 承担）→
        B = band·(α·B_fg + (1−α)·B_bg) + (1−band)·B_phys
        —— 带内两张物理渲染按 α 选择、带外恒为 B_phys；找补结构上不可能。
        aux={"map": α}（matte，复用 error_mask 接口位）。
    """
    if matte_mode:
        out = net(img, disp, phys, cvec, cond_maps=cond, mask_gate=None)
        alpha = out["matte"]                          # raw α∈[0,1]（门控由 band splice 承担）
        B_band = alpha * phys + (1.0 - alpha) * bg
        B = band * B_band + (1.0 - band) * phys
        # logit 一并返回：matte 损失走 BCE-on-logits（饱和处梯度不消失，避免 α 一旦
        # 被 recon 压到 0 就被 L1·σ' 的消失梯度永久卡死，见 train_probe_matte 实证）。
        return B, {"map": alpha, "logit": out["matte_logit"]}
    out = net(img, disp, phys, cvec, cond_maps=cond, mask_gate=band)
    return out["bokeh"], {"map": out["error_mask"], "neural": out["bokeh_neural"]}


# ==============================================================================
# 2) 损失：L1 + 感知（LPIPS-VGG，可选）
# ==============================================================================
def build_perceptual(device: str):
    """LPIPS(VGG) 感知损失。构建失败（如无网下载不了 VGG 权重）则退回 None，
    训练自动降级为纯 L1 并告警——不让感知损失成为跑通训练的阻塞项。"""
    try:
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)                  # 感知网是冻结的"测量仪"
        return net
    except Exception as e:
        import warnings
        warnings.warn(f"[train] LPIPS 构建失败（{e}），退回纯 L1 损失。")
        return None


# ==============================================================================
# 3) 评测：边界区域 L1（细化网净收益的关键指标）
# ==============================================================================
def evaluate_boundary(net, val_samples, mask_gain: float = 10.0,
                      matte_mode: bool = False):
    """在固定验证样本上量化"边界区域"的 L1：phys vs refined。

    为什么不用全图 L1：边界高误差区只占 ~8% 像素，全图 L1 被平坦区主导，
    细化网的收益在第 4 位小数里看不见。这里只统计 m_gt>0.2 的像素。
    """
    import torch
    net.eval()
    bp = bf = npx = 0.0
    gp = gf = 0.0
    with torch.no_grad():
        for img, disp, phys, gt, cvec, cond, band, bg, _mgt in val_samples:
            B, _ = refine_forward(
                net, img[None], disp[None], phys[None], cvec[None],
                None if cond is None else cond[None],
                None if band is None else band[None],
                None if bg is None else bg[None], matte_mode)
            ep = (phys[None] - gt[None]).abs().mean(1)        # [1,H,W]
            ef = (B - gt[None]).abs().mean(1)
            sel = (mask_gain * ep).clamp(0, 1) > 0.2
            bp += float(ep[sel].sum())
            bf += float(ef[sel].sum())
            npx += float(sel.sum())
            gp += float(ep.mean())
            gf += float(ef.mean())
    net.train()
    n = len(val_samples)
    return {"boundary_phys": bp / max(npx, 1.0),
            "boundary_fused": bf / max(npx, 1.0),
            "global_phys": gp / n, "global_fused": gf / n}


# ==============================================================================
# 4) 可视化：物理 vs 细化 vs GT（训练中/结束时的人工核对窗口）
# ==============================================================================
def save_visualization(net, val_samples, path: Path, device: str,
                       matte_mode: bool = False):
    """对固定验证样本出对比面板：B_phys / B / GT / map(误差图或 matte) / 两者的 |diff×4|。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    net.eval()
    n = len(val_samples)
    fig, axes = plt.subplots(n, 6, figsize=(20, 3.5 * n), squeeze=False)
    with torch.no_grad():
        for i, (img, disp, phys, gt, cvec, cond, band, bg, _mgt) in enumerate(val_samples):
            B, aux = refine_forward(
                net, img[None], disp[None], phys[None], cvec[None],
                None if cond is None else cond[None],
                None if band is None else band[None],
                None if bg is None else bg[None], matte_mode)
            B = B[0]
            map_name = "matte alpha" if matte_mode else "error mask m"
            panels = [
                ("B_phys (renderer)", phys, None),
                ("B refined", B, None),
                ("bokeh GT", gt, None),
                (map_name, aux["map"][0, 0], "mask"),
                ("|B_phys-GT|x4", (phys - gt).abs().mean(0) * 4, "diff"),
                ("|B-GT|x4", (B - gt).abs().mean(0) * 4, "diff"),
            ]
            for ax, (name, t, kind) in zip(axes[i], panels):
                arr = t.detach().float().cpu().numpy()
                if arr.ndim == 3:
                    ax.imshow(arr.transpose(1, 2, 0).clip(0, 1))
                else:
                    ax.imshow(arr.clip(0, 1), cmap="Spectral_r", vmin=0, vmax=1)
                if i == 0:
                    ax.set_title(name, fontsize=9)
                ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    net.train()


# ==============================================================================
# 5) 主训练循环
# ==============================================================================
def main(argv=None):
    import torch
    from data.synth import SynthBokehDataset, SynthConfig
    from refine.network import RefineNet, count_parameters

    cfg = load_config()
    tr = cfg["train"]
    ap = argparse.ArgumentParser(description="边界细化网训练（M2）")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=int(tr["batch_size"]))
    ap.add_argument("--crop", type=int, default=int(tr["crop"]))
    ap.add_argument("--lr", type=float, default=float(tr["lr"]))
    ap.add_argument("--lambda-perc", type=float,
                    default=float(tr.get("lambda_perceptual", 0.1)))
    ap.add_argument("--lambda-neural", type=float,
                    default=float(tr.get("lambda_neural", 0.5)))
    ap.add_argument("--lambda-mask", type=float,
                    default=float(tr.get("lambda_mask", 0.2)))
    ap.add_argument("--mask-gain", type=float,
                    default=float(tr.get("mask_gain", 10.0)),
                    help="m_gt = clamp(gain·|B_phys−GT|, 0, 1) 的误差放大系数")
    ap.add_argument("--boundary-weight", type=float,
                    default=float(tr.get("boundary_weight", 4.0)),
                    help="重建损失的逐像素权重 1+bw·m_gt：边界只占 ~8% 像素，"
                         "不加权时边界梯度被平坦区稀释 ~12 倍")
    ap.add_argument("--lambda-matte", type=float,
                    default=float(tr.get("lambda_matte", 0.3)),
                    help="Plan B：matte 直监 L1(α, α_gt) 带内权重（D29）")
    ap.add_argument("--matte", dest="matte", action="store_true", default=None,
                    help="强制 Plan B matte 选择式重渲（覆盖 config）")
    ap.add_argument("--no-matte", dest="matte", action="store_false",
                    help="强制 v1 误差图融合式（消融基线）")
    ap.add_argument("--block", choices=["film_res", "naf"], default=None,
                    help="残差块类型（P2a，D32）：film_res=v1 plain conv×2；"
                         "naf=NAFNet 块（depthwise+SimpleGate+SCA）。覆盖 config")
    ap.add_argument("--ar-mid", type=int, default=None,
                    help="ARNet 通道宽（naf 同参对照建议 144；默认 config 96）")
    ap.add_argument("--iu-mid", type=int, default=None,
                    help="IUNet 通道宽（naf 同参对照建议 72；默认 config 48）")
    ap.add_argument("--amp", action="store_true", default=False,
                    help="混合精度训练（fp16 网络前向+损失，fp32 物理 PSF）。"
                         "RTX 30xx/40xx/50xx Tensor Core 可显著提速（~30-50%%）")
    ap.add_argument("--no-perceptual", action="store_true")
    ap.add_argument("--bg-dirs", type=str, nargs="+", default=None,
                    help="背景图目录（可多个）。不传则用 SynthConfig 默认的占位池；"
                         "正式训练应传大相册目录，如 COCO train2017")
    ap.add_argument("--resume", type=str, default=None,
                    help="从 checkpoint 续训（加载模型+优化器+迭代数）")
    ap.add_argument("--out", type=str, default="outputs/train_run")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--viz-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=int(cfg["project"]["seed"]))
    ap.add_argument("--smoke", action="store_true",
                    help="30 步快速验证：loss 应下降、出图可核对")
    args = ap.parse_args(argv)
    if args.smoke:
        args.iters, args.log_every = 30, 5
        args.viz_every = args.save_every = args.iters

    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- 细化网 v2 (P1) 开关（configs/base.yaml refine 节；全关=复现 v1）----
    rf = cfg["refine"]
    spatial = bool(rf.get("spatial_cond", False))
    psf_desc = bool(rf.get("psf_desc", False)) and spatial   # 描述子依赖空间图
    extra_cond_ch = (9 if psf_desc else 3) if spatial else 0
    gate = bool(rf.get("mask_edge_gate", False))
    matte_mode = bool(rf.get("matte_mode", False)) if args.matte is None else args.matte
    # P2a（D32）：残差块类型与宽度。block 'naf' 在同 ch 下约半参，故配 naf 时默认
    # 加宽 ar_mid/iu_mid 到与 film_res 同预算（~707k），做"同参数换块"的公平对照。
    block_type = args.block if args.block is not None else str(rf.get("block_type", "film_res"))
    ar_mid = args.ar_mid if args.ar_mid is not None else int(rf.get("ar_mid", 96))
    iu_mid = args.iu_mid if args.iu_mid is not None else int(rf.get("iu_mid", 48))
    print(f"[train] P1 条件：spatial_cond={spatial} psf_desc={psf_desc} "
          f"(extra_cond_ch={extra_cond_ch})  mask_edge_gate={gate}  "
          f"matte_mode={matte_mode}（Plan B 选择式重渲）  "
          f"block={block_type} ar_mid={ar_mid} iu_mid={iu_mid}", flush=True)

    # ---- 数据：在线合成（训练流 + 4 个固定验证样本）----
    snap = bool(cfg["depth"].get("snap_edges", True))
    synth_kw = dict(crop=args.crop, device=device)
    if args.bg_dirs is not None:
        synth_kw["bg_dirs"] = args.bg_dirs                # 训练/验证共用同一背景池
    ds = SynthBokehDataset(SynthConfig(seed=args.seed, **synth_kw))
    val_ds = SynthBokehDataset(SynthConfig(seed=123, **synth_kw))  # 固定种子=固定验证集
    print(f"[train] 背景池 {len(ds.bg_pool.paths):,} 张图", flush=True)
    # 4 个固定验证样本（训练全程不变，便于跨 checkpoint 对比同一画面）。
    val_samples = []
    for _ in range(4):
        b = make_batch(val_ds, 1, snap, extra_cond_ch, gate, matte_mode)  # batch=1
        val_samples.append(tuple(None if t is None else t[0] for t in b))

    # ---- 网络 / 优化器 / 损失 ----
    net = RefineNet(cond_dim=int(cfg["refine"]["cond_dim"]),
                    extra_cond_ch=extra_cond_ch, matte_mode=matte_mode,
                    block_type=block_type, ar_mid=ar_mid, iu_mid=iu_mid).to(device)
    n_params = count_parameters(net)
    assert n_params < int(cfg["refine"]["max_params_million"] * 1e6), \
        f"参数量 {n_params:,} 超出轻量铁律！"
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    perc = None if args.no_perceptual else build_perceptual(device)
    l1 = torch.nn.L1Loss()

    it0 = 0                                          # 起始迭代（续训时>0）
    if args.resume is not None:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        it0 = int(ck["iter"])
        print(f"[train] 从 {args.resume} 续训（已完成 {it0} 步）", flush=True)

    # AMP：GradScaler 只在 CUDA+amp 下启用；CPU / no-amp 走 None 分支。
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device == "cuda") else None
    print(f"[train] device={device} params={n_params:,} batch={args.batch} "
          f"crop={args.crop} lr={args.lr} iters={args.iters} "
          f"perceptual={'LPIPS-VGG' if perc else 'off'} snap_edges={snap} "
          f"amp={'on' if scaler else 'off'}", flush=True)

    # ---- 循环 ----
    net.train()
    history = []                                     # (iter, l1, perc, l1_phys)
    t0 = time.time()
    for it in range(it0 + 1, args.iters + 1):
        # make_batch 在 autocast 外：物理 PSF/FFT 需要 fp32 精度。
        img, disp, phys, gt, cvec, cond, band, bg, matte_gt = make_batch(
            ds, args.batch, snap, extra_cond_ch, gate, matte_mode)

        # 网络前向 + 损失：可 fp16（autocast 自动管理精度）。
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            B, aux = refine_forward(net, img, disp, phys, cvec, cond, band, bg,
                                    matte_mode)
            # 误差图/边界权重 GT：m_gt = 物理渲染的真实误差（边界≈1、平坦区≈0）。
            m_gt = (args.mask_gain
                    * (phys - gt).abs().mean(dim=1, keepdim=True)).clamp(0.0, 1.0)
            # 边界加权 L1：权重 1+bw·m_gt，除以权重均值保持量纲（与普通 L1 可比）。
            w = 1.0 + args.boundary_weight * m_gt
            loss_l1 = ((B - gt).abs() * w).mean() / w.mean()
            loss_p = torch.zeros((), device=device)
            if perc is not None:
                # LPIPS 期望 [-1,1]；normalize=True 让它自己从 [0,1] 换算。
                loss_p = perc(B.clamp(0, 1), gt, normalize=True).mean()

            if matte_mode:
                # Plan B：matte 直监 BCE-on-logits（D29/D30）。
                import torch.nn.functional as F
                loss_n = torch.zeros((), device=device)
                bm = band
                bce = F.binary_cross_entropy_with_logits(
                    aux["logit"], matte_gt, reduction="none")
                loss_m = (bce * bm).sum() / bm.sum().clamp(min=1.0)
                loss = (loss_l1 + args.lambda_perc * loss_p
                        + args.lambda_matte * loss_m)
                map_mean = float((aux["map"] * bm).sum().detach()
                                 / bm.sum().clamp(min=1.0))
            else:
                loss_n = ((aux["neural"] - gt).abs() * w).mean() / w.mean()
                loss_m = l1(aux["map"], m_gt if band is None else m_gt * band)
                loss = (loss_l1 + args.lambda_perc * loss_p
                        + args.lambda_neural * loss_n + args.lambda_mask * loss_m)
                map_mean = float(aux["map"].mean().detach())

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        with torch.no_grad():
            l1_phys = float(l1(phys, gt))            # 物理渲染基线（诊断用）
        history.append((it, float(loss_l1.detach()), float(loss_p.detach()),
                        l1_phys, float(loss_n.detach()), float(loss_m.detach()),
                        map_mean))

        if it % args.log_every == 0 or it == it0 + 1:
            dt = (time.time() - t0) / (it - it0)     # 只按本次会话的步数计速
            mem = (torch.cuda.max_memory_allocated() / 2**30
                   if device == "cuda" else 0.0)
            mname = "ᾱ" if matte_mode else "m̄"     # matte 均值 / 误差图均值
            print(f"[train] it {it:6d}/{args.iters}  L1={loss_l1.detach().item():.4f} "
                  f"(phys基线={l1_phys:.4f})  LPIPS={loss_p.detach().item():.4f}  "
                  f"L1n={loss_n.detach().item():.4f}  Lm={loss_m.detach().item():.4f} "
                  f"{mname}={map_mean:.3f}  "
                  f"{dt:.2f}s/it  峰值显存 {mem:.1f}GB", flush=True)
        if it % args.viz_every == 0:
            save_visualization(net, val_samples,
                               out_dir / f"viz_it{it:06d}.png", device, matte_mode)
            ev = evaluate_boundary(net, val_samples, args.mask_gain, matte_mode)
            print(f"[eval]  it {it:6d}  边界L1: phys={ev['boundary_phys']:.4f} "
                  f"refined={ev['boundary_fused']:.4f} "
                  f"(差值 {ev['boundary_fused'] - ev['boundary_phys']:+.4f}，负=净收益)",
                  flush=True)
        if it % args.save_every == 0 or it == args.iters:
            torch.save({"iter": it, "model": net.state_dict(),
                        "optimizer": opt.state_dict(),
                        "args": vars(args), "n_params": n_params,
                        # 架构标记：e2e 推理按此选择 matte 选择式 / v2 整图 / v1 滑窗，
                        # 并按 block/宽度重建网络（P2a，D32）。
                        "extra_cond_ch": extra_cond_ch,
                        "mask_edge_gate": gate, "matte_mode": matte_mode,
                        "block_type": block_type, "ar_mid": ar_mid, "iu_mid": iu_mid},
                       out_dir / "ckpt_latest.pth")

    # ---- 收尾：损失曲线 + 历史落盘 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    its, l1s, ps, l1ph, l1ns, lms, mbars = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # 图内文字用英文（Agg 默认字体无 CJK 字形，中文会变成方框+刷屏警告）。
    ax.plot(its, l1s, label="L1(B, GT)")
    ax.plot(its, l1ph, label="L1(B_phys, GT) physical baseline", ls="--", alpha=0.7)
    if not matte_mode:
        ax.plot(its, l1ns, label="L1(B_neural, GT)", alpha=0.5)
    ax.plot(its, lms, label="matte loss" if matte_mode else "mask loss", alpha=0.5)
    if perc is not None:
        ax.plot(its, ps, label="LPIPS(B, GT)", alpha=0.7)
    ax.set_xlabel("iteration")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("refine net training (B below dashed line = net gain)")
    fig.tight_layout()
    fig.savefig(str(out_dir / "loss_curve.png"), dpi=100)
    plt.close(fig)
    ev = evaluate_boundary(net, val_samples, args.mask_gain, matte_mode)
    print(f"[eval]  最终边界L1: phys={ev['boundary_phys']:.4f} "
          f"fused={ev['boundary_fused']:.4f} "
          f"(差值 {ev['boundary_fused'] - ev['boundary_phys']:+.4f}，负=净收益)", flush=True)
    (out_dir / "history.json").write_text(json.dumps(
        {"columns": ["iter", "l1", "lpips", "l1_phys", "l1_neural",
                     "mask_loss", "mask_mean"], "rows": history,
         "final_eval": ev}))
    print(f"[train] 完成。checkpoint/曲线/可视化 -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
