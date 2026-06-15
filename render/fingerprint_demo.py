"""
render/fingerprint_demo.py
==========================
【镜头指纹标定原型】RQ3 的第一份可行性证据（CLAUDE.md 第 1 节）。

实验设计（自标定 / self-calibration，闭环验证）：
    1. 用一组【已知】控制参数 c* = (d_f*, K*, W040*) 渲染一张"目标镜头"散景图 B*。
    2. 把参数初始化到【错误值】（连符号都给错），冻结图像与深度，
       仅对 c 做梯度下降，最小化  L1( render(c), B* )。
    3. 看参数能否收敛回 c* —— 能，则"从样张反求像差系数"的可微链路成立。

为什么先做合成目标而不是真实样张：
    合成目标有真值可对照，能干净地回答"链路通不通、损失面好不好走"；
    真实样张标定（Helios/Trioplan）是 M4 的事，需要先过这一关。

运行：
    python -m render.fingerprint_demo
产物（outputs/fingerprint_test/）：
    convergence.png      参数轨迹（对照真值虚线）+ 损失曲线
    comparison.png       目标 / 初始 / 收敛后 三图对比
    result.json          数值结果（真值/初值/恢复值/相对误差）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "fingerprint_test"
DEFAULT_IMG = PROJECT_ROOT / "third_party" / "BokehMe" / "inputs" / "21.jpg"

# 真值（"目标镜头"）与初值（故意错：W040 连符号都反）
TRUE = {"d_f": 0.24, "K": 30.0, "W040": +1.5}
INIT = {"d_f": 0.55, "K": 12.0, "W040": -0.8}

# 【分阶段优化】（吃过的亏，记入 DECISIONS：联合优化会掉进局部极小——
#  优化器用"全局球差模糊 W040"冒充"深度相关离焦模糊 K"，K 塌缩到下限。
#  对策与标定文献一致：先拟合低阶项 (d_f, K)（像差冻结为 0），再放开像差。）
STAGE1_ITERS = 120   # 只优化 d_f, K（W040 冻结在 0）
STAGE2_ITERS = 300   # 三参数联合延拓（W040 从 0 出发，见 run 处注释）


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from depth.estimator import load_depth_backend
    from optics.aberrations import AberrationCoeffs
    from render.demo import load_image
    from render.renderer import RenderControl, render

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG

    # ---- 0) 素材：小分辨率（优化要跑几百次前向，384 宽足够看清结构）----
    rgb = load_image(img_path, max_width=384)
    backend = load_depth_backend("depth_anything_v2", device=device)
    disp_np = backend.infer_disparity((rgb * 255).astype("uint8"))
    image = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)
    disparity = torch.from_numpy(disp_np).to(device)
    print(f"[fingerprint] device={device}, image={img_path.name}, size={tuple(image.shape)}")

    def make_ctrl(d_f, K, w040, n_layers=16):
        coeffs = AberrationCoeffs().replace(W040_spherical=w040)
        return RenderControl(focus_disparity=d_f, aperture_K=K, coeffs=coeffs,
                             n_layers=n_layers)

    # ---- 1) 渲染目标图 B*（真值参数，no_grad）----
    with torch.no_grad():
        target = render(image, disparity, make_ctrl(**{
            "d_f": TRUE["d_f"], "K": TRUE["K"], "w040": TRUE["W040"]}),
            field_varying=False)

    # ---- 2) 待优化参数（初值故意偏离）----
    d_f = torch.tensor(INIT["d_f"], device=device, requires_grad=True)
    K = torch.tensor(INIT["K"], device=device, requires_grad=True)
    w040 = torch.tensor(INIT["W040"], device=device, requires_grad=True)

    with torch.no_grad():
        init_render = render(image, disparity, make_ctrl(d_f, K, w040),
                             field_varying=False).clone()

    # ---- 3) 分阶段优化 ----
    hist = {"loss": [], "d_f": [], "K": [], "W040": []}

    def run_stage(params_lr: list, iters: int, w040_frozen: bool, tag: str):
        opt = torch.optim.Adam(params_lr)
        for it in range(iters):
            opt.zero_grad()
            w = torch.zeros((), device=device) if w040_frozen else w040
            out = render(image, disparity, make_ctrl(d_f, K, w), field_varying=False)
            loss = (out - target).abs().mean()
            loss.backward()
            opt.step()
            # 投影回物理合法范围（projected gradient，比重参数化简单直接）。
            with torch.no_grad():
                d_f.clamp_(0.02, 0.98)
                K.clamp_(2.0, 80.0)
                w040.clamp_(-3.0, 3.0)
            hist["loss"].append(float(loss.detach()))
            hist["d_f"].append(float(d_f.detach()))
            hist["K"].append(float(K.detach()))
            hist["W040"].append(0.0 if w040_frozen else float(w040.detach()))
            if it % 25 == 0 or it == iters - 1:
                print(f"  [{tag}] iter {it:4d}  loss={hist['loss'][-1]:.5f}  "
                      f"d_f={hist['d_f'][-1]:.3f}  K={hist['K'][-1]:.1f}  "
                      f"W040={hist['W040'][-1]:+.3f}")

    t0 = time.time()
    # 阶段 1：只拟合低阶项 (d_f, K)，像差冻结为 0 —— 离焦/光圈主导整体模糊分布。
    # 注意：若目标镜头球差强，纯离焦模型会用"压低 d_f 让全图都虚"去模仿柔光，
    # (d_f,K) 因此带模型失配偏差 —— 必须靠阶段 2 的联合延拓修回来。
    run_stage([{"params": [d_f], "lr": 2e-2}, {"params": [K], "lr": 1.0}],
              STAGE1_ITERS, w040_frozen=True, tag="stage1 d_f,K")
    # 阶段 2：【延拓法 continuation】W040 从阶段 1 的值(0)出发——不是从随机/错误
    # 初值出发（那会把联合优化拽回坏盆地）。三参数全开、保持正常学习率，
    # 让 (d_f, W040) 沿耦合方向一起爬回真值（d_f 升 → 主体变锐，W040 升 → 补回柔光）。
    with torch.no_grad():
        w040.zero_()
    run_stage([{"params": [d_f], "lr": 2e-2}, {"params": [K], "lr": 1.0},
               {"params": [w040], "lr": 5e-2}],
              STAGE2_ITERS, w040_frozen=False, tag="stage2 +W040")
    dt = time.time() - t0
    n_total = STAGE1_ITERS + STAGE2_ITERS
    print(f"[fingerprint] {n_total} 次迭代用时 {dt:.1f}s（{dt / n_total * 1000:.0f} ms/iter）")

    # ---- 4) 结果落盘 ----
    rec = {"d_f": float(d_f), "K": float(K), "W040": float(w040)}
    rel_err = {k: abs(rec[k] - TRUE[k]) / (abs(TRUE[k]) + 1e-9) for k in TRUE}
    result = {"true": TRUE, "init": INIT, "recovered": rec,
              "rel_err": rel_err, "final_loss": hist["loss"][-1],
              "iters": n_total, "seconds": dt}
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[fingerprint] 恢复值: {rec}")
    print(f"[fingerprint] 相对误差: " + ", ".join(f"{k}={v * 100:.1f}%" for k, v in rel_err.items()))

    # 收敛曲线
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, key in zip(axes[:3], ["d_f", "K", "W040"]):
        ax.plot(hist[key], label="estimate")
        ax.axhline(TRUE[key], ls="--", c="r", label="true")
        ax.set_title(key); ax.set_xlabel("iter"); ax.legend(); ax.grid(alpha=0.3)
    axes[3].semilogy(hist["loss"]); axes[3].set_title("L1 loss"); axes[3].set_xlabel("iter")
    axes[3].grid(alpha=0.3)
    fig.suptitle("Lens fingerprint self-calibration: recover (d_f, K, W040) by gradient descent")
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "convergence.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 三图对比
    with torch.no_grad():
        final_render = render(image, disparity, make_ctrl(d_f, K, w040),
                              field_varying=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, (name, im) in zip(axes, [("target (true lens)", target),
                                     ("init (wrong params)", init_render),
                                     ("recovered", final_render)]):
        ax.imshow(im.clamp(0, 1).cpu().numpy().transpose(1, 2, 0))
        ax.set_title(name, fontsize=10); ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "comparison.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[fingerprint] -> {OUT_DIR}/convergence.png, comparison.png, result.json")


if __name__ == "__main__":
    main()
