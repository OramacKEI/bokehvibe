"""
depth/estimator.py
==================
深度估计封装 —— 管线最上游（CLAUDE.md 第 2 节）。

职责：单张全焦图像 I  ──►  归一化相对【视差】D ∈ [0,1]（近大远小）。

================================================================================
【为什么是"视差"而不是"深度"？—— 常驻概念坑，务必牢记（CLAUDE.md 第 12 节）】
--------------------------------------------------------------------------------
后续的带符号散焦圈公式是：
        r = K · |D − d_f|     ，符号由 (D − d_f) 决定（前景/背景）
这里的 D 必须是【视差(disparity)：近处值大、远处值小】，**不是物理深度**。
Depth Anything V2 的原始输出恰好就是"仿射不变的逆深度"，即视差，方向正确（近大远小），
我们只需把它线性归一化到 [0,1]，**不需要翻转方向**。

================================================================================
【后端无关设计 —— 本文件的核心架构决策】
--------------------------------------------------------------------------------
深度网在本项目里是【冻结、仅推理】的模块（绝不训练），而且对合成数据可以【离线预计算】。
因此"换一个更强的深度模型"应当是低成本的消融/升级操作（对应 CLAUDE.md 第 9 节第 6 项消融、
第 11 节风险"深度质量限制边界"）。

为此我们定义一个统一抽象接口 `DepthBackend`，所有后端都只对外暴露一个方法：
        infer_disparity(image) -> np.ndarray   # 归一化视差 [0,1]，近=1 远=0
管线其余部分只依赖这个接口，不关心底层是哪个模型。

- 默认后端：`DepthAnythingV2Backend`（默认 **DA V2-Large / vitl，335M，input_size=770**，
  边界/细结构质量明显优于 Base/Small，D37；冻结仅推理、不算可训练参数。
  encoder='vitb'(Base) / 'vits'(Small/Apache-2.0) 可切回）。
- 备选后端：`DepthAnything3Backend`（DA3，占位 stub）。DA3 更强但更重（Small 80M / Mono-Large
  350M），定位偏多视图几何、输出是 depth-ray 表示，需额外转换；仅当边界质量成为瓶颈时再考虑，
  详见该类的说明。

================================================================================
【依赖与环境】
--------------------------------------------------------------------------------
本文件用到 torch / opencv，它们在专用 conda env `bokeh` 里。为了在环境尚未就绪时
仍能 import 本模块做静态检查/阅读，所有重量级依赖(torch/cv2)采用【惰性导入】
（在函数/方法内部 import），而不是在文件顶部 import。numpy 在 base 环境已有，可顶部导入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np  # numpy 在 base 环境就有，安全；torch/cv2 则惰性导入

# ------------------------------------------------------------------------------
# 路径常量：从本文件位置反推项目根，定位上游仓库与权重，避免依赖"当前工作目录"。
# ------------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent                       # /home/jing/bokeh
DA_V2_REPO = PROJECT_ROOT / "third_party" / "Depth-Anything-V2"


def _da_v2_ckpt(encoder: str) -> Path:
    """编码器档位 → 权重文件路径（checkpoints/depth_anything_v2_{encoder}.pth）。"""
    return PROJECT_ROOT / "checkpoints" / f"depth_anything_v2_{encoder}.pth"


DA_V2_CKPT = _da_v2_ckpt("vits")     # 向后兼容旧名（Small 权重路径）

# DA V2 各编码器(encoder)对应的网络结构超参（取自上游 run.py）。
# 默认 'vitb'(=Base,97M)：推理深度的全局关系/主体内部均匀性明显优于 Small（D25，
# 真实图核图实测）；冻结仅推理、不算可训练参数 → 不碰 RQ2 轻量性。
# 许可：'vits'=Apache-2.0；'vitb'/'vitl'=CC-BY-NC-4.0（学术非商用可用，论文须声明）。
_DA_V2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# 接受的输入类型：图片路径(str/Path) 或 一张 RGB 的 numpy 图(H,W,3, uint8)。
ImageInput = Union[str, Path, "np.ndarray"]


def _pick_device(device: str | None) -> str:
    """选择计算设备。None 时优先 cuda，其次 cpu（GPU 未修复时仍可在 cpu 上跑通流程）。"""
    if device is not None:
        return device
    import torch  # 惰性导入
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_to_disparity01(raw: "np.ndarray", clip_pct: float = 0.5) -> "np.ndarray":
    """把后端的原始输出线性归一化到 [0,1]，语义=视差(近=1, 远=0)。

    DA V2 原始输出是仿射不变逆深度(值越大越近)，归一化后最近≈1、最远≈0。

    【为何用百分位而非裸 min/max】裸 min-max 对【离群像素】敏感：一个极近的反光点
    或一个极远的天空像素会独占 0/1 端，把其余内容压到很窄的区间 → 视差尺度不稳，
    而 CoC 公式 r=K·|D−d_f| 依赖绝对视差尺度 → 同一物理深度的虚化量在不同图间漂移。
    改用 [clip_pct, 100−clip_pct] 百分位定标定范围、再裁剪到 [0,1]：对离群稳健，
    尺度一致（与 auto_focus_params 用百分位算 d_f/K/tol 的口径也更自洽）。
    """
    raw = raw.astype(np.float32)
    lo = float(np.percentile(raw, clip_pct))
    hi = float(np.percentile(raw, 100.0 - clip_pct))
    if hi - lo < 1e-8:
        # 退化情形（全图近同值），返回全 0，避免除零。
        return np.zeros_like(raw)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


# ==============================================================================
# 抽象基类：所有深度后端的统一接口。
# ==============================================================================
class DepthBackend:
    """深度后端抽象基类。子类只需实现 `infer_disparity`。

    约定（所有后端必须遵守）：
      - 输入：一张图（路径或 RGB numpy 图）。
      - 输出：np.ndarray，形状 (H, W)，dtype float32，取值 [0,1]，语义=视差(近=1, 远=0)。
      - 模型权重【冻结】：构造时即置 eval()、关梯度，绝不参与训练。
    """

    name: str = "abstract"

    def infer_disparity(self, image: ImageInput) -> "np.ndarray":
        raise NotImplementedError("子类需实现：返回归一化视差 [0,1]，近=1")


# ==============================================================================
# 默认后端：Depth Anything V2-Large (vitl)。encoder='vitb'/'vits' 切回 Base/Small。
# ==============================================================================
class DepthAnythingV2Backend(DepthBackend):
    """封装冻结的 DA V2 推理（默认 Large/vitl）。

    用法：
        backend = DepthAnythingV2Backend()          # 默认 Large/vitl；加载一次反复调用
        disp = backend.infer_disparity("a.jpg")     # -> (H,W) float32, [0,1], 近=1
    """

    name = "depth_anything_v2"

    def __init__(
        self,
        encoder: str = "vitl",
        device: str | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        # ---- 惰性导入重量级依赖 ----
        import sys
        import torch

        self.encoder = encoder
        self.device = _pick_device(device)

        # 把上游仓库目录加入 import 搜索路径，才能 `from depth_anything_v2.dpt import ...`。
        # 用 insert(0) 确保优先命中我们克隆的这份实现。
        repo = str(DA_V2_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from depth_anything_v2.dpt import DepthAnythingV2  # 来自 third_party/Depth-Anything-V2

        if encoder not in _DA_V2_CONFIGS:
            raise ValueError(
                f"不支持的 encoder: {encoder}；可选 'vits'(Small)/'vitb'(Base)/'vitl'(Large)")

        # ---- 构建网络并载入权重（ckpt_path 未指定时按 encoder 自动选档）----
        ckpt = Path(ckpt_path) if ckpt_path is not None else _da_v2_ckpt(encoder)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"找不到 DA V2 权重：{ckpt}\n"
                f"请先下载 {encoder} 权重到该路径"
                f"（HuggingFace depth-anything/Depth-Anything-V2-*，见 PROJECT_STATUS.md）。"
            )

        self.model = DepthAnythingV2(**_DA_V2_CONFIGS[encoder])
        # map_location='cpu' 先把权重读到 CPU，再统一搬到目标设备，避免设备不匹配报错。
        state = torch.load(str(ckpt), map_location="cpu")
        self.model.load_state_dict(state)

        # ---- 冻结：eval 模式 + 关闭所有参数梯度。本模块永不训练。----
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    def _read_image_bgr(self, image: ImageInput) -> "np.ndarray":
        """把多种输入统一成 DA V2 期望的 BGR uint8 图。

        DA V2 的 infer_image 内部会做 BGR->RGB，所以这里必须交给它 BGR：
          - 传入路径：用 cv2.imread 读，本身就是 BGR。
          - 传入 numpy：约定为 RGB（最常见），翻转最后一维转成 BGR。
        """
        import cv2

        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"无法读取图片：{image}")
            return bgr
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"期望 (H,W,3) 的 RGB 图，得到 shape={image.shape}")
            # 约定 numpy 输入为 RGB，转 BGR 给模型。
            return image[:, :, ::-1].copy()
        raise TypeError(f"不支持的输入类型：{type(image)}")

    def infer_disparity(self, image: ImageInput, input_size: int = 770) -> "np.ndarray":
        """I -> 归一化视差 D ∈ [0,1]（近=1, 远=0）。已在 no_grad 下推理。

        Args:
            image: 图片路径，或 RGB 的 (H,W,3) uint8 numpy 图。
            input_size: DA V2 内部 resize 的短边目标（须为 14 的倍数）。**默认 770**（>官方 518）：
                散景对【边界/细结构】敏感，更高 input_size 让烟囱/树/细枝等深度更细（D37 实测
                Large@768 明显优于 Base@518）。代价：CPU 略慢（~10s vs 1s），GPU 几无感。

        Returns:
            (H, W) float32，取值 [0,1]，与输入同分辨率。语义=视差，可直接喂 CoC 公式。
        """
        import torch

        bgr = self._read_image_bgr(image)
        with torch.no_grad():
            # infer_image: 内部完成 预处理->前向->插值回原分辨率，返回 (H,W) 的 numpy 原始视差。
            raw = self.model.infer_image(bgr, input_size=input_size)
        return _normalize_to_disparity01(raw)


# ==============================================================================
# 备选后端：Depth Anything 3（占位 stub，暂不实现）。
# ==============================================================================
class DepthAnything3Backend(DepthBackend):
    """DA3 后端（占位）。仅当"深度边界质量成为瓶颈"时再考虑启用（消融/升级）。

    启用前需注意（详见调研结论，2026-06）：
      - 许可：DA3-Small(80M)/Base(120M)/Mono-Large(350M)/Metric-Large 为 Apache-2.0；
        DA3-Large/Giant 为 CC-BY-NC（非商用），不可用于本项目。
      - 输出：DA3 用 depth-ray 表示，定位多视图几何，单图深度需做转换并【确认方向】
        （务必复用 _normalize_to_disparity01 的"近=1"约定，必要时翻转符号）。
      - 依赖：需额外安装 xformers + torch>=2，比 V2 重。
      - 轻量性：80M~350M 远大于 V2-Small 的 25M，会推高推理显存/时延，削弱"轻量"卖点(RQ2)。

    实现时：把 DA3 的单图输出转成 (H,W) 视差并归一化到 [0,1]，对齐基类约定即可，
    管线其余部分无需改动 —— 这正是"后端无关"设计的价值。
    """

    name = "depth_anything_3"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DA3 后端暂未实现：默认用 V2-Small。若边界质量成瓶颈再启用，"
            "候选 DA3-Small / DA3Mono-Large（均 Apache-2.0）。"
        )


# ==============================================================================
# 工厂函数 + 便捷入口
# ==============================================================================
_BACKENDS = {
    "depth_anything_v2": DepthAnythingV2Backend,
    "depth_anything_3": DepthAnything3Backend,
}


def load_depth_backend(name: str = "depth_anything_v2", **kwargs) -> DepthBackend:
    """按名字加载深度后端。默认 DA V2-Small。

    Args:
        name: 'depth_anything_v2'(默认) | 'depth_anything_3'(占位)。
        **kwargs: 透传给后端构造函数（如 device、ckpt_path）。
    """
    if name not in _BACKENDS:
        raise ValueError(f"未知后端 '{name}'；可选：{list(_BACKENDS)}")
    return _BACKENDS[name](**kwargs)


def infer_disparity(image: ImageInput, backend: DepthBackend | None = None) -> "np.ndarray":
    """便捷函数：对单张图出归一化视差。

    注意：每次调用都新建后端会反复加载权重、很慢。批量处理时请先 load_depth_backend()
    拿到 backend 复用，再循环调用 backend.infer_disparity(...)。这里默认现建现用，仅便于快速试跑。
    """
    if backend is None:
        backend = load_depth_backend()
    return backend.infer_disparity(image)


# ==============================================================================
# 最小可视化：把视差存成灰度图，供人工核对方向/质量（CLAUDE.md 第 12 节）。
# ==============================================================================
def save_disparity_visualization(disparity: "np.ndarray", out_path: str | Path) -> None:
    """把 [0,1] 视差图存成灰度 PNG（近处亮/白，远处暗/黑），便于肉眼检查。

    用 matplotlib 的 'gray' 配色——与多数散景论文（及 DA V2 的 `--grayscale` 选项）一致，
    避免发散型彩虹配色(Spectral)误导深度断层/单调顺序的判断。near=1 → 白。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境(服务器/tty)下也能存图
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(disparity, cmap="gray", vmin=0.0, vmax=1.0)
    plt.colorbar(label="disparity  (near=1, far=0)")
    plt.title("Depth Anything V2  (check: near regions should be brighter)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 自测脚本：环境就绪 + GPU 恢复后，可直接 `python depth/estimator.py <图片路径>` 验证。
# ==============================================================================
if __name__ == "__main__":
    import sys

    # 默认拿上游仓库自带的示例图试跑；也可命令行传入自己的图片路径。
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # DA V2 仓库 assets/examples 下有示例图
        examples = sorted((DA_V2_REPO / "assets" / "examples").glob("*.*"))
        if not examples:
            print("未找到示例图，请传入图片路径：python depth/estimator.py <img>")
            sys.exit(1)
        img_path = str(examples[0])

    print(f"[depth] 加载 DA V2 后端（默认 Large/vitl）...")
    backend = load_depth_backend("depth_anything_v2")
    print(f"[depth] device = {backend.device}，推理图片：{img_path}")
    disp = backend.infer_disparity(img_path)
    print(f"[depth] 视差图 shape={disp.shape}, min={disp.min():.3f}, max={disp.max():.3f}")

    out = PROJECT_ROOT / "outputs" / "depth_test" / "disparity_vis.png"
    save_disparity_visualization(disp, out)
    print(f"[depth] 已保存可视化到：{out}（请肉眼核对：近处更亮=方向正确）")
