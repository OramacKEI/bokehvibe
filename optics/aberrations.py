"""
optics/aberrations.py
=====================
像差系数的「定义 / 采样 / 预设」。

本模块只负责【数据结构】与【随机采样 / 风格预设】，不做任何物理计算
（物理在 pupil.py / psf.py）。它定义了贯穿整个项目的控制向量 `c` 中
与镜头像差相关的那一部分 `a`。

对照 CLAUDE.md 第 3 节，控制参数分三组：

1) 波前像差（相位项，单位：波长 λ 的倍数，即 "waves"）
     W020·ρ²            离焦   defocus         （主离焦量由渲染器按层注入，这里留基底偏移）
     W040·ρ⁴            球差   spherical       ← 奶油柔散(<0) / 肥皂泡亮边(>0) 的主控钮
     W131·H·ρ³·cosθ     彗差   coma            （第二档）
     W222·H²·ρ²·cos²θ   像散   astigmatism     （第二档，爆炸焦外）
     W220·H²·ρ²         场曲   field curvature （第二档）
   其中 H = 归一化像高（像点到画面中心的距离，0=中心，~1=边角）。

2) 光瞳振幅（遮罩项，与相位无关）
     n_blades                  光圈叶片数（0=圆形）        → 多边形散景
     (vignette_strength, R_v)  随像高平移的猫眼截断        → 旋焦 / 单侧亮边
   （洋葱圈/非球面周期扰动已移出研究范围，见 DECISIONS D15）

3) 色差（逐通道，默认启用，成本仅 3× PSF）
     loca_rgb：R/G/B 各自的 W020 微小偏移 → 纵向色差 LoCA（焦前偏紫/焦后偏绿自动涌现）
     laca_rgb：R/G/B 的 PSF 随像高的【径向平移】量 → 横向色差 LaCA
       （物理：LaCA 本质是各通道放大率不同 → 离轴处各通道 PSF 中心彼此错开，
         是"平移"而非"缩放"——缩放更接近球色差，渲染不出真实的单侧紫/绿镶边。
         约定与猫眼一致：PSF 字典里沿 +x 平移，渲染时随方位角一起旋转。）

================================================================================
【可微性说明】
本文件用普通 Python float 定义系数，便于配置/采样/序列化。
真正参与 torch 计算时（pupil.py / psf.py），这些 float 会与 torch 张量做运算，
结果自动是张量。若要做【镜头指纹标定】（对系数做梯度下降去匹配目标镜头），
只需把对应字段替换成 `requires_grad=True` 的标量张量即可——torch 算子会自动建图，
无需改物理代码。这正是"全程 torch、避免不可微操作"的好处（CLAUDE.md 第 12 节）。

【独立采样 = 解耦监督】
sample_random() 让每个系数从各自分布【独立】抽取，从而不同效应在训练对里天然不相关，
这是 RQ1 可控性/解耦的监督来源（CLAUDE.md 第 7、8 节）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class AberrationCoeffs:
    """镜头像差控制向量 `a`。波前系数单位均为「波长倍数(waves)」。

    字段值可以是 Python float（默认，用于配置/采样），也可以替换为 torch 标量张量
    （用于可微的镜头指纹标定）——pupil.py 的算子对两者都成立。
    """

    # ---------------- 波前像差（相位）----------------
    W020_defocus: float = 0.0      # 离焦基底偏移；主离焦量由渲染器按层注入（D9 现场算 PSF）
    W040_spherical: float = 0.0    # 球差：第一档核心钮。<0 奶油柔散；>0 肥皂泡亮边；符号翻转→前/背景形态反转
    W131_coma: float = 0.0         # 彗差（第二档）
    W222_astigmatism: float = 0.0  # 像散（第二档，强值→爆炸焦外）
    W220_field_curv: float = 0.0   # 场曲（第二档）

    # ---------------- 光瞳振幅（遮罩）----------------
    n_blades: int = 0              # 光圈叶片数；0=圆形光圈（≥3 才是多边形）
    blade_rotation: float = 0.0    # 多边形整体旋转角（弧度），控制散景朝向
    vignette_strength: float = 0.0 # 猫眼强度：随像高 H 的光瞳平移量系数（0=无口径蚀）
    vignette_radius: float = 1.3   # 口径蚀第二孔径半径 R_v（相对光瞳半径；越小切得越狠）

    # ---------------- 色差（逐通道）----------------
    # LoCA：在各通道的 W020 上叠加的偏移量（waves）。典型让 R、B 相对 G 反向偏移。
    loca_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # LaCA：各通道 PSF 在像高 H=1 处沿径向(+x 约定)的平移量，单位=PSF 网格像素。
    # 实际平移 = laca_rgb[c]·H。典型 R 为正、B 为负（红外移/蓝内移），G 取 0 作基准。
    laca_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        # 合法性检查：叶片数要么 0（圆），要么 ≥3（多边形）。
        if self.n_blades not in (0,) and self.n_blades < 3:
            raise ValueError(f"n_blades 应为 0(圆) 或 ≥3，得到 {self.n_blades}")

    def replace(self, **kwargs) -> "AberrationCoeffs":
        """返回一个仅修改指定字段的副本（用于解耦实验：单独扰动一个系数）。"""
        return replace(self, **kwargs)


# ==============================================================================
# 随机采样：在线合成训练数据时为每对样本独立采样一组系数。
# ==============================================================================
def sample_random(rng=None) -> AberrationCoeffs:
    """独立采样一组像差系数。各系数从各自分布独立抽取 → 训练对天然解耦。

    Args:
        rng: numpy.random.Generator；None 时新建一个（便于复现可外部传入带种子的 rng）。

    Returns:
        AberrationCoeffs 实例。

    注：这里的分布范围是 M1 的初始经验值，后续会在物理验证 / 训练中校准。
        范围设得"温和"以避免一上来就产生非物理的极端 PSF。
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    # 叶片数：圆形或常见的 5~9 边形光圈。
    n_blades = int(rng.choice([0, 5, 6, 7, 8, 9]))

    return AberrationCoeffs(
        # 波前像差：球差是主角，给较宽范围（含正负）；其余第二档给较窄范围。
        W020_defocus=float(rng.uniform(-0.5, 0.5)),
        W040_spherical=float(rng.uniform(-2.0, 2.0)),
        W131_coma=float(rng.uniform(-1.0, 1.0)),
        W222_astigmatism=float(rng.uniform(0.0, 1.5)),
        W220_field_curv=float(rng.uniform(-1.0, 1.0)),
        # 光圈/口径蚀
        n_blades=n_blades,
        blade_rotation=float(rng.uniform(0.0, 2 * np.pi)),
        vignette_strength=float(rng.uniform(0.0, 0.6)),
        vignette_radius=float(rng.uniform(1.1, 1.5)),
        # 色差：让 R 与 B 相对 G 反向小偏移，模拟 LoCA；LaCA 为 H=1 处的径向平移(px)，
        # R 外移(+)、B 内移(−)，幅度 0~2px 量级（相对 crop≈96-128 的 PSF 网格是细微镶边）。
        loca_rgb=(float(rng.uniform(0.0, 0.4)), 0.0, float(rng.uniform(-0.4, 0.0))),
        laca_rgb=(float(rng.uniform(0.0, 2.0)), 0.0, float(rng.uniform(-2.0, 0.0))),
    )


# ==============================================================================
# 风格预设：CLAUDE.md 第 3 节"效果如何由参数涌现"那张表的直接落地。
# 用于物理验证脚本：逐个渲染预设，肉眼核对是否涌现对应焦外风格。
# ==============================================================================
PRESETS: dict[str, AberrationCoeffs] = {
    # 理想镜头：无像差，圆形光圈 → 均匀亮圆散景（基准对照）
    "ideal": AberrationCoeffs(),

    # 奶油柔散：球差欠矫正 W040<0 → 中心亮、边缘柔的"奶油"散景
    "cream_soft": AberrationCoeffs(W040_spherical=-1.5),

    # 肥皂泡亮边环：球差过矫正 W040>0 → 边缘亮环
    "soap_bubble": AberrationCoeffs(W040_spherical=+1.5),

    # 多边形散景：6 叶片光圈
    "hexagon": AberrationCoeffs(n_blades=6),

    # 旋焦 swirl：强口径蚀(猫眼)，在像高 H>0 处显现（需在非零 H 渲染才看得到）
    "swirl_catseye": AberrationCoeffs(vignette_strength=0.5, vignette_radius=1.2),

    # 双高斯单侧亮边：球差亮环 × 视场相关猫眼（两旋钮乘积，需 H>0）
    "double_gauss_edge": AberrationCoeffs(W040_spherical=+1.2, vignette_strength=0.5, vignette_radius=1.2),
}
