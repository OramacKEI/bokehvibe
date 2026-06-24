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
     W040·ρ⁴            初级球差 spherical     ← 奶油柔散(<0) / 肥皂泡亮边(>0) 的主控钮
     W060·ρ⁶            高阶球差 spherical-2   ← 盘内带状结构 / 真 nisen 二线（D34，对照 Wu2010）
     W131·H·ρ³·cosθ     彗差   coma            （第二档）
     W222·H²·ρ²·cos²θ   像散   astigmatism     （第二档，爆炸焦外）
     W220·H²·ρ²         场曲   field curvature （第二档）
   其中 H = 归一化像高（像点到画面中心的距离，0=中心，~1=边角）。

2) 光瞳振幅（遮罩项，与相位无关）
     n_blades                  光圈叶片数（0=圆形）        → 多边形散景
     (vignette_strength, R_v)  随像高平移的猫眼截断        → 旋焦 / 单侧亮边
     apodization               径向透过率衰减 exp(−a·ρ²)   → 变迹(STF/APD)：盘边柔化无硬边
   （洋葱圈/非球面周期扰动已移出研究范围，见 DECISIONS D15）

3) 色差（逐通道，默认启用，成本仅 3× PSF）
     loca_rgb：R/G/B 各自的 W020 微小偏移 → 纵向色差 LoCA（【绿-品红轴】：焦前品红/焦后绿自动涌现）
     laca_rgb：R/G/B 的 PSF 随像高的【径向平移】量 → 横向色差 LaCA
       （物理：LaCA 本质是各通道放大率不同 → 离轴处各通道 PSF 中心彼此错开，
         是"平移"而非"缩放"——缩放更接近球色差，渲染不出真实的单侧紫/绿镶边。
         约定与猫眼一致：PSF 字典里沿 +x 平移，渲染时随方位角一起旋转。）
     spherochrom_rgb：R/G/B 各自的 W040(球差) 偏移 → 球色差（盘边亮环逐通道不同 →
         紫/绿镶边随离焦变化；是 LoCA"只移盘心"的形态版补充）

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
    W040_spherical: float = 0.0    # 初级球差：第一档核心钮。<0 奶油柔散；>0 肥皂泡亮边；符号翻转→前/背景形态反转
    W060_spherical2: float = 0.0   # 高阶(5阶/二级)球差 ρ⁶：盘内【带状】结构（亮环不在最外缘而在中途，
                                   # 或"暗心+亮环+亮核"复合形态）→ 真实 nisen 二线 / 精确复刻具体镜头（D34，对照 Wu2010 Fig7）
    W131_coma: float = 0.0         # 彗差（第二档）
    W222_astigmatism: float = 0.0  # 像散（第二档，强值→爆炸焦外）
    W220_field_curv: float = 0.0   # 场曲（第二档）

    # ---------------- 光瞳振幅（遮罩）----------------
    n_blades: int = 0              # 光圈叶片数；0=圆形光圈（≥3 才是多边形）
    blade_rotation: float = 0.0    # 多边形整体旋转角（弧度），控制散景朝向
    blade_curvature: float = 0.0   # 光圈叶片【边缘曲率】∈[0,1]：0=直边正多边形，1=圆。
                                   # 真实光圈叶片是弧形金属片→边向外凸、角仍尖（非直线）。
                                   # 中间值(~0.2~0.4)=真实可调光圈的微凸边，散景更柔和自然。
    vignette_strength: float = 0.0 # 猫眼强度：随像高 H 的光瞳平移量系数（0=无口径蚀）
    vignette_radius: float = 1.3   # 口径蚀第二孔径半径 R_v（相对光瞳半径；越小切得越狠）
    # 变迹 apodization（STF/APD 镜头）：光瞳透过率随半径平滑衰减 T(ρ)=exp(−apod·ρ²)。
    # >0 → 边缘透光渐暗 → 散景盘【边缘柔化、无硬边】（Sony STF / Fuji APD 招牌"奶油"焦外，
    # 与球差欠矫的奶油是【不同机制】：变迹改振幅、球差改相位）。0=无变迹（均匀通光）。
    apodization: float = 0.0

    # ---------------- 色差（逐通道）----------------
    # LoCA：在各通道的 W020 上叠加的偏移量（waves）。取【绿-品红二级光谱轴】：R、B 同向、
    # G 相对它们【反向】偏移 → 焦后【绿】边 / 焦前【品红】边——真实摄影最标志性的 LoCA
    # bokeh fringing（对应复消色差镜头的剩余二级光谱，G 与 R/B 分离）。约定 (+a, −a, +a)：
    # 焦后(W020<0) G 离焦最大→盘边偏绿、R+B 盘小→盘心品红；焦前自动翻转成品红边。
    # 【注意】不要用 (R正,0,B负) 的红-蓝轴——那是简单消色差的单调色散，渲不出标志性绿/品红色晕。
    # loca_rgb[0]=a 仍作 ctrl_vec 的 LoCA 强度标量代表（值越大 LoCA 越强）。
    loca_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # LaCA：各通道 PSF 在像高 H=1 处沿径向(+x 约定)的平移量，单位=PSF 网格像素。
    # 实际平移 = laca_rgb[c]·H。典型 R 为正、B 为负（红外移/蓝内移），G 取 0 作基准。
    laca_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 球色差 spherochromatism：各通道在 W040(球差) 上叠加的偏移量（waves）。
    # 物理：球差随波长变 → 各通道散景盘的【亮边/亮心强度不同】→ 散景盘【边缘】的
    # 紫/绿镶边随离焦变化（比 LoCA 的整体偏移更真实——LoCA 只移盘心、不改盘的形态）。
    # 典型让 R、B 相对 G 反向偏移。是 LoCA 的"形态版"补充。
    spherochrom_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)

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
def sample_random(rng=None, extended: bool = False) -> AberrationCoeffs:
    """独立采样一组像差系数。各系数从各自分布独立抽取 → 训练对天然解耦。

    Args:
        rng: numpy.random.Generator；None 时新建一个（便于复现可外部传入带种子的 rng）。
        extended: 是否采样【ctrl_vec 之外】的新像差（W060 高阶球差 / 变迹 / 球色差）。
            **默认 False（训练用）**：这三者尚未进 ctrl_vec（细化网 FiLM 条件，13 维），
            若在训练里采样会出现"GT 随它们变、网络条件看不到"的失配——尤其【球色差】是
            逐通道色差、连单色描述子图(P1b)也抓不到 → 网络对其全盲、回归均值加噪（W060/变迹
            改 PSF 形状、描述子图能部分捕捉，没那么糟，但一并 gate 以求干净）。
            要让它们成为【可控】维度：扩 ctrl_vec 13→16(+W060/apod/spherochrom_R) + 重训（D36）。
            **demo/指纹/showcase 用 PRESETS 显式设值，不受本开关影响。**

    Returns:
        AberrationCoeffs 实例。

    注：分布范围是 M1 的初始经验值，后续会在物理验证 / 训练中校准。范围设得"温和"避免极端 PSF。
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    # 叶片数：圆形或常见的 5~9 边形光圈。
    n_blades = int(rng.choice([0, 5, 6, 7, 8, 9]))

    # ctrl_vec 外的扩展像差：仅 extended=True 才采样（理由见 docstring / D36）。
    w060 = float(rng.choice([0.0, 0.0, rng.uniform(-1.5, 1.5)])) if extended else 0.0
    apod = float(rng.choice([0.0, 0.0, rng.uniform(1.0, 3.0)])) if extended else 0.0
    sph_rgb = ((float(rng.uniform(0.0, 0.5)), 0.0, float(rng.uniform(-0.5, 0.0)))
               if extended else (0.0, 0.0, 0.0))
    # LoCA 绿-品红轴强度（R/B 同向 +loca_R、G 反向 −loca_R）；loca_R 即 ctrl_vec 的 LoCA 标量。
    loca_R = float(rng.uniform(0.0, 0.4))

    return AberrationCoeffs(
        # 波前像差：球差是主角，给较宽范围（含正负）；其余第二档给较窄范围。
        W020_defocus=float(rng.uniform(-0.5, 0.5)),
        W040_spherical=float(rng.uniform(-2.0, 2.0)),
        W060_spherical2=w060,                              # ctrl_vec 外，默认 0（见 docstring）
        W131_coma=float(rng.uniform(-1.0, 1.0)),
        W222_astigmatism=float(rng.uniform(0.0, 1.5)),
        W220_field_curv=float(rng.uniform(-1.0, 1.0)),
        # 光圈/口径蚀
        n_blades=n_blades,
        blade_rotation=float(rng.uniform(0.0, 2 * np.pi)),
        blade_curvature=float(rng.uniform(0.0, 0.5)),       # 叶片边曲率（真实光圈微凸边）
        vignette_strength=float(rng.uniform(0.0, 0.6)),
        vignette_radius=float(rng.uniform(1.1, 1.5)),
        apodization=apod,                                  # ctrl_vec 外，默认 0（见 docstring）
        # 色差（LoCA/LaCA）：在 ctrl_vec 内（loca_R/laca_R），训练正常采样。
        # LoCA 取【绿-品红轴】(+a,−a,+a)：R/B 同向、G 反向 → 焦后绿/焦前品红（见 loca_rgb 字段注释）；
        # LaCA 为 H=1 处径向平移(px)，R 外移(+)/B 内移(−)（放大率色差，红-蓝单调轴正确，保持不变）。
        loca_rgb=(loca_R, -loca_R, loca_R),
        laca_rgb=(float(rng.uniform(0.0, 2.0)), 0.0, float(rng.uniform(-2.0, 0.0))),
        spherochrom_rgb=sph_rgb,                           # ctrl_vec 外 + 单色描述子抓不到，默认 0
    )


# ==============================================================================
# 风格预设：CLAUDE.md 第 3 节"效果如何由参数涌现"那张表的直接落地。
# 用于物理验证脚本：逐个渲染预设，肉眼核对是否涌现对应焦外风格。
# ==============================================================================
PRESETS: dict[str, AberrationCoeffs] = {
    # 理想镜头：无像差，圆形光圈 → 均匀亮圆散景（基准对照）
    "ideal": AberrationCoeffs(),

    # 奶油柔散（D48）：轻欠矫正球差 W040=-1.0（给一点亮心）+ 变迹 apodization=1.5（柔边填充盘=真虚化）。
    # 修正 D47 纯强球差的问题：纯欠矫正球差本质是"亮心 + 保留焦外结构"（Helios 式【柔化】，非虚化——
    # PSF 中心亮致卷积保留原结构，用户反馈"没虚化只变糊"）；真正的奶油【虚化】靠【变迹】（Sony STF 式
    # 柔边填充盘）。轻球差+变迹兼得：柔虚化 + 略亮心 + 焦内更锐（变迹焦内影响远小于强球差）。
    # 与 stf_apodization（纯强变迹 2.5、无球差）区别：cream 有球差亮心 + 中等变迹。
    "cream_soft": AberrationCoeffs(W040_spherical=-1.0, apodization=1.5),

    # 肥皂泡亮边环（D47 忠实复刻 Trioplan）：强过矫正球差 W040=2.0 + 高阶 W060=0.8
    # → 锐利明亮边缘环 + 较暗填充中心（真实肥皂泡剖面=平底+陡亮环，边/心~8；非旧 1.5 的碗状弱环）。
    # 焦内主体带球差 glow（偏软）是 soap bubble 镜头的真实标志（用户选"忠实复刻"取向，接受焦内软）。
    "soap_bubble": AberrationCoeffs(W040_spherical=+2.0, W060_spherical2=+0.8),

    # 多边形散景：6 叶片光圈，叶片边带真实微凸曲率（0.25，非直边——真实光圈叶片是弧形金属片）
    "hexagon": AberrationCoeffs(n_blades=6, blade_curvature=0.25),

    # 旋焦 swirl：强口径蚀(猫眼)，在像高 H>0 处显现（需在非零 H 渲染才看得到）。
    # D43：vignette_strength 0.5→0.9、radius 1.2→1.05（切得更狠）——旧值在画面边缘猫眼太弱、
    # 旋焦不明显（用户反馈④）；增强后四角散景盘明显被切成朝画面中心的眼形、旋焦感清晰。
    "swirl_catseye": AberrationCoeffs(vignette_strength=0.9, vignette_radius=1.05),

    # 双高斯单侧亮边：球差亮环 × 视场相关猫眼（两旋钮乘积，需 H>0）。D43 同步增强猫眼。
    "double_gauss_edge": AberrationCoeffs(W040_spherical=+1.2, vignette_strength=0.9, vignette_radius=1.05),

    # STF 变迹：径向透过率渐暗 → 散景盘边缘柔化、无硬边（Sony STF 招牌奶油焦外）。
    # 与 cream_soft（球差欠矫，改相位）是不同机制：变迹改振幅，盘内更均匀、仅边缘渐隐。
    "stf_apodization": AberrationCoeffs(apodization=2.5),

    # 球色差：各通道球差不同 → 散景盘边缘随离焦的紫/绿镶边（焦前偏一色、焦后偏另一色）。
    "spherochromatic": AberrationCoeffs(W040_spherical=+0.8,
                                        spherochrom_rgb=(0.4, 0.0, -0.4)),

    # 二线 nisen：高阶球差 W060 与初级 W040 反号 → 盘边【尖锐强亮环】+【暗心】；卷积线条时
    # 线两侧各被亮环描一道 → 双线（中间暗）= 真实二线性的成因（比单纯 W040 的肥皂泡更"硬"更分层，
    # 对照 Wu2010 Fig7）。【物理澄清】大离焦散景盘下几何焦散环必在盘【外缘】（离焦项主导，实测
    # 各 W040/W060 组合峰位均 0.9~0.97·R）——故二线性靠"外缘尖环+暗心"实现，而非"中途环"（后者
    # 仅在中等离焦/接近合焦时可能出现）。W060 越大环越尖、暗心越暗、二线越分明（双线测试
    # outputs/aberration_grid/_nisen_twoline.png）。
    # 【D53 回调 2.0→1.3】D52 曾增到 2.0 求二线更分明，但二线改善微小（中间平台 0.858→0.839）却
    # 显著加剧前后焦盘不对称（W060 高阶项放大）；D53 给 balance 也补偿 W060(系数1.5)后，1.3 已二线
    # 清晰且前后比仅 1.26（接近初级球差），故回 1.3。注：W060(高阶球差)前后焦盘不对称是球差类固有
    # 物理，balance 补偿 W040+W060 后已降到与初级球差相当（不是完全对称——球差本就前后异形）。
    "nisen": AberrationCoeffs(W040_spherical=-1.0, W060_spherical2=+1.3),

    # 【星芒 starburst 不单列预设——它是多边形光圈的衍射表现，与 n_blades 联动】
    # 星芒来自【光阑形状】而非波前像差（≠彗差，彗差是几何的彗星状单侧尾）：多边形光阑的
    # 夫琅禾费衍射 spike，【偶数边 N→N 芒、奇数边→2N 芒】（方阑 4→十字、五边→10、六边→6、
    # 八边→8），spike 方向⊥各边。它与散景盘【同源】：同一个 n_blades 既决定散景盘边数、
    # 又决定星芒芒数；同一个 blade_curvature 既软化盘角、又软化 spike（真实镜头叶片微凸 →
    # 盘角变圆 + 星芒变弱，故现代镜头星芒不锐）。因此【无需独立星芒预设】——任何多边形光圈
    # （如上面 hexagon）切到 psf_mode='wave' + 小离焦亮点即自动涌现对应星芒（geom 几何散射
    # 无衍射 spike；直边 blade_curvature=0 出最锐 spike）。联动演示见
    # render.aberration_grid.render_aperture_starburst_panel。
}
