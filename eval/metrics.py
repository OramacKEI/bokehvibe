"""
eval/metrics.py
===============
评测协议（CLAUDE.md 第 9 节）。六个维度：

1. 可控性保真度：给定 (d_f, K, a) 对合成 GT 算 PSNR / SSIM / LPIPS。
2. 解耦度 (cross-talk)：单独改一个系数，度量其它效应签名的非预期变化。
3. 真实感：RealBokeh / EBB! Val294 指标 + 小规模 user study。
4. 效率：可训练参数量、FLOPs、时延、显存（对比 BokehMe++/Bokehlicious/BokehDiff）。
5. 镜头指纹：对双高斯单侧亮边、Helios 旋焦、Trioplan 肥皂泡的定性复刻。
6. 消融：去像差控制 / 去视场相关 / 去细化网 / 去真实微调。

TODO(M3): 逐项实现指标函数与汇总报告。
"""

from __future__ import annotations


def psnr_ssim_lpips(pred, target):
    raise NotImplementedError("M3：实现保真度指标")


def crosstalk(model, base_coeffs):
    """解耦度：扰动单个系数，测量其它效应签名的非预期变化。

    实现提示（M3）：**直接复用** optics/decoupling.py 的那套效应签名
    （ring_ratio / skewness / elongation / transmission），把"在 PSF 上测"
    换成"在渲染图像的散景区域上测"——同一套签名跨 PSF/图像两个层面使用，
    正是 CLAUDE.md 第 9 节第 2 条的设计意图。
    """
    raise NotImplementedError("M3：实现解耦度/串扰度量（复用 optics/decoupling 签名）")
