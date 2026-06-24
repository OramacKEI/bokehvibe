"""
make_boundary_comparison.py
----------------------------
从训练可视化网格图里裁出"物理渲染 vs 细化网络"对比，供幻灯片使用。

默认模式（--mode error）：对比误差图 col4 vs col5，差异最显眼。
可选模式（--mode bokeh）：对比散景图 col0 vs col1，但视觉差异细微。

列顺序（0-based）：
    0: B_phys (renderer)
    1: B refined (fused)
    2: bokeh GT
    3: error mask m
    4: |B_phys-GT|×4   ← error 模式左列
    5: |B-GT|×4        ← error 模式右列

用法：
    python scripts/make_boundary_comparison.py
    python scripts/make_boundary_comparison.py --mode bokeh
    python scripts/make_boundary_comparison.py --input outputs/train_run3/viz_it033000.png
    python scripts/make_boundary_comparison.py --row 3
    python scripts/make_boundary_comparison.py --crop-x 120 --crop-y 80
"""

import argparse
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Canny 替代：用 scipy 梯度幅值做边缘检测
# ---------------------------------------------------------------------------

def edge_magnitude(gray_arr: np.ndarray) -> np.ndarray:
    """
    计算灰度图的梯度幅值，代替 cv2.Canny。
    gray_arr: float32, shape (H, W)，范围 0-255。
    返回: float32 梯度幅值图，同形状。
    """
    from scipy.ndimage import sobel
    # 分别求 x / y 方向梯度
    gx = sobel(gray_arr, axis=1)
    gy = sobel(gray_arr, axis=0)
    magnitude = np.hypot(gx, gy)
    return magnitude


def find_edge_dense_crop(cell_img: Image.Image,
                          patch_size: int = 150,
                          crop_size: int = 200) -> tuple[int, int]:
    """
    在 cell_img 中找边缘密度最高的 patch_size×patch_size 小块，
    返回以该小块为中心的 crop_size×crop_size 裁剪框的左上角坐标 (x, y)，
    确保裁剪框不超出图像边界。
    """
    W, H = cell_img.size

    # 转灰度 → numpy float32
    gray = np.array(cell_img.convert("L"), dtype=np.float32)

    # 计算边缘幅值
    mag = edge_magnitude(gray)

    # 用滑动求和（积分图）找边缘密度最高的 patch
    # 积分图：pad 边缘以支持边界块
    from scipy.ndimage import uniform_filter
    # uniform_filter(mag, patch_size) 等价于 patch_size×patch_size 均值卷积
    density = uniform_filter(mag, size=patch_size, mode="constant", cval=0.0)

    # 找密度最高点（即最佳 patch 中心）
    # 为了让中心不贴近边缘，在有效区域内搜索
    half_p = patch_size // 2
    half_c = crop_size // 2
    margin = max(half_p, half_c)

    # 限制搜索区域，避免放大框超边界
    search = density[margin: H - margin, margin: W - margin]
    if search.size == 0:
        # 图像太小，退回中心
        cy, cx = H // 2, W // 2
    else:
        flat_idx = np.argmax(search)
        local_y, local_x = np.unravel_index(flat_idx, search.shape)
        cy = local_y + margin
        cx = local_x + margin

    # 计算放大框左上角，并 clamp 至边界
    x0 = int(cx - half_c)
    y0 = int(cy - half_c)
    x0 = max(0, min(x0, W - crop_size))
    y0 = max(0, min(y0, H - crop_size))
    return x0, y0


# ---------------------------------------------------------------------------
# 字体加载（支持 DejaVu 或 PIL 默认）
# ---------------------------------------------------------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    尝试加载 DejaVu Sans；若失败则用 PIL 内置 bitmap 字体（不支持 size，仅备用）。
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    # 最后退路：PIL 内置（忽略 size，文字会很小）
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="生成幻灯片用的边界对比图")
    parser.add_argument("--input", default="outputs/train_run3/viz_it033000.png",
                        help="输入网格图路径")
    parser.add_argument("--output", default="outputs/slides/boundary_comparison.png",
                        help="输出图路径")
    parser.add_argument("--row", type=int, default=1,
                        help="选哪一行（0-based，默认 1）")
    parser.add_argument("--crop-x", type=int, default=None,
                        help="手动指定放大区域左上角 x（若不指定则自动找）")
    parser.add_argument("--crop-y", type=int, default=None,
                        help="手动指定放大区域左上角 y（若不指定则自动找）")
    parser.add_argument("--crop-size", type=int, default=200,
                        help="放大区域边长（默认 200px）")
    parser.add_argument("--zoom", type=float, default=2.5,
                        help="放大倍数（默认 2.5x）")
    parser.add_argument("--mode", choices=["error", "bokeh"], default="error",
                        help="error=对比误差图(col4 vs col5，差异更显眼)；"
                             "bokeh=对比散景图(col0 vs col1)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. 加载网格图，计算各格尺寸
    # -----------------------------------------------------------------------
    grid = Image.open(args.input).convert("RGB")
    grid_w, grid_h = grid.size
    n_cols, n_rows = 6, 4

    cell_w = grid_w // n_cols   # 每列宽
    cell_h = grid_h // n_rows   # 每行高

    row_idx = args.row
    print(f"[INFO] 使用行索引 {row_idx}（第 {row_idx + 1} 行）")

    # -----------------------------------------------------------------------
    # 2. 按模式选列
    # -----------------------------------------------------------------------
    def crop_cell(col: int) -> Image.Image:
        left   = col * cell_w
        upper  = row_idx * cell_h
        right  = left + cell_w
        lower  = upper + cell_h
        return grid.crop((left, upper, right, lower))

    if args.mode == "error":
        # 误差图：col4=|B_phys-GT|×4, col5=|B-GT|×4，差异最显眼
        b_phys   = crop_cell(4)
        b_refine = crop_cell(5)
        label_left  = "Error: physical render"
        label_right = "Error: after network"
        print("[INFO] 模式=error：对比误差图 col4 vs col5")
    else:
        # 散景图：col0=B_phys, col1=B refined，视觉差异细微
        b_phys   = crop_cell(0)
        b_refine = crop_cell(1)
        label_left  = "Physical render (no network)"
        label_right = "After refinement network"
        print("[INFO] 模式=bokeh：对比散景图 col0 vs col1")

    # -----------------------------------------------------------------------
    # 3. 自动（或手动）找放大区域
    # -----------------------------------------------------------------------
    crop_size = args.crop_size

    if args.crop_x is not None and args.crop_y is not None:
        # 手动指定
        cx0, cy0 = args.crop_x, args.crop_y
        # 确保不超边界
        cx0 = max(0, min(cx0, b_phys.width  - crop_size))
        cy0 = max(0, min(cy0, b_phys.height - crop_size))
        print(f"[INFO] 手动指定放大区域：左上角 ({cx0}, {cy0})")
    else:
        # 自动搜索
        cx0, cy0 = find_edge_dense_crop(b_phys, patch_size=150, crop_size=crop_size)
        print(f"[INFO] 自动选定放大区域：左上角 ({cx0}, {cy0})，"
              f"大小 {crop_size}×{crop_size}px")

    # 放大框（右下角）
    cx1 = cx0 + crop_size
    cy1 = cy0 + crop_size

    # -----------------------------------------------------------------------
    # 4. 从两列图分别裁出放大区域，缩放到 zoom 倍
    # -----------------------------------------------------------------------
    zoom_size = int(round(crop_size * args.zoom))  # 500px at 2.5x

    def make_zoom(cell_img: Image.Image) -> Image.Image:
        patch = cell_img.crop((cx0, cy0, cx1, cy1))
        return patch.resize((zoom_size, zoom_size), Image.LANCZOS)

    zoom_phys   = make_zoom(b_phys)
    zoom_refine = make_zoom(b_refine)

    # -----------------------------------------------------------------------
    # 5. 在主图上画红色虚线矩形标注放大区域
    # -----------------------------------------------------------------------
    def draw_dashed_rect(img: Image.Image, box: tuple[int,int,int,int],
                          color=(255, 0, 0), width=2, dash=8) -> Image.Image:
        """在图上画虚线矩形（不修改原图，返回新图）。"""
        out = img.copy()
        draw = ImageDraw.Draw(out)
        x0, y0, x1, y1 = box

        def dashed_line(pts):
            """pts: [(x,y),...] 折线，按 dash 长度交替画/不画"""
            drawn = True
            buf = 0
            for i in range(len(pts) - 1):
                ax, ay = pts[i]
                bx, by = pts[i + 1]
                seg_len = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                if seg_len == 0:
                    continue
                dx_u = (bx - ax) / seg_len
                dy_u = (by - ay) / seg_len
                t = 0.0
                while t < seg_len:
                    t_next = min(t + dash - buf, seg_len)
                    if drawn:
                        px0 = ax + dx_u * t
                        py0 = ay + dy_u * t
                        px1 = ax + dx_u * t_next
                        py1 = ay + dy_u * t_next
                        draw.line([(px0, py0), (px1, py1)], fill=color, width=width)
                    buf = (buf + t_next - t) % dash
                    drawn = (buf == 0) or not drawn if (buf == 0) else drawn
                    if buf == 0:
                        drawn = not drawn
                    t = t_next

        # 四条边（顺时针）
        dashed_line([(x0, y0), (x1, y0)])   # 上
        dashed_line([(x1, y0), (x1, y1)])   # 右
        dashed_line([(x1, y1), (x0, y1)])   # 下
        dashed_line([(x0, y1), (x0, y0)])   # 左
        return out

    # 简洁版：直接调 PIL rectangle（PIL 不原生支持虚线，用短线段模拟）
    def draw_dashed_rect_v2(img: Image.Image, box, color=(255,0,0), width=2, dash=6):
        """
        在图上叠加一个红色虚线矩形，返回新图（原图不变）。
        用短线段列表模拟虚线效果。
        """
        out = img.copy()
        draw = ImageDraw.Draw(out)
        x0, y0, x1, y1 = box
        # 四条边上逐段画实线
        for edge_pts in [
            [(x, y0) for x in range(x0, x1 + 1)],   # 上
            [(x1, y) for y in range(y0, y1 + 1)],   # 右
            [(x, y1) for x in range(x1, x0 - 1, -1)],  # 下
            [(x0, y) for y in range(y1, y0 - 1, -1)],  # 左
        ]:
            on = True
            for i, pt in enumerate(edge_pts):
                if i % (dash * 2) == 0:
                    on = True
                elif i % dash == 0:
                    on = False
                if on:
                    draw.point(pt, fill=color)
        return out

    b_phys_marked   = draw_dashed_rect_v2(b_phys,   (cx0, cy0, cx1, cy1))
    b_refine_marked = draw_dashed_rect_v2(b_refine, (cx0, cy0, cx1, cy1))

    # -----------------------------------------------------------------------
    # 6. 在放大图左上角标注 "zoomed in"
    # -----------------------------------------------------------------------
    font_label = load_font(22)   # 顶部列标签
    font_zoom  = load_font(16)   # 放大图小标注

    def add_zoom_label(img: Image.Image, text: str = "zoomed in") -> Image.Image:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        # 红色背景小框 + 白字
        bbox = draw.textbbox((0, 0), text, font=font_zoom)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        draw.rectangle([4, 4, 4 + tw + pad * 2, 4 + th + pad * 2], fill=(200, 0, 0))
        draw.text((4 + pad, 4 + pad), text, fill=(255, 255, 255), font=font_zoom)
        return out

    zoom_phys   = add_zoom_label(zoom_phys)
    zoom_refine = add_zoom_label(zoom_refine)

    # -----------------------------------------------------------------------
    # 7. 在主图顶部加列标签
    # -----------------------------------------------------------------------
    label_h = 36   # 标签区高度（px）

    def add_top_label(img: Image.Image, text: str) -> Image.Image:
        """
        在图上方加白底黑字标签条，返回高度增加了 label_h 的新图。
        若文字比图像宽，则水平扩展画布以容纳文字（图像居中）。
        """
        W, H = img.size
        # 先量文字宽度
        _dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        tbbox = _dummy.textbbox((0, 0), text, font=font_label)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]
        pad_x = 16   # 文字两侧最小留白
        new_w = max(W, tw + pad_x * 2)

        out = Image.new("RGB", (new_w, H + label_h), color=(255, 255, 255))
        # 原图水平居中粘贴
        img_x = (new_w - W) // 2
        out.paste(img, (img_x, label_h))
        draw = ImageDraw.Draw(out)
        tx = (new_w - tw) // 2
        ty = (label_h - th) // 2
        draw.text((tx, ty), text, fill=(0, 0, 0), font=font_label)
        return out

    b_phys_final   = add_top_label(b_phys_marked,   label_left)
    b_refine_final = add_top_label(b_refine_marked, label_right)

    # -----------------------------------------------------------------------
    # 8. 拼合：主图左右排列 + 下方放大图
    # -----------------------------------------------------------------------
    sep = 8   # 白色分隔线宽度

    main_w = b_phys_final.width + sep + b_refine_final.width
    main_h = max(b_phys_final.height, b_refine_final.height)

    zoom_total_w = zoom_phys.width + sep + zoom_refine.width
    zoom_total_h = max(zoom_phys.height, zoom_refine.height)

    # 整体画布高度 = 主图行 + 间距 + 放大图行
    canvas_h = main_h + sep + zoom_total_h
    canvas_w = max(main_w, zoom_total_w)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    # 主图区：水平居中
    x_off = (canvas_w - main_w) // 2
    canvas.paste(b_phys_final,   (x_off, 0))
    canvas.paste(b_refine_final, (x_off + b_phys_final.width + sep, 0))

    # 放大图区：与主图各自对齐（物理渲染放大图对齐到左列，细化放大图对齐到右列）
    # 放大图宽 zoom_size，主图列宽 cell_w；左对齐列中心
    left_col_x  = x_off
    right_col_x = x_off + b_phys_final.width + sep

    # 把放大图水平居中在各列宽度内
    left_zoom_x  = left_col_x  + (b_phys_final.width   - zoom_phys.width)  // 2
    right_zoom_x = right_col_x + (b_refine_final.width - zoom_refine.width) // 2

    canvas.paste(zoom_phys,   (left_zoom_x,  main_h + sep))
    canvas.paste(zoom_refine, (right_zoom_x, main_h + sep))

    # -----------------------------------------------------------------------
    # 9. 保存
    # -----------------------------------------------------------------------
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    canvas.save(args.output, dpi=(150, 150))

    print(f"[INFO] 输出图尺寸：{canvas.size[0]} × {canvas.size[1]} px")
    print(f"[INFO] 已保存至：{args.output}")
    print()
    print("若自动选区效果不佳（选到背景噪点而非前景边界），可手动传入坐标，例如：")
    print("  python scripts/make_boundary_comparison.py --crop-x 120 --crop-y 80")


if __name__ == "__main__":
    main()
