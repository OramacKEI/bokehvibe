"""
scripts/make_pipeline_strip.py
==============================
Generates a horizontal pipeline strip for slide figures.

4-panel layout (using pre-existing outputs):
    Input  →  Depth (DA V2, frozen)  →  Aberration Renderer  →  Refinement Net

Sources (from outputs/):
    render_test/demo_input.png              -- all-focus input photo
    render_test/demo_disparity_snapped.png  -- edge-snapped disparity from DA V2
    render_test/demo_soap_bubble.png        -- physical bokeh (soap-bubble preset)
    e2e_test/e2e_soap_bubble_refined.png    -- after boundary refinement net

If the refinement-net panel is missing (e.g., no e2e outputs yet), falls back to
the 3-panel version and prints a note.

Output:
    outputs/slides/pipeline_strip.png   (dpi=200, white background)

Usage:
    python scripts/make_pipeline_strip.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "slides"

# Panel definitions: (relative path under outputs/, display label)
PANEL_DEFS = [
    ("render_test/demo_input.png",              "Input"),
    ("render_test/demo_disparity_snapped.png",  "Depth\n(DA V2, frozen)"),
    ("render_test/demo_soap_bubble.png",        "Aberration Renderer\n(soap-bubble style)"),
    ("e2e_test/e2e_soap_bubble_refined.png",    "Refinement Net\n(boundary fix)"),
]

TARGET_HEIGHT = 480   # px -- all panels resized to this height before layout


def load_panel(rel_path: str) -> np.ndarray:
    """Load image, convert to RGB, resize to TARGET_HEIGHT, return float [0,1]."""
    p = PROJECT_ROOT / "outputs" / rel_path
    if not p.exists():
        raise FileNotFoundError(p)
    img = Image.open(p).convert("RGB")
    w, h = img.size
    new_w = int(round(w * TARGET_HEIGHT / h))
    img = img.resize((new_w, TARGET_HEIGHT), Image.LANCZOS)
    return np.asarray(img) / 255.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load panels; skip missing ones with a warning
    panels: list[tuple[np.ndarray, str]] = []
    for rel, label in PANEL_DEFS:
        try:
            arr = load_panel(rel)
            panels.append((arr, label))
            print(f"[strip] loaded  {rel}  shape={arr.shape}")
        except FileNotFoundError as exc:
            print(f"[strip] SKIP (not found): {exc}")

    if not panels:
        raise RuntimeError("No panels found – run render.demo and render.e2e_demo first.")

    n = len(panels)
    if n < 4:
        print(f"[strip] NOTE: only {n}/4 panels available; "
              "run render/e2e_demo.py to add the Refinement Net panel.")

    # Figure width proportional to panel pixel widths; height fixed
    panel_w_px = [arr.shape[1] for arr, _ in panels]
    arrow_w_in = 0.35          # inches per inter-panel arrow gap
    panel_h_in = 3.2           # panel display height in inches (DPI-independent)
    dpi = 200
    px_per_in = TARGET_HEIGHT / panel_h_in   # pixels-per-inch at which panels are shown
    panel_w_in = [w / px_per_in for w in panel_w_px]

    fig_w = sum(panel_w_in) + (n - 1) * arrow_w_in + 0.3
    fig_h = panel_h_in + 0.85   # +space for title and labels

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Top padding for title, bottom for labels
    top_pad   = 0.14
    bot_pad   = 0.18
    usable_h  = 1.0 - top_pad - bot_pad
    total_w   = sum(panel_w_in) + (n - 1) * arrow_w_in
    left_margin = 0.02

    x = left_margin / fig_w  # start x in figure-fraction
    for i, ((arr, label), pw_in) in enumerate(zip(panels, panel_w_in)):
        w_frac = pw_in / fig_w
        ax = fig.add_axes([x, bot_pad, w_frac, usable_h])
        ax.imshow(arr, aspect="auto")
        ax.set_title(label, fontsize=9.5, fontweight="bold", pad=4,
                     linespacing=1.35)
        ax.axis("off")
        x += w_frac

        # Arrow between panels
        if i < n - 1:
            arr_x = x + (arrow_w_in / 2) / fig_w
            arr_y = bot_pad + usable_h / 2
            fig.text(arr_x, arr_y, "→", fontsize=20, ha="center", va="center",
                     color="#444444", fontweight="bold")
            x += arrow_w_in / fig_w

    fig.suptitle(
        "Bokeh Rendering Pipeline: All-focus Image  →  DA V2 Depth  →  "
        "Physical Aberration Renderer  →  Refinement Net",
        fontsize=10.5, y=0.99, ha="center",
    )

    out = OUT_DIR / "pipeline_strip.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    result = Image.open(out)
    print(f"[strip] saved  {out}  size={result.size}  dpi={dpi}")


if __name__ == "__main__":
    main()
