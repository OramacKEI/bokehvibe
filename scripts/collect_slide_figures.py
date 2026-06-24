"""
scripts/collect_slide_figures.py
=================================
Collect publication-quality figures for group-meeting slides into outputs/slides/.

For each source image:
  - Composites any transparency onto a white background (RGBA → RGB)
  - Re-saves at dpi=200 (changes metadata; pixel dimensions are preserved)
  - Renames to a descriptive slide-friendly filename

Then generates outputs/slides/README.md listing each figure and the
recommended slide it belongs to.

Source files expected (relative to PROJECT_ROOT/outputs/):
    psf_test/presets.png
    psf_test/w040_sweep.png
    render_test/demo_grid.png
    render_test/demo_disparity_snapped.png
    slides/pipeline_strip.png      (built by scripts/make_pipeline_strip.py)
    train_run2/loss_curve.png
    train_run2/viz_it039000.png
    refine_test/conditioning.png
    synth_test/samples.png

Usage:
    # First build the pipeline strip, then collect:
    python scripts/make_pipeline_strip.py
    python scripts/collect_slide_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
SLIDES_DIR = OUTPUTS / "slides"
DPI = 200

# (source relative to OUTPUTS, dest filename, slide suggestion, notes)
FIGURE_LIST = [
    (
        "psf_test/presets.png",
        "fig01_psf_presets.png",
        "Slide: Aberration-Controlled PSF (PSF物理验证)",
        "Shows all lens-style presets in background/foreground rows. "
        "Use for the 'PSF physics' slide.",
    ),
    (
        "psf_test/w040_sweep.png",
        "fig02_psf_w040_sweep.png",
        "Slide: Spherical Aberration Sweep (W040 vs defocus sign)",
        "W040 × defocus sign matrix; bright-edge ring appears when "
        "sign(W040) ≠ sign(W020). Use for the 'aberration control' slide.",
    ),
    (
        "render_test/demo_grid.png",
        "fig03_render_demo_grid.png",
        "Slide: Aberration Render Results (full-photo results grid)",
        "All preset styles side-by-side on a real photo. "
        "Use for the 'first-light results' or 'style comparison' slide.",
    ),
    (
        "render_test/demo_disparity_snapped.png",
        "fig04_render_disparity.png",
        "Slide: Depth Estimation (DA V2 disparity, edge-snapped)",
        "Pseudo-colour disparity from DA V2 after edge snapping. "
        "Use for the 'depth module' or pipeline slide.",
    ),
    (
        "slides/pipeline_strip.png",
        "fig05_pipeline_strip.png",
        "Slide: System Overview (pipeline strip)",
        "4-panel horizontal strip: Input → Depth → Physics Render → "
        "Refinement Net. Use as the main pipeline diagram slide.",
    ),
    (
        "train_run2/loss_curve.png",
        "fig06_loss_curve.png",
        "Slide: Training Progress (loss curve, train_run2)",
        "L1 and LPIPS curves over 50k iterations from train_run2. "
        "Use for the 'training' or 'experiments' slide.",
    ),
    (
        "train_run2/viz_it039000.png",
        "fig07_viz_it039000.png",
        "Slide: Training Visualisation @ 39k steps (train_run2)",
        "Side-by-side: input / disparity / physical bokeh / refined / GT / "
        "error. Use for the 'qualitative results' slide. "
        "CHECK: verify no Chinese labels in this image.",
    ),
    (
        "refine_test/conditioning.png",
        "fig08_conditioning.png",
        "Slide: Conditioning Maps (spatial + PSF descriptor channels)",
        "17-channel condition maps fed into the refinement net (FiLM). "
        "Use for the 'network design' or 'conditioning' slide.",
    ),
    (
        "synth_test/samples.png",
        "fig09_synth_samples.png",
        "Slide: Synthetic Training Data (online synthesis examples)",
        "Randomly synthesised (input, disparity, bokeh GT, mask) tuples. "
        "Use for the 'data pipeline' slide.",
    ),
]


def composite_on_white(img: Image.Image) -> Image.Image:
    """Flatten RGBA/LA onto a white background; return RGB image."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        # Use the alpha channel as mask
        if img.mode == "RGBA":
            bg.paste(img, mask=img.split()[3])
        else:
            bg.paste(img.convert("RGB"), mask=img.split()[1])
        return bg
    return img.convert("RGB")


def collect() -> list[dict]:
    """Copy/re-save figures into SLIDES_DIR at dpi=200. Returns report rows."""
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    for src_rel, dst_name, slide_hint, notes in FIGURE_LIST:
        src = OUTPUTS / src_rel
        dst = SLIDES_DIR / dst_name

        row: dict = {
            "source": src_rel,
            "dest": str(dst.relative_to(PROJECT_ROOT)),
            "slide_hint": slide_hint,
            "status": "MISSING",
            "size": "—",
            "dpi": "—",
            "notes": notes,
        }

        if not src.exists():
            print(f"[collect] MISSING  {src_rel}")
            report.append(row)
            continue

        try:
            img = Image.open(src)
            img_rgb = composite_on_white(img)
            img_rgb.save(str(dst), dpi=(DPI, DPI))
            # Verify
            saved = Image.open(dst)
            row["status"] = "OK"
            row["size"] = f"{saved.size[0]}×{saved.size[1]}"
            row["dpi"] = str(saved.info.get("dpi", DPI))
            print(f"[collect] OK       {dst_name}  {row['size']}")
        except Exception as exc:
            row["status"] = f"ERROR: {exc}"
            print(f"[collect] ERROR    {src_rel}: {exc}")

        report.append(row)

    return report


def write_readme(report: list[dict]) -> None:
    """Write outputs/slides/README.md listing figures and slide suggestions."""
    lines = [
        "# Slide Figures — outputs/slides/",
        "",
        "All images: white background, dpi=200, English labels.",
        "Generated by `scripts/collect_slide_figures.py`.",
        "",
        "---",
        "",
        "| File | Size (px) | Status | Recommended slide |",
        "|------|-----------|--------|-------------------|",
    ]
    for row in report:
        dst_short = Path(row["dest"]).name
        lines.append(
            f"| `{dst_short}` "
            f"| {row['size']} "
            f"| {row['status']} "
            f"| {row['slide_hint']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Figures requiring manual inspection",
        "",
        "- **fig07_viz_it039000.png** — training visualisation snapshot from train_run2.",
        "  Verify no Chinese labels appear in the panel titles before inserting into slides.",
        "  (All known label strings in `train/train.py` are English, but worth a visual check.)",
        "",
        "- **fig06_loss_curve.png** — check y-axis label and title are legible at slide scale.",
        "",
        "## Notes on pipeline_strip",
        "",
        "- `fig05_pipeline_strip.png` is built by `scripts/make_pipeline_strip.py`.",
        "  If the Refinement Net panel is missing, re-run `render/e2e_demo.py` first.",
        "",
        "## Input image used",
        "",
        "All render outputs were produced from `assets/slide_demo.jpg`",
        "(= `third_party/BokehMe/inputs/21.jpg`, the BokehMe canonical demo photo).",
        "",
    ]

    readme = SLIDES_DIR / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"[collect] README written: {readme}")


def main() -> None:
    print(f"[collect] Collecting slide figures into {SLIDES_DIR} ...")
    report = collect()

    ok = sum(1 for r in report if r["status"] == "OK")
    missing = sum(1 for r in report if r["status"] == "MISSING")
    error = sum(1 for r in report if r["status"] not in ("OK", "MISSING"))
    print(f"[collect] Done: {ok} OK, {missing} missing, {error} errors")

    write_readme(report)

    print()
    print("=== Summary table ===")
    print(f"{'File':<42} {'Size':>12}  {'Status'}")
    print("-" * 62)
    for row in report:
        dst_short = Path(row["dest"]).name
        print(f"{dst_short:<42} {row['size']:>12}  {row['status']}")


if __name__ == "__main__":
    main()
