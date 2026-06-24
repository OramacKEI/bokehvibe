# Bokeh Rendering with Optical Aberration Control

Single-image shallow depth-of-field synthesis with physically-grounded, continuously controllable lens aberrations.

## Overview

Given an all-in-focus image, this pipeline renders realistic bokeh with user-adjustable optical aberration styles — spherical aberration (soap bubble / creamy), coma, astigmatism, field curvature, and cat's eye vignetting — without requiring a real lens or depth sensor.

```
All-in-focus image
    │
    ▼
Depth Anything V2 (frozen)  →  disparity map
    │
    ▼
Aberration-aware differentiable renderer  ←  control vector c = {d_f, K, aberration coefficients}
    │   (pupil → FFT → PSF → spatially-varying scatter)
    ▼
Boundary refinement network (~1–2M params, FiLM-conditioned on c)
    │
    ▼
Bokeh image
```

## Aberration styles

| Style | Key parameter |
|---|---|
| Soap bubble (bright ring) | W040 > 0 |
| Creamy soft | W040 < 0 + apodization |
| Swirl / cat's eye | field-dependent vignetting |
| Polygon bokeh | n_blades, blade_curvature |
| Longitudinal CA | per-channel W020 offset |
| Nisen / double-line | W060 |

## Quick start

```bash
# Install dependencies
pip install torch torchvision opencv-python scipy

# Clone Depth Anything V2 and download weights
# (see depth/estimator.py for paths)

# Run demo on a single image
python render/demo.py --input your_image.jpg --style soap_bubble

# End-to-end demo (depth + render + refine)
python render/e2e_demo.py --input your_image.jpg
```

## Structure

```
optics/          # Pupil → PSF generator (physics core)
render/          # Differentiable renderer (no weights)
refine/          # Boundary refinement network (trainable ~1–2M)
depth/           # Depth Anything V2 wrapper
data/            # Synthetic training data pipeline
train/           # Training loop
configs/         # Experiment configs
```

## Requirements

- Python 3.11, PyTorch (CUDA)
- Single consumer GPU (12 GB tested)
- Depth Anything V2 weights: place under `checkpoints/`
