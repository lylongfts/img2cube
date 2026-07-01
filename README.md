<div align="center">

# img2cube

### Free camera-matching extension for Blender — solve perspective from a photo, then drop 3D snap points

[![Blender](https://img.shields.io/badge/Blender-4.2%20–%205.x-orange)](https://www.blender.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)
[![Free](https://img.shields.io/badge/price-free%20forever-brightgreen)]()

**Made by [Long Live The Cube](https://www.youtube.com/@longlivethecube) · [❤️ Support on Ko-fi](https://ko-fi.com/longlivethecube)**

</div>

---

You have a photo of a building. You want to model it in Blender — but setting up the camera
to match the real perspective is tedious and error-prone.
**img2cube does it automatically — draw a few lines, click Solve, done.**

## What's New in v1.0.6

- 📐 **Depth Graph with T-junction support** — metric depths now propagate through the entire line network, so unmeasured segments get accurate 3D lengths automatically
- 🧲 **Snap to intersections** — origin pick and snap points auto-snap to exact computed line intersections instead of raw click positions
- 📏 **Extension line clicking** — snap points can now be placed on the thin guide lines (infinite extensions), not just drawn segments
- 📊 **Estimated dimensions overlay** — toggle "Show Estimated Dimensions" to see computed lengths on all segments in real time

## Why img2cube

| | img2cube | Manual camera setup | fSpy ($0) | Perspective Plotter ($) |
|---|:---:|:---:|:---:|:---:|
| Works inside Blender (no external app) | ✅ | ✅ | ❌ separate app | ❌ |
| Auto focal length + rotation from lines | ✅ | ❌ manual | ✅ | ✅ |
| **3D snap points on photo lines** | ✅ | ❌ | ❌ | ❌ |
| Real-world scale from one measurement | ✅ | ❌ guess | ❌ | ❌ |
| Depth graph for accurate 3D placement | ✅ | ❌ | ❌ | ❌ |
| Dimension overlay on photo | ✅ | ❌ | ❌ | ❌ |
| No external dependencies | ✅ | — | — | — |
| Free forever | ✅ | ✅ | ✅ | ❌ |

## Features

- **Camera solve from vanishing points** — draw lines along parallel edges in your photo; the solver recovers focal length, rotation, and principal point via SVD
- **2-VP or 3-VP solve** — two vanishing points for a quick solve; three for higher accuracy (no principal-point assumption needed)
- **Precision feedback** — residual reported in pixels: <1 px = excellent, <3 px = good, with guidance on how to improve
- **Real-world scale** — enter one known measurement (e.g. "this door is 2.1 m") and the entire scene scales to match
- **Depth graph propagation** — measured lengths propagate through connected segments via BFS, even across T-junctions
- **3D snap points** — click on any line or extension to place a Blender Empty at the exact 3D position, ready for modeling
- **Intersection snapping** — snap points and origin pick auto-snap to computed line intersections for pixel-perfect placement
- **GPU overlay drawing** — colored line segments, guide extensions, rubber-band preview, endpoint dots, and dimension labels drawn in real time
- **Zero dependencies** — uses only `numpy` + `mathutils`, both shipped with Blender. No bundled wheels, easy to review on extensions.blender.org

## Install

1. Download `img2cube-x.x.x.zip` from either:
   - 🟠 **Gumroad** (recommended): [lylongsoul.gumroad.com/l/qbvoyo](https://lylongsoul.gumroad.com/l/qbvoyo)
   - 🐙 **GitHub Releases**: [github.com/lylongfts/img2cube/releases](https://github.com/lylongfts/img2cube/releases)
2. Drag the zip into Blender, or: Edit → Preferences → Get Extensions → Install from Disk.
3. Done. Open an image in the **Image Editor** → N-panel → **img2cube** tab.

## Usage — 5-Step Workflow

### 1. Image Setup
Load a photo in the Image Editor, click **"Use this image"**.
The extension creates `img2cube_camera`, sets render resolution to match, and attaches the photo as camera background.

### 2. Draw Parallel Lines
Select an axis (**X** = red, **Y** = green, **Z** = blue) and draw line segments along edges that are parallel in real life.
Need at least **2 lines per axis, for at least 2 axes** (4 lines minimum).

### 3. Solve & Set Origin
Click **Solve** — the camera's focal length and rotation are recovered automatically.
Then click **Establish Origin** and pick a real-world corner where two lines meet. That point becomes (0, 0, 0).

### 4. Scale
Pick a line and enter its real-world length. The scene rescales instantly.
Turn on **"Show Estimated Dimensions"** to preview computed lengths on all segments.

### 5. Add Snap Points
Click on any drawn line or extension — a Blender Empty is placed at the exact 3D position.
Points auto-snap to nearby line intersections for precision.

> **Tip:** Fewer accurate lines beat many rough ones. Extra lines are mainly useful as snap-point targets.

## How the Math Works

1. **Vanishing points** — each group of parallel-in-reality segments defines a VP via SVD of homogeneous line equations
2. **Intrinsics** — from 2+ orthogonal VPs, solves for focal length *f* and principal point
3. **Rotation** — VP directions in camera space yield the world-axis directions → full rotation matrix
4. **Scale** — one known real-world length on a segment → recover metric depth via backprojection rays
5. **Depth graph** — BFS propagation of metric depths through connected segment nodes, including T-junctions

Coordinate convention: vision frame (x right, y down, camera looks +Z) — converted to Blender convention in `convert.py`.

## Tutorials

Full workflow demonstrations on YouTube:
**[youtube.com/@longlivethecube](https://www.youtube.com/@longlivethecube)**

## Support the project

img2cube is free and always will be. If it saved you time:
**[❤️ ko-fi.com/longlivethecube](https://ko-fi.com/longlivethecube)**

## License

GPL-3.0-or-later (required for Blender extensions).
