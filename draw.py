# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU overlay: draws drawn segments (colour-coded by axis) and found
intersection points in the Image Editor, plus a live rubber-band while drawing.
Extended guide lines are drawn thin, reaching to the image edges."""

import gpu
from gpu_extras.batch import batch_for_shader

from .properties import AXIS_COLORS

_handle = None
# Module-level live-draw state, set by the modal operator.
preview = {"active": False, "p1": None, "p2": None, "axis": 0}
found_points = []  # list of (px, py) intersection points to highlight


def _pixel_to_region(region, px, py, img_w, img_h):
    """Image pixel (y from top) -> region pixel coords, via the Image Editor view2d."""
    u = px / img_w
    v = 1.0 - (py / img_h)
    return region.view2d.view_to_region(u, v, clip=False)


def _extend_to_edges(x1, y1, x2, y2, img_w, img_h):
    """Extend the infinite line through (x1,y1)-(x2,y2) to the image boundary.
    Returns two points on the image boundary (or None on degenerate input)."""
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None

    # parametric: P(t) = (x1 + t*dx, y1 + t*dy)
    # find t values where P(t) hits x=0, x=img_w, y=0, y=img_h
    ts = []
    if abs(dx) > 1e-9:
        ts.append(-x1 / dx)            # x=0
        ts.append((img_w - x1) / dx)   # x=img_w
    if abs(dy) > 1e-9:
        ts.append(-y1 / dy)            # y=0
        ts.append((img_h - y1) / dy)   # y=img_h

    valid = []
    for t in ts:
        px = x1 + t * dx
        py = y1 + t * dy
        if -1 <= px <= img_w + 1 and -1 <= py <= img_h + 1:
            valid.append((t, px, py))

    if len(valid) < 2:
        return None

    valid.sort(key=lambda v: v[0])
    t_min = valid[0]
    t_max = valid[-1]
    return ((t_min[1], t_min[2]), (t_max[1], t_max[2]))


def _draw():
    import bpy
    ctx = bpy.context
    props = ctx.scene.img2cube
    if not props.image_name:
        return
    region = ctx.region
    iw, ih = props.image_width, props.image_height
    if iw == 0 or ih == 0:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    # --- extended guide lines (thin, faded) --- #
    gpu.state.line_width_set(1.0)
    for seg in props.segments:
        ext = _extend_to_edges(seg.x1, seg.y1, seg.x2, seg.y2, iw, ih)
        if ext is None:
            continue
        col = AXIS_COLORS.get(seg.axis, (1, 1, 1, 1))
        a = _pixel_to_region(region, ext[0][0], ext[0][1], iw, ih)
        b = _pixel_to_region(region, ext[1][0], ext[1][1], iw, ih)
        batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
        shader.uniform_float("color", (col[0], col[1], col[2], 0.25))
        batch.draw(shader)

    # --- committed segments (thick, solid; selected one glows) --- #
    sel_idx = props.selected_segment
    snap_mode = props.is_adding_snaps
    pick_origin_mode = props.is_picking_origin

    # When in snap mode, draw a wide glow under each segment so the user knows
    # they can click on any of them.
    if snap_mode:
        gpu.state.line_width_set(8.0)
        for seg in props.segments:
            col = AXIS_COLORS.get(seg.axis, (1, 1, 1, 1))
            a = _pixel_to_region(region, seg.x1, seg.y1, iw, ih)
            b = _pixel_to_region(region, seg.x2, seg.y2, iw, ih)
            batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
            shader.uniform_float("color", (col[0], col[1], col[2], 0.35))
            batch.draw(shader)

    gpu.state.line_width_set(3.0)
    for i, seg in enumerate(props.segments):
        col = AXIS_COLORS.get(seg.axis, (1, 1, 1, 1))
        a = _pixel_to_region(region, seg.x1, seg.y1, iw, ih)
        b = _pixel_to_region(region, seg.x2, seg.y2, iw, ih)
        batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
        shader.uniform_float("color", col)
        batch.draw(shader)
    # selected segment overlay (bright white outline)
    if 0 <= sel_idx < len(props.segments):
        seg = props.segments[sel_idx]
        gpu.state.line_width_set(6.0)
        a = _pixel_to_region(region, seg.x1, seg.y1, iw, ih)
        b = _pixel_to_region(region, seg.x2, seg.y2, iw, ih)
        batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
        batch.draw(shader)
        gpu.state.line_width_set(3.0)
        batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
        shader.uniform_float("color", (1.0, 0.95, 0.2, 1.0))
        batch.draw(shader)

    # --- segment endpoints (small dots) --- #
    gpu.state.point_size_set(5.0)
    for seg in props.segments:
        col = AXIS_COLORS.get(seg.axis, (1, 1, 1, 1))
        pts = [
            _pixel_to_region(region, seg.x1, seg.y1, iw, ih),
            _pixel_to_region(region, seg.x2, seg.y2, iw, ih),
        ]
        batch = batch_for_shader(shader, 'POINTS', {"pos": pts})
        shader.uniform_float("color", col)
        batch.draw(shader)

    # --- live rubber-band --- #
    if preview["active"] and preview["p1"] and preview["p2"]:
        col = AXIS_COLORS.get(preview["axis"], (1, 1, 1, 1))
        # extended guide for the live line
        ext = _extend_to_edges(*preview["p1"], *preview["p2"], iw, ih)
        if ext:
            gpu.state.line_width_set(1.0)
            a = _pixel_to_region(region, ext[0][0], ext[0][1], iw, ih)
            b = _pixel_to_region(region, ext[1][0], ext[1][1], iw, ih)
            batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
            shader.uniform_float("color", (col[0], col[1], col[2], 0.2))
            batch.draw(shader)
        # the live segment itself
        gpu.state.line_width_set(3.0)
        a = _pixel_to_region(region, *preview["p1"], iw, ih)
        b = _pixel_to_region(region, *preview["p2"], iw, ih)
        batch = batch_for_shader(shader, 'LINES', {"pos": [a, b]})
        shader.uniform_float("color", (col[0], col[1], col[2], 0.6))
        batch.draw(shader)

    # --- found intersection points --- #
    if found_points:
        pts = [_pixel_to_region(region, px, py, iw, ih) for (px, py) in found_points]
        gpu.state.point_size_set(8.0)
        batch = batch_for_shader(shader, 'POINTS', {"pos": pts})
        shader.uniform_float("color", (0.1, 0.9, 1.0, 1.0)) # Cyan
        batch.draw(shader)

    # --- origin marker --- #
    if props.origin_set:
        ox = _pixel_to_region(region, props.origin_pixel_x, props.origin_pixel_y, iw, ih)
        # crosshair (two short lines forming an X)
        gpu.state.line_width_set(2.0)
        s = 12
        cross = [
            (ox[0] - s, ox[1] - s), (ox[0] + s, ox[1] + s),
            (ox[0] - s, ox[1] + s), (ox[0] + s, ox[1] - s),
        ]
        batch = batch_for_shader(shader, 'LINES', {"pos": cross})
        shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
        batch.draw(shader)
        # central dot
        gpu.state.point_size_set(10.0)
        batch = batch_for_shader(shader, 'POINTS', {"pos": [ox]})
        shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
        batch.draw(shader)

    # --- dimensions overlay --- #
    if getattr(props, "show_dimensions", False) and props.solved and getattr(props, "n_measured", 0) > 0:
        import blf
        from . import operators

        lengths = operators.compute_segment_lengths(props)

        font_id = 0
        blf.size(font_id, 17)

        for i, seg in enumerate(props.segments):
            mx = (seg.x1 + seg.x2) / 2.0
            my = (seg.y1 + seg.y2) / 2.0
            rp = _pixel_to_region(region, mx, my, iw, ih)

            info = lengths[i] if i < len(lengths) else {"value": None,
                                                        "status": "unknown"}
            if info["status"] == "measured":
                text = f"{info['value']:.3f} m"
                rgb = (0.35, 1.0, 0.45)          # green = measured
            elif info["status"] == "estimated":
                text = f"~{info['value']:.3f} m"
                rgb = (1.0, 0.82, 0.25)          # amber = estimated
            else:
                text = "? (draw to a corner)"
                rgb = (0.85, 0.85, 0.90)         # grey = undetermined

            _draw_text_outlined(font_id, rp[0] + 8, rp[1] + 8, text, rgb)

    gpu.state.blend_set('NONE')


def _draw_text_outlined(font_id, x, y, text, rgb, px=2):
    """Draw text with a solid black outline so it stays readable over any
    background (grey/amber on a bright photo is otherwise hard to see)."""
    import blf
    blf.color(font_id, 0.0, 0.0, 0.0, 1.0)
    for dx in (-px, 0, px):
        for dy in (-px, 0, px):
            if dx == 0 and dy == 0:
                continue
            blf.position(font_id, x + dx, y + dy, 0)
            blf.draw(font_id, text)
    blf.color(font_id, rgb[0], rgb[1], rgb[2], 1.0)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)


def enable():
    global _handle
    if _handle is None:
        import bpy
        _handle = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_PIXEL')


def disable():
    global _handle
    if _handle is not None:
        import bpy
        bpy.types.SpaceImageEditor.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
