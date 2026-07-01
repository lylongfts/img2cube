# SPDX-License-Identifier: GPL-3.0-or-later
"""img2cube operators (v1.0.6) — direct geometry per the 5-step workflow.

Key change from v1.0.5: shared depth-graph infrastructure with T-junction
support.  Every function that needs 3D positions now queries the graph, so
lines connected (by corner or T-junction) to a measured edge are placed at
their correct metric depth.  Isolated lines fall back to scale_factor.
"""

import numpy as np
import bpy
import mathutils
from mathutils.geometry import intersect_line_line
from bpy.types import Operator
from bpy.props import IntProperty, FloatProperty

from . import solver, convert, draw
from .properties import AXIS_ITEMS

_AXIS_LABEL = {int(it[0]): it[1] for it in AXIS_ITEMS}


# =========================================================================== #
# Shared helpers
# =========================================================================== #
def _props(context):
    return context.scene.img2cube


def _seg_tuple(s):
    return ((s.x1, s.y1), (s.x2, s.y2))


def _point_to_segment_distance(p, a, b):
    """2D distance from point p to SEGMENT a-b.  Returns (distance, t clamped [0,1])."""
    ap = p - a
    ab = b - a
    ab_len_sq = float(ab @ ab)
    if ab_len_sq < 1e-9:
        return float(np.linalg.norm(ap)), 0.0
    t = float(ap @ ab) / ab_len_sq
    t_clamped = max(0.0, min(1.0, t))
    closest = a + t_clamped * ab
    return float(np.linalg.norm(p - closest)), t_clamped


def _point_to_line_distance(p, a, b):
    """2D distance from point p to the INFINITE line through a-b.
    Returns (distance, unclamped_t)."""
    ap = p - a
    ab = b - a
    ab_len_sq = float(ab @ ab)
    if ab_len_sq < 1e-9:
        return float(np.linalg.norm(ap)), 0.0
    t = float(ap @ ab) / ab_len_sq
    closest = a + t * ab
    return float(np.linalg.norm(p - closest)), t


def _find_segment_near_pixel(props, px, py, tol_px=15.0, axis_filter=None,
                             allow_extension=False):
    """Return (best_index, distance, t) for the segment closest to the pixel.

    When allow_extension=True the infinite line through each segment is tested
    (not just the drawn portion), so clicks on the thin guide lines register.
    Returns (-1, inf, 0) if nothing within tol_px.
    """
    p = np.array([px, py])
    best = (-1, float('inf'), 0.0)
    dist_fn = _point_to_line_distance if allow_extension else _point_to_segment_distance
    for i, s in enumerate(props.segments):
        if axis_filter is not None:
            if isinstance(axis_filter, int):
                if s.axis != axis_filter:
                    continue
            elif s.axis not in axis_filter:
                continue
        a = np.array([s.x1, s.y1])
        b = np.array([s.x2, s.y2])
        d, t = dist_fn(p, a, b)
        if d < best[1]:
            best = (i, d, t)
    if best[1] > tol_px:
        return (-1, best[1], best[2])
    return best


def _gather_vps(props):
    """Compute finite VPs and per-axis residuals from drawn segments."""
    vps, axes, residuals = [], [], []
    for axis in (0, 1, 2):
        segs = [_seg_tuple(s) for s in props.segments if s.axis == axis]
        if len(segs) < 2:
            continue
        v = solver.vanishing_point(segs)
        if not solver.vp_is_finite(v):
            continue
        vp_xy = solver.vp_to_xy(v)
        vps.append(vp_xy)
        axes.append(axis)
        residuals.append(solver.vp_residual_px(segs, vp_xy))
    return vps, axes, residuals


def _solve_rotation_from_props(props):
    """Recompute rotation R and axes-used list from current solve state."""
    vps, axes_used, _ = _gather_vps(props)
    pp = (props.solved_pp_x, props.solved_pp_y)
    f = props.solved_focal_px
    R = solver.solve_rotation(vps, pp, f)
    return R, axes_used


# =========================================================================== #
# Depth graph — shared infrastructure for 3D placement and length estimation
# =========================================================================== #
import math
from collections import defaultdict, deque


def _depth_ratio_for_pair(ray_a, ray_b, u):
    """Depth ratio depth_a / depth_b given that the 3D vector from b to a is
    parallel to unit direction u.

    From:  depth_a * ray_a - depth_b * ray_b = lambda * u
    Cross both sides with u:
        depth_a * (ray_a x u) = depth_b * (ray_b x u)
    Project onto (ray_a x u):
        depth_a / depth_b = (ray_b x u) . (ray_a x u) / |ray_a x u|^2
    """
    a0 = np.cross(ray_a, u)
    a1 = np.cross(ray_b, u)
    den = float(a0 @ a0)
    if den < 1e-12:
        return None
    ratio = float((a1 @ a0) / den)
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    return ratio


def build_node_graph(props):
    """Build a connectivity graph of segment nodes, including T-junctions.

    Returns a dict with keys:
        nodes       — list of (px, py)
        node_of     — dict (seg_idx, which) -> node_id   (which: 0=start, 1=end)
        node_ray    — list of unit rays per node
        adj         — defaultdict(list) of (neighbour, depth_ratio)
        seg_nodes   — dict seg_idx -> (n0, n1, ratio_n0_over_n1)
        R, axes_used, pp, f — solver results cached
    """
    if not props.solved:
        return None

    R, axes_used = _solve_rotation_from_props(props)
    pp = (props.solved_pp_x, props.solved_pp_y)
    f = props.solved_focal_px
    tol = max(6.0, 0.004 * math.hypot(props.image_width, props.image_height))

    # --- Build nodes from endpoints ---
    nodes = []
    node_of = {}

    def _get_or_create_node(px, py):
        for ni, q in enumerate(nodes):
            if math.hypot(px - q[0], py - q[1]) <= tol:
                return ni
        nodes.append((px, py))
        return len(nodes) - 1

    for i, s in enumerate(props.segments):
        node_of[(i, 0)] = _get_or_create_node(s.x1, s.y1)
        node_of[(i, 1)] = _get_or_create_node(s.x2, s.y2)

    node_ray = [solver.ray_through_pixel(p[0], p[1], pp, f) for p in nodes]

    # --- Endpoint edges: each segment links its two endpoint nodes ---
    adj = defaultdict(list)
    seg_nodes = {}
    for i, s in enumerate(props.segments):
        if s.axis not in axes_used:
            continue
        n0 = node_of[(i, 0)]
        n1 = node_of[(i, 1)]
        if n0 == n1:
            continue
        u = R[:, axes_used.index(s.axis)]
        u = u / np.linalg.norm(u)
        ratio = _depth_ratio_for_pair(node_ray[n0], node_ray[n1], u)
        if ratio is None:
            continue
        seg_nodes[i] = (n0, n1, ratio)
        adj[n0].append((n1, ratio))
        adj[n1].append((n0, 1.0 / ratio))

    # --- T-junction edges ---
    # An endpoint of seg A is a T-junction on seg B if it lies close to the
    # LINE of B (but not at B's endpoints, which are already merged above).
    for i, si in enumerate(props.segments):
        for w in (0, 1):
            ni = node_of[(i, w)]
            pi = np.array(nodes[ni])
            for j, sj in enumerate(props.segments):
                if j == i or sj.axis not in axes_used:
                    continue
                nj0 = node_of[(j, 0)]
                nj1 = node_of[(j, 1)]
                # skip if already same node as an endpoint of j (corner, not T)
                if ni == nj0 or ni == nj1:
                    continue
                aj = np.array([sj.x1, sj.y1])
                bj = np.array([sj.x2, sj.y2])
                d, t = _point_to_line_distance(pi, aj, bj)
                # only accept T-junctions on the drawn portion (0..1) with
                # small margin, not at the far extensions
                if d > tol or t < -0.05 or t > 1.05:
                    continue
                # T-junction found: node ni sits on line j.
                # depth[ni] * ray[ni] - depth[nj0] * ray[nj0] is parallel
                # to u_j (seg j's axis direction).
                uj = R[:, axes_used.index(sj.axis)]
                uj = uj / np.linalg.norm(uj)
                ratio_i_over_j0 = _depth_ratio_for_pair(node_ray[ni],
                                                        node_ray[nj0], uj)
                if ratio_i_over_j0 is not None:
                    # avoid duplicate edges
                    if not any(nb == nj0 for nb, _ in adj[ni]):
                        adj[ni].append((nj0, ratio_i_over_j0))
                        adj[nj0].append((ni, 1.0 / ratio_i_over_j0))

    return {
        "nodes": nodes, "node_of": node_of, "node_ray": node_ray,
        "adj": adj, "seg_nodes": seg_nodes,
        "R": R, "axes_used": axes_used, "pp": pp, "f": f,
    }


def compute_metric_depths(props, graph=None):
    """BFS from every measured segment to compute metric node depths.

    Returns dict  node_id -> depth  (in world units, consistent with the
    measured segments).  Nodes not reachable from any measured segment are
    absent from the dict.
    """
    if graph is None:
        graph = build_node_graph(props)
    if graph is None:
        return {}

    seg_nodes = graph["seg_nodes"]
    node_ray = graph["node_ray"]
    adj = graph["adj"]
    final_depth = {}       # node_id -> list of metric depths from each source
    depth_lists = defaultdict(list)

    for i, s in enumerate(props.segments):
        if s.known_length <= 0 or i not in seg_nodes:
            continue
        n0, n1, ratio = seg_nodes[i]
        # BFS relative depths from this measured segment
        rel = {n1: 1.0, n0: ratio}
        dq = deque([n0, n1])
        while dq:
            u = dq.popleft()
            for v, r_u_over_v in adj[u]:
                if v not in rel:
                    rel[v] = rel[u] / r_u_over_v
                    dq.append(v)
        # Scale factor: make seg i have its known_length
        P0 = rel[n0] * node_ray[n0]
        P1 = rel[n1] * node_ray[n1]
        cur = float(np.linalg.norm(P1 - P0))
        if cur < 1e-9:
            continue
        k = float(s.known_length) / cur
        for nid, rd in rel.items():
            depth_lists[nid].append(rd * k)

    # Average depths from multiple sources for robustness
    return {nid: float(np.mean(vals)) for nid, vals in depth_lists.items()}


def compute_placement_depths(props, graph=None):
    """Depths for PLACEMENT (snap points + camera), never for reported lengths.

    Extends compute_metric_depths: clusters that contain no measured edge are
    still positioned, by pinning each such cluster (as a rigid unit whose SHAPE
    is correct — only its one unknowable global scale is guessed) to the depth
    of the nearest already-placed node in image space.  This stops isolated
    lines from being dropped at a meaningless scale_factor depth and flying off.
    The pinned depths are not metric-trustworthy, which is why length reporting
    keeps using compute_metric_depths (anchored-only) instead.
    """
    if graph is None:
        graph = build_node_graph(props)
    if graph is None:
        return {}
    adj = graph["adj"]
    nodes = graph["nodes"]
    N = len(nodes)

    depth = dict(compute_metric_depths(props, graph))  # anchored (metric) nodes

    # Discover connected components with local relative depths.
    seen = set()
    comps = []
    for start in range(N):
        if start in seen:
            continue
        rel = {start: 1.0}
        dq = deque([start])
        seen.add(start)
        comp = [start]
        while dq:
            u = dq.popleft()
            for v, r_u_over_v in adj[u]:
                if v not in rel:
                    rel[v] = rel[u] / r_u_over_v
                    seen.add(v)
                    dq.append(v)
                    comp.append(v)
        comps.append((comp, rel))

    unplaced = [(c, r) for (c, r) in comps if not any(v in depth for v in c)]

    # Nothing measured at all: seed the largest cluster at a nominal depth so
    # the whole scene is finite and sensibly sized.
    if not depth and unplaced:
        comp, rel = max(unplaced, key=lambda cr: len(cr[0]))
        med = float(np.median([rel[w] for w in comp])) or 1.0
        for v in comp:
            depth[v] = rel[v] * (10.0 / med)
        unplaced = [(c, r) for (c, r) in comps if not any(v in depth for v in c)]

    # Iteratively pin each remaining cluster to the nearest already-placed node.
    changed = True
    while unplaced and changed:
        changed = False
        best = None  # (img_dist, cluster_index, node_in_cluster, placed_node)
        for ci, (comp, rel) in enumerate(unplaced):
            for u in comp:
                pu = nodes[u]
                for a in depth:
                    pa = nodes[a]
                    dd = math.hypot(pu[0] - pa[0], pu[1] - pa[1])
                    if best is None or dd < best[0]:
                        best = (dd, ci, u, a)
        if best is not None:
            _, ci, u_pin, a_pin = best
            comp, rel = unplaced[ci]
            sc = depth[a_pin] / rel[u_pin]
            for v in comp:
                depth[v] = rel[v] * sc
            unplaced.pop(ci)
            changed = True
    return depth


def compute_segment_lengths(props):
    """Return one dict per segment: {'value': float|None, 'status': str}.

    status is 'measured', 'estimated', or 'unknown'.
    """
    n = len(props.segments)
    result = [{'value': None, 'status': 'unknown'} for _ in range(n)]
    if n == 0 or not props.solved:
        return result

    for i, s in enumerate(props.segments):
        if s.known_length > 0:
            result[i] = {'value': float(s.known_length), 'status': 'measured'}

    graph = build_node_graph(props)
    if graph is None:
        return result

    depths = compute_metric_depths(props, graph)
    if not depths:
        return result

    node_ray = graph["node_ray"]
    seg_nodes = graph["seg_nodes"]
    for j in range(n):
        if result[j]['status'] == 'measured':
            continue
        if j not in seg_nodes:
            continue
        m0, m1, _ = seg_nodes[j]
        if m0 in depths and m1 in depths:
            Q0 = depths[m0] * node_ray[m0]
            Q1 = depths[m1] * node_ray[m1]
            result[j] = {'value': float(np.linalg.norm(Q1 - Q0)),
                         'status': 'estimated'}
    return result


# =========================================================================== #
# 3D placement helpers (now depth-graph aware)
# =========================================================================== #
def _segment_axis_dir_cam(props, seg, R, axes_used):
    """Get the 3D direction of `seg` in cam frame (vision)."""
    if seg.axis not in axes_used:
        return None
    return R[:, axes_used.index(seg.axis)]


def _segment_3d_in_cam(props, seg, R, axes_used, scale,
                        seg_idx=None, metric_depths=None, graph=None):
    """Compute the segment's 3D endpoints in cam (vision) frame.

    If metric_depths (from compute_metric_depths) is provided AND both of this
    segment's nodes have known depths, use them for exact placement.  Otherwise
    fall back to the old normalised-depth * scale_factor approach.
    """
    d = _segment_axis_dir_cam(props, seg, R, axes_used)
    if d is None:
        return None
    pp = (props.solved_pp_x, props.solved_pp_y)
    f = props.solved_focal_px

    # Try metric depths first
    if metric_depths and graph and seg_idx is not None:
        node_of = graph["node_of"]
        node_ray = graph["node_ray"]
        n0 = node_of.get((seg_idx, 0))
        n1 = node_of.get((seg_idx, 1))
        if n0 is not None and n1 is not None:
            d0 = metric_depths.get(n0)
            d1 = metric_depths.get(n1)
            if d0 is not None and d1 is not None:
                return d0 * node_ray[n0], d1 * node_ray[n1]

    # Fallback: normalised depths * scale
    sol = solver.solve_segment_depths(_seg_tuple(seg), d, pp, f)
    if sol is None:
        return None
    L = float(seg.known_length) if seg.known_length > 0 else float(scale)
    return sol["P1"] * L, sol["P2"] * L


def _cam_vision_to_blender_world(p_cam_vision, cam_obj):
    """Convert a vision-frame cam-local point to Blender world coordinates."""
    local = mathutils.Vector((p_cam_vision[0],
                              -p_cam_vision[1],
                              -p_cam_vision[2]))
    return cam_obj.matrix_world @ local


def _ray_through_pixel_blender_world(props, cam_obj, px, py):
    """Return (origin, direction) of the camera ray through pixel (px, py) in
    Blender world coordinates."""
    pp = (props.solved_pp_x, props.solved_pp_y)
    f = props.solved_focal_px
    r_v = solver.ray_through_pixel(px, py, pp, f)
    d_local = mathutils.Vector((r_v[0], -r_v[1], -r_v[2]))
    M = cam_obj.matrix_world.to_3x3()
    d_world = M @ d_local
    d_world.normalize()
    origin = cam_obj.matrix_world.translation.copy()
    return origin, d_world


def _intersect_line_line_safe(a1, a2, b1, b2):
    """mathutils intersect_line_line that returns midpoint, or None."""
    res = intersect_line_line(a1, a2, b1, b2)
    if res is None:
        return None
    pa, pb = res
    return (pa + pb) / 2.0


def _closest_point_on_second_line(a1, a2, b1, b2):
    """Closest point that lies ON the infinite line b1-b2 to the line a1-a2."""
    res = intersect_line_line(a1, a2, b1, b2)
    if res is None:
        return None
    _pa, pb = res
    return pb


def _find_nearest_intersection_pixel(props, click_px, click_py, tol):
    """Find the nearest 2D intersection of two differently-axed lines to the
    click position.  Returns (inter_px, inter_py, seg_a_idx, seg_b_idx) or
    None if nothing is within tol."""
    click = np.array([click_px, click_py])
    best = None  # (dist, inter, idx_a, idx_b)
    n = len(props.segments)
    for i in range(n):
        si = props.segments[i]
        for j in range(i + 1, n):
            sj = props.segments[j]
            if si.axis == sj.axis:
                continue
            inter = solver.segment_intersection(
                ((si.x1, si.y1), (si.x2, si.y2)),
                ((sj.x1, sj.y1), (sj.x2, sj.y2)))
            if inter is None:
                continue
            d = float(np.linalg.norm(inter - click))
            if d <= tol and (best is None or d < best[0]):
                best = (d, inter, i, j)
    if best is None:
        return None
    return (float(best[1][0]), float(best[1][1]), best[2], best[3])


def _snap_target_pixel(props, seg_idx, click_px, click_py, snap_tol):
    """Return (pixel, is_corner) for where to drop the snap point.

    Snap to the nearest intersection with another segment if within snap_tol,
    otherwise project click onto seg's infinite line.
    """
    seg = props.segments[seg_idx]
    a = np.array([seg.x1, seg.y1])
    b = np.array([seg.x2, seg.y2])
    click = np.array([click_px, click_py])

    best = None
    for j, s in enumerate(props.segments):
        if j == seg_idx:
            continue
        inter = solver.segment_intersection(
            ((seg.x1, seg.y1), (seg.x2, seg.y2)),
            ((s.x1, s.y1), (s.x2, s.y2)))
        if inter is None:
            continue
        d = float(np.linalg.norm(inter - click))
        if d <= snap_tol and (best is None or d < best[0]):
            best = (d, inter)
    if best is not None:
        return best[1], True

    ab = b - a
    ab_len_sq = float(ab @ ab)
    if ab_len_sq < 1e-9:
        return a, False
    t = float((click - a) @ ab) / ab_len_sq
    return a + t * ab, False


# --------------------------------------------------------------------------- #
# Modal helpers
# --------------------------------------------------------------------------- #
_PASS_THROUGH = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
    'TRACKPADPAN', 'TRACKPADZOOM',
    'NDOF_MOTION', 'NDOF_BUTTON_FIT',
}


def _mouse_in_window_region(context, event):
    region = context.region
    if region is None or region.type != 'WINDOW':
        return False
    x, y = event.mouse_x, event.mouse_y
    return (region.x <= x <= region.x + region.width and
            region.y <= y <= region.y + region.height)


def _region_to_pixel(context, mx, my):
    """Image-editor region coords to image-pixel coords (y from top)."""
    region = context.region
    props = _props(context)
    u, v = region.view2d.region_to_view(mx, my)
    return u * props.image_width, (1.0 - v) * props.image_height


# =========================================================================== #
# Step 1 — Image setup
# =========================================================================== #
class IMG2CUBE_OT_setup(Operator):
    """Use the image shown in this Image Editor: create a matched camera and
    enable the drawing overlay."""
    bl_idname = "img2cube.setup"
    bl_label = "Use this image"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sp = context.space_data
        return sp and sp.type == 'IMAGE_EDITOR' and sp.image is not None

    def execute(self, context):
        props = _props(context)
        img = context.space_data.image
        props.image_name = img.name
        props.image_width = img.size[0]
        props.image_height = img.size[1]

        scn = context.scene
        scn.render.resolution_x = props.image_width
        scn.render.resolution_y = props.image_height

        cam_obj = bpy.data.objects.get("img2cube_camera")
        if cam_obj is None:
            cam_data = bpy.data.cameras.new("img2cube_camera")
            cam_obj = bpy.data.objects.new("img2cube_camera", cam_data)
            scn.collection.objects.link(cam_obj)
        cam = cam_obj.data
        cam.show_background_images = True
        cam.background_images.clear()
        bg = cam.background_images.new()
        bg.image = img
        bg.alpha = 1.0
        scn.camera = cam_obj

        draw.found_points.clear()
        draw.enable()
        self.report({'INFO'}, f"img2cube ready on '{img.name}'")
        return {'FINISHED'}


# =========================================================================== #
# Step 2 — Draw parallel lines
# =========================================================================== #
class IMG2CUBE_OT_draw_lines(Operator):
    """Click TWO points to place one line (click A, click B). Repeat for more.
    Right-click or Esc to stop. Scroll wheel and middle-mouse pan/zoom normally."""
    bl_idname = "img2cube.draw_lines"
    bl_label = "Draw lines (active axis)"
    _p1 = None

    @classmethod
    def poll(cls, context):
        p = _props(context)
        return p.image_name and context.space_data.type == 'IMAGE_EDITOR'

    def invoke(self, context, event):
        props = _props(context)
        if props.is_drawing:
            props.is_drawing = False
            draw.preview["active"] = False
            context.area.tag_redraw()
            return {'FINISHED'}

        self._p1 = None
        props.is_drawing = True
        props.is_picking_origin = False
        props.is_adding_snaps = False
        draw.preview["active"] = True
        draw.preview["axis"] = int(props.active_axis)
        draw.preview["p1"] = draw.preview["p2"] = None
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        props = _props(context)
        if not props.is_drawing:
            return {'FINISHED'}
        if event.type in _PASS_THROUGH:
            return {'PASS_THROUGH'}

        in_region = _mouse_in_window_region(context, event)

        if event.type == 'MOUSEMOVE':
            if self._p1 is not None and in_region:
                draw.preview["p1"] = self._p1
                draw.preview["p2"] = _region_to_pixel(
                    context, event.mouse_region_x, event.mouse_region_y)
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not in_region:
                return {'PASS_THROUGH'}
            pt = _region_to_pixel(context, event.mouse_region_x,
                                  event.mouse_region_y)
            if self._p1 is None:
                self._p1 = pt
                draw.preview["p1"] = pt
                draw.preview["p2"] = pt
            else:
                seg = props.segments.add()
                seg.axis = int(props.active_axis)
                seg.x1, seg.y1 = self._p1
                seg.x2, seg.y2 = pt
                self._p1 = None
                draw.preview["p1"] = draw.preview["p2"] = None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'ESC', 'RIGHTMOUSE'}:
            props.is_drawing = False
            draw.preview["active"] = False
            draw.preview["p1"] = draw.preview["p2"] = None
            context.area.tag_redraw()
            return {'FINISHED'}

        return {'PASS_THROUGH'}


class IMG2CUBE_OT_clear_axis(Operator):
    """Remove all drawn lines for the currently active axis."""
    bl_idname = "img2cube.clear_axis"
    bl_label = "Clear active axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = _props(context)
        axis = int(props.active_axis)
        for i in reversed(range(len(props.segments))):
            if props.segments[i].axis == axis:
                if props.selected_segment == i:
                    props.selected_segment = -1
                props.segments.remove(i)
        context.area.tag_redraw()
        return {'FINISHED'}


class IMG2CUBE_OT_remove_segment(Operator):
    """Delete this specific drawn line."""
    bl_idname = "img2cube.remove_segment"
    bl_label = "Remove segment"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()

    def execute(self, context):
        props = _props(context)
        if 0 <= self.index < len(props.segments):
            if props.selected_segment == self.index:
                props.selected_segment = -1
            props.segments.remove(self.index)
        context.area.tag_redraw()
        return {'FINISHED'}


# =========================================================================== #
# Click-to-select a line (used by the Step-4 scale UI and panel buttons)
# =========================================================================== #
class IMG2CUBE_OT_pick_segment(Operator):
    """Click a drawn line in the image to select it (for scaling)."""
    bl_idname = "img2cube.pick_segment"
    bl_label = "Pick line in image"

    @classmethod
    def poll(cls, context):
        p = _props(context)
        return (p.image_name and len(p.segments) > 0
                and context.space_data.type == 'IMAGE_EDITOR')

    def invoke(self, context, event):
        props = _props(context)
        if props.is_selecting:
            props.is_selecting = False
            context.area.tag_redraw()
            return {'FINISHED'}
        props.is_selecting = True
        props.is_drawing = False
        props.is_picking_origin = False
        props.is_adding_snaps = False
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        props = _props(context)
        if not props.is_selecting:
            return {'FINISHED'}
        if event.type in _PASS_THROUGH:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not _mouse_in_window_region(context, event):
                return {'PASS_THROUGH'}
            px, py = _region_to_pixel(context, event.mouse_region_x,
                                      event.mouse_region_y)
            idx, dist, _t = _find_segment_near_pixel(props, px, py, tol_px=15.0)
            if idx >= 0:
                props.selected_segment = idx
                props.is_selecting = False
                self.report(
                    {'INFO'},
                    f"Selected line [{_AXIS_LABEL[props.segments[idx].axis]}] "
                    f"#{idx} — click its 'set length' button to enter the size")
                context.area.tag_redraw()
                return {'FINISHED'}
            self.report({'INFO'}, "No line near click — try again")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'ESC', 'RIGHTMOUSE'}:
            props.selected_segment = -1
            props.is_selecting = False
            context.area.tag_redraw()
            return {'FINISHED'}
        return {'PASS_THROUGH'}


class IMG2CUBE_OT_select_segment(Operator):
    """Select or deselect a segment by index (from the panel list)."""
    bl_idname = "img2cube.select_segment"
    bl_label = "Select segment"
    index: IntProperty()

    def execute(self, context):
        props = _props(context)
        if 0 <= self.index < len(props.segments):
            if props.selected_segment == self.index:
                props.selected_segment = -1
            else:
                props.selected_segment = self.index
        context.area.tag_redraw()
        return {'FINISHED'}


class IMG2CUBE_OT_set_segment_length(Operator):
    """Set the real-world length of a single segment (pops up a dialog)."""
    bl_idname = "img2cube.set_segment_length"
    bl_label = "Set known real length"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()
    length: FloatProperty(name="Real length", default=1.0, min=0.0, unit='LENGTH')

    def invoke(self, context, event):
        props = _props(context)
        if 0 <= self.index < len(props.segments):
            self.length = props.segments[self.index].known_length or 1.0
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        props = _props(context)
        if 0 <= self.index < len(props.segments):
            props.segments[self.index].known_length = self.length
            _recompute_scale(props)
            if props.origin_set:
                _reposition_camera_from_origin(context)
        context.area.tag_redraw()
        return {'FINISHED'}


def _recompute_scale(props):
    lengths = [s.known_length for s in props.segments if s.known_length > 0]
    if lengths:
        props.scale_factor = float(np.mean(lengths))
        props.n_measured = len(lengths)
    else:
        props.scale_factor = 1.0
        props.n_measured = 0


# =========================================================================== #
# Step 3a — Solve camera
# =========================================================================== #
class IMG2CUBE_OT_solve(Operator):
    """Solve focal length, principal point, and orientation from the drawn VPs."""
    bl_idname = "img2cube.solve"
    bl_label = "Solve camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = _props(context)
        vps, axes, residuals = _gather_vps(props)
        if len(vps) < 2:
            self.report({'ERROR'},
                        "Need at least 2 axes with 2+ lines each.")
            return {'CANCELLED'}

        wh = (props.image_width, props.image_height)
        try:
            intr = solver.solve_intrinsics(vps, wh)
            R = solver.solve_rotation(vps, intr["pp"], intr["f"])
        except Exception as e:
            self.report({'ERROR'}, f"Solve failed: {e}")
            return {'CANCELLED'}

        cam_obj = bpy.data.objects.get("img2cube_camera")
        if cam_obj is None:
            self.report({'ERROR'}, "Run 'Use this image' first.")
            return {'CANCELLED'}

        D = 10.0
        centre = -D * R[:, 2]
        convert.apply_to_camera(
            cam_obj, {"f": intr["f"], "pp": intr["pp"], "R": R, "centre": centre},
            props.image_width, props.image_height)

        props.solved = True
        props.solved_focal_px = intr["f"]
        props.solved_pp_x, props.solved_pp_y = intr["pp"]
        props.solved_n_vps = len(vps)
        props.solve_residual_px = float(np.mean(residuals)) if residuals else 0.0
        props.origin_set = False
        props.origin_seg_a = -1
        props.origin_seg_b = -1

        self.report({'INFO'},
                    f"Solved from {len(vps)} VP(s). f={intr['f']:.0f}px, "
                    f"lens={cam_obj.data.lens:.1f}mm, "
                    f"residual={props.solve_residual_px:.2f}px")
        return {'FINISHED'}


# =========================================================================== #
# Step 3b — Establish origin (auto-snaps to exact intersection)
# =========================================================================== #
class IMG2CUBE_OT_set_origin(Operator):
    """Click near the intersection of two drawn lines of DIFFERENT axes; the
    exact 2D intersection point is computed automatically and used as the world
    origin (0,0,0).  Esc/right-click to cancel."""
    bl_idname = "img2cube.set_origin"
    bl_label = "Establish Origin (click an intersection)"

    @classmethod
    def poll(cls, context):
        p = _props(context)
        return (p.solved and len(p.segments) >= 2
                and context.space_data.type == 'IMAGE_EDITOR')

    def invoke(self, context, event):
        props = _props(context)
        if props.is_picking_origin:
            props.is_picking_origin = False
            context.area.tag_redraw()
            return {'FINISHED'}
        props.is_picking_origin = True
        props.is_drawing = False
        props.is_adding_snaps = False
        props.is_selecting = False
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        props = _props(context)
        if not props.is_picking_origin:
            return {'FINISHED'}
        if event.type in _PASS_THROUGH:
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not _mouse_in_window_region(context, event):
                return {'PASS_THROUGH'}
            px, py = _region_to_pixel(context, event.mouse_region_x,
                                      event.mouse_region_y)

            # Find the nearest real intersection of two different-axis lines.
            diag = math.hypot(props.image_width, props.image_height)
            snap_tol = max(20.0, 0.01 * diag)
            hit = _find_nearest_intersection_pixel(props, px, py, snap_tol)

            if hit is None:
                self.report({'WARNING'},
                            "Click closer to where two different-axis "
                            "lines cross.")
                return {'RUNNING_MODAL'}

            ix, iy, seg_a_idx, seg_b_idx = hit

            # Use the EXACT intersection pixel, not the raw click.
            props.origin_pixel_x = ix
            props.origin_pixel_y = iy
            props.origin_seg_a = seg_a_idx
            props.origin_seg_b = seg_b_idx
            props.origin_set = True

            _reposition_camera_from_origin(context)

            props.is_picking_origin = False
            context.area.tag_redraw()
            sa = props.segments[seg_a_idx]
            sb = props.segments[seg_b_idx]
            self.report({'INFO'},
                        f"Origin snapped to intersection of "
                        f"[{_AXIS_LABEL[sa.axis]}] #{seg_a_idx} and "
                        f"[{_AXIS_LABEL[sb.axis]}] #{seg_b_idx}")
            return {'FINISHED'}

        if event.type in {'ESC', 'RIGHTMOUSE'}:
            props.is_picking_origin = False
            context.area.tag_redraw()
            return {'CANCELLED'}

        return {'PASS_THROUGH'}


def _reposition_camera_from_origin(context):
    """Compute the 3D origin in cam-vision frame from the two picked segments,
    then re-place the Blender camera so that this 3D point lies at world (0,0,0).
    Now uses metric depths from the graph when available."""
    props = _props(context)
    if not props.origin_set or not props.solved:
        return
    cam_obj = bpy.data.objects.get("img2cube_camera")
    if cam_obj is None:
        return

    graph = build_node_graph(props)
    metric = compute_placement_depths(props, graph) if graph else {}

    R = graph["R"] if graph else _solve_rotation_from_props(props)[0]
    axes_used = graph["axes_used"] if graph else _solve_rotation_from_props(props)[1]
    pp = (props.solved_pp_x, props.solved_pp_y)
    f = props.solved_focal_px
    scale = props.scale_factor if props.scale_factor > 0 else 1.0

    seg_a_idx = props.origin_seg_a
    seg_b_idx = props.origin_seg_b
    seg_a = props.segments[seg_a_idx]
    seg_b = props.segments[seg_b_idx]

    epa = _segment_3d_in_cam(props, seg_a, R, axes_used, scale,
                              seg_idx=seg_a_idx, metric_depths=metric,
                              graph=graph)
    epb = _segment_3d_in_cam(props, seg_b, R, axes_used, scale,
                              seg_idx=seg_b_idx, metric_depths=metric,
                              graph=graph)
    if epa is None or epb is None:
        return

    # Ray through the (snapped) origin pixel
    r = solver.ray_through_pixel(props.origin_pixel_x, props.origin_pixel_y,
                                 pp, f)
    A1 = mathutils.Vector((0.0, 0.0, 0.0))
    A2 = mathutils.Vector((float(r[0]), float(r[1]), float(r[2])))

    points = []
    for ep in (epa, epb):
        P1, P2 = ep
        B1 = mathutils.Vector((float(P1[0]), float(P1[1]), float(P1[2])))
        B2 = mathutils.Vector((float(P2[0]), float(P2[1]), float(P2[2])))
        hit = _intersect_line_line_safe(A1, A2, B1, B2)
        if hit is not None:
            points.append(hit)
    if not points:
        return
    origin_cam_v = sum(points, mathutils.Vector()) / len(points)
    origin_cam_v_np = np.array([origin_cam_v.x, origin_cam_v.y, origin_cam_v.z])

    centre = -R.T @ origin_cam_v_np
    convert.apply_to_camera(
        cam_obj,
        {"f": f, "pp": pp, "R": R, "centre": centre},
        props.image_width, props.image_height)


# =========================================================================== #
# Step 5 — Add snap points (now works on extension lines too)
# =========================================================================== #
class IMG2CUBE_OT_add_snap_points(Operator):
    """Click any drawn line OR its extension (thin guide) in the image to add a
    3D vertex at the exact spot the click ray meets the line in 3D.
    Esc/right-click to stop."""
    bl_idname = "img2cube.add_snap_points"
    bl_label = "Add snap points along lines"

    @classmethod
    def poll(cls, context):
        p = _props(context)
        return (p.solved and p.origin_set
                and context.space_data.type == 'IMAGE_EDITOR')

    def invoke(self, context, event):
        props = _props(context)
        if props.is_adding_snaps:
            props.is_adding_snaps = False
            context.area.tag_redraw()
            return {'FINISHED'}
        props.is_adding_snaps = True
        props.is_drawing = False
        props.is_picking_origin = False
        props.is_selecting = False
        _ensure_output_collection(props)
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        props = _props(context)
        if not props.is_adding_snaps:
            return {'FINISHED'}
        if event.type in _PASS_THROUGH:
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not _mouse_in_window_region(context, event):
                return {'PASS_THROUGH'}
            px, py = _region_to_pixel(context, event.mouse_region_x,
                                      event.mouse_region_y)
            # Try extension lines too (allow_extension=True)
            idx, dist, _t = _find_segment_near_pixel(
                props, px, py, tol_px=20.0, allow_extension=True)
            if idx < 0:
                self.report({'INFO'}, "Click closer to a drawn line or its extension.")
                return {'RUNNING_MODAL'}
            ok = _add_snap_point_on_segment(context, idx, px, py)
            if ok:
                self.report({'INFO'}, f"+ snap point on line #{idx}")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'ESC', 'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            props.is_adding_snaps = False
            context.area.tag_redraw()
            return {'FINISHED'}

        return {'PASS_THROUGH'}


def _ensure_output_collection(props):
    coll = bpy.data.collections.get("img2cube_points")
    if coll is None:
        coll = bpy.data.collections.new("img2cube_points")
        bpy.context.scene.collection.children.link(coll)
    return coll


def _add_snap_point_on_segment(context, seg_idx, click_px, click_py):
    """Place a snap-point Empty exactly ON the drawn line's 3D extension,
    using metric depths when available.  Corner-snaps to nearby intersections."""
    props = _props(context)
    cam_obj = bpy.data.objects.get("img2cube_camera")
    if cam_obj is None:
        return False

    diag = math.hypot(props.image_width, props.image_height)
    snap_tol = max(10.0, 0.006 * diag)
    target, is_corner = _snap_target_pixel(props, seg_idx,
                                           click_px, click_py, snap_tol)
    tx, ty = float(target[0]), float(target[1])

    seg = props.segments[seg_idx]

    # Build graph + placement depths for accurate 3D placement
    graph = build_node_graph(props)
    metric = compute_placement_depths(props, graph) if graph else {}
    R = graph["R"] if graph else _solve_rotation_from_props(props)[0]
    axes_used = graph["axes_used"] if graph else _solve_rotation_from_props(props)[1]
    scale = props.scale_factor if props.scale_factor > 0 else 1.0

    ep = _segment_3d_in_cam(props, seg, R, axes_used, scale,
                             seg_idx=seg_idx, metric_depths=metric,
                             graph=graph)
    if ep is None:
        return False
    P1, P2 = ep

    origin_w, dir_w = _ray_through_pixel_blender_world(props, cam_obj, tx, ty)
    A1 = origin_w
    A2 = origin_w + dir_w

    B1 = _cam_vision_to_blender_world(P1, cam_obj)
    B2 = _cam_vision_to_blender_world(P2, cam_obj)

    hit = _closest_point_on_second_line(A1, A2, B1, B2)
    if hit is None:
        return False

    coll = _ensure_output_collection(props)
    n = len([o for o in coll.objects if o.name.startswith("i2c_pt")])
    e = bpy.data.objects.new(f"i2c_pt.{n:03d}", None)
    e.empty_display_type = 'PLAIN_AXES'
    e.empty_display_size = 0.15
    e.location = hit
    coll.objects.link(e)

    draw.found_points.append((tx, ty))
    return True


class IMG2CUBE_OT_clear_snap_points(Operator):
    """Delete all generated snap points and wireframe."""
    bl_idname = "img2cube.clear_snap_points"
    bl_label = "Clear all snap points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        coll = bpy.data.collections.get("img2cube_points")
        if coll:
            for ob in list(coll.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
        for m in list(bpy.data.meshes):
            if m.users == 0 and m.name.startswith("img2cube_"):
                bpy.data.meshes.remove(m)
        draw.found_points.clear()
        context.area.tag_redraw()
        return {'FINISHED'}


# =========================================================================== #
CLASSES = (
    IMG2CUBE_OT_setup,
    IMG2CUBE_OT_draw_lines,
    IMG2CUBE_OT_clear_axis,
    IMG2CUBE_OT_remove_segment,
    IMG2CUBE_OT_pick_segment,
    IMG2CUBE_OT_select_segment,
    IMG2CUBE_OT_set_segment_length,
    IMG2CUBE_OT_solve,
    IMG2CUBE_OT_set_origin,
    IMG2CUBE_OT_add_snap_points,
    IMG2CUBE_OT_clear_snap_points,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
