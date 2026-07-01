# SPDX-License-Identifier: GPL-3.0-or-later
"""
img2cube core solver.

Pure-numpy camera calibration from vanishing points. No bpy import here on
purpose: this module can be unit-tested outside Blender, and Blender ships
numpy so it works unchanged inside the extension.

Pipeline:
    1. Each axis = a group of image-space line segments that are PARALLEL in
       reality. Their vanishing point is the least-squares common intersection.
    2. From 2 or 3 mutually-orthogonal vanishing points we recover the camera
       intrinsics (focal length, principal point) and the rotation.
    3. A single known real-world length on one drawn segment fixes the scale,
       which lets us place the camera at a metric distance.

Coordinate convention used INSIDE this module ("vision" frame):
    - image pixels: x right, y DOWN, origin top-left.
    - camera looks along +Z, x right, y down (OpenCV-style).
The Blender conversion (Y/Z swap, -Z view direction, y-flip) lives in the
extension layer, not here, so the math stays clean and self-consistent.
"""

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Lines and vanishing points
# --------------------------------------------------------------------------- #
def line_through(p1, p2):
    """Homogeneous line [a, b, c] through two 2D image points (a*x+b*y+c=0)."""
    p1h = np.array([p1[0], p1[1], 1.0])
    p2h = np.array([p2[0], p2[1], 1.0])
    return np.cross(p1h, p2h)


def vanishing_point(segments):
    """
    Least-squares vanishing point for a set of segments that are parallel in
    reality. `segments` is a list of ((x1,y1),(x2,y2)).

    Returns a homogeneous 3-vector. If the third component is ~0 the lines are
    (near-)parallel in the image and the VP is at infinity; callers should use
    `vp_is_finite` / `vp_to_xy` to handle that.
    """
    if len(segments) < 2:
        raise ValueError("Need at least 2 segments to define a vanishing point.")

    rows = []
    for (a, b) in segments:
        l = line_through(a, b)
        n = np.linalg.norm(l[:2])  # normalise by the (a,b) part for conditioning
        if n < EPS:
            continue
        rows.append(l / n)
    A = np.asarray(rows, dtype=float)

    # VP v minimises ||A v||  ->  smallest right singular vector.
    _, _, Vt = np.linalg.svd(A)
    v = Vt[-1]
    # Fix a stable sign so downstream direction signs are reproducible.
    if v[2] < 0:
        v = -v
    return v


def vp_is_finite(v, tol=1e-6):
    return abs(v[2]) > tol * max(1.0, np.linalg.norm(v[:2]))


def vp_to_xy(v):
    """Homogeneous VP -> Euclidean image point. Assumes finite (check first)."""
    return np.array([v[0] / v[2], v[1] / v[2]])


# --------------------------------------------------------------------------- #
# Camera intrinsics from vanishing points
# --------------------------------------------------------------------------- #
def orthocenter(a, b, c):
    """Orthocenter of the triangle (a, b, c) — the principal point for 3 VPs."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    bc, ca = c - b, a - c
    # altitude from A: (P-A)·BC = 0 ; altitude from B: (P-B)·CA = 0
    M = np.array([bc, ca])
    rhs = np.array([bc @ a, ca @ b])
    return np.linalg.solve(M, rhs)


def focal_from_pair(v1, v2, pp):
    """f from two orthogonal finite VPs and a known principal point pp."""
    d = -np.dot(np.asarray(v1) - pp, np.asarray(v2) - pp)
    if d <= 0:
        return None  # geometry inconsistent (VPs not on opposite sides)
    return float(np.sqrt(d))


def solve_intrinsics(vps, image_wh):
    """
    vps: list of 2 or 3 Euclidean vanishing points (finite ones).
    image_wh: (width, height) in pixels, used for the default principal point.

    Returns dict {pp, f, n_vps}. Raises if geometry is degenerate.

    - 3 VPs: principal point = orthocenter of the VP triangle (no assumption).
    - 2 VPs: principal point assumed at image centre (valid for un-cropped,
             non-shifted shots), f from the orthogonality of the pair.
    """
    w, h = image_wh
    centre = np.array([w / 2.0, h / 2.0])

    if len(vps) >= 3:
        v1, v2, v3 = (np.asarray(v) for v in vps[:3])
        pp = orthocenter(v1, v2, v3)
        fs = [focal_from_pair(a, b, pp)
              for a, b in ((v1, v2), (v1, v3), (v2, v3))]
        fs = [f for f in fs if f]
        if not fs:
            raise ValueError("Could not recover a positive focal length from 3 VPs.")
        return {"pp": pp, "f": float(np.mean(fs)), "n_vps": 3}

    if len(vps) == 2:
        v1, v2 = np.asarray(vps[0]), np.asarray(vps[1])
        pp = centre
        f = focal_from_pair(v1, v2, pp)
        if f is None:
            raise ValueError("2-VP solve failed: VPs inconsistent with a centred "
                             "principal point (image may be cropped off-centre).")
        return {"pp": pp, "f": f, "n_vps": 2}

    raise ValueError("solve_intrinsics needs at least 2 finite vanishing points.")


# --------------------------------------------------------------------------- #
# Rotation from vanishing points
# --------------------------------------------------------------------------- #
def _axis_dir(vp, pp, f):
    """Unit camera-space direction for a finite VP."""
    d = np.array([vp[0] - pp[0], vp[1] - pp[1], f])
    return d / np.linalg.norm(d)


def solve_rotation(vps, pp, f, axis_order=("x", "y", "z")):
    """
    Build the world->camera rotation from the VP directions.

    With 3 VPs we use all three directions (re-orthonormalised). With 2 VPs the
    third axis is the cross product. Sign of each axis is chosen so the basis is
    right-handed and the camera looks toward +Z (positive scene depth).

    Returns a 3x3 numpy array R whose COLUMNS are the world X/Y/Z axes expressed
    in the camera frame (i.e. R = R_camera_from_world).
    """
    dirs = [_axis_dir(np.asarray(v), pp, f) for v in vps]

    if len(dirs) >= 3:
        x, y, z = dirs[0], dirs[1], dirs[2]
        # Re-orthonormalise (Gram-Schmidt) to absorb small inconsistencies.
        x = x / np.linalg.norm(x)
        y = y - (y @ x) * x
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
    else:
        x, y = dirs[0], dirs[1]
        y = y - (y @ x) * x
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)

    R = np.column_stack([x, y, z])

    # Enforce right-handedness.
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]

    return R


# --------------------------------------------------------------------------- #
# Scale and camera placement
# --------------------------------------------------------------------------- #
def backproject_ray(px_pt, pp, f):
    """Unit ray (camera frame) through an image pixel."""
    d = np.array([px_pt[0] - pp[0], px_pt[1] - pp[1], f])
    return d / np.linalg.norm(d)


def solve_scale(seg_px, known_length, axis_world_dir, pp, f):
    """
    Recover metric scale from one segment whose real length is known.

    seg_px: ((x1,y1),(x2,y2)) endpoints of the measured segment in pixels.
    known_length: real length of that segment (e.g. metres).
    axis_world_dir: the world-axis unit vector the segment runs along, expressed
                    in the camera frame (a column of R).

    The two endpoints lie on rays r1, r2 from the camera. The segment lies along
    a known direction `u`, so endpoint depths satisfy
        t1*r1 - t2*r2 = L * u            (vector eq, 3 rows, 2 unknowns t1,t2)
    solved in least squares; the segment's apparent length then maps to `L`,
    giving the world unit -> metre factor. Returns metres-per-world-unit, here
    normalised so that 1 world unit == known_length over that segment, i.e. we
    return the camera distance scale used to place the camera.
    """
    r1 = backproject_ray(seg_px[0], pp, f)
    r2 = backproject_ray(seg_px[1], pp, f)
    u = np.asarray(axis_world_dir, dtype=float)
    u = u / np.linalg.norm(u)

    # t1*r1 - t2*r2 = L*u  ->  [r1, -r2] [t1; t2] = L*u
    A = np.column_stack([r1, -r2])
    sol, *_ = np.linalg.lstsq(A, known_length * u, rcond=None)
    t1, t2 = sol
    return {"t1": float(t1), "t2": float(t2),
            "endpoint1_cam": r1 * t1, "endpoint2_cam": r2 * t2}


# --------------------------------------------------------------------------- #
# Intersections
# --------------------------------------------------------------------------- #
def segment_intersection(seg_a, seg_b):
    """
    Intersection of the infinite lines through two segments, in image pixels.
    Returns (x, y) or None if parallel.
    """
    la = line_through(*seg_a)
    lb = line_through(*seg_b)
    p = np.cross(la, lb)
    if abs(p[2]) < EPS:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


# --------------------------------------------------------------------------- #
# 3D reasoning: rays, planes built from co-planar segments, line-plane meets
# --------------------------------------------------------------------------- #
def ray_through_pixel(px, py, pp, f):
    """Unit camera-space ray (vision frame) through an image pixel."""
    d = np.array([px - pp[0], py - pp[1], f], dtype=float)
    return d / np.linalg.norm(d)


def solve_segment_depths(seg, axis_dir_cam, pp, f):
    """
    Given an image segment and the segment's known real-world direction in
    camera frame, recover the two endpoints' depths along their camera rays
    up to overall scale, then normalise so the segment has 3D length 1.

    Math:
        endpoint_1_3D = t1 * r1
        endpoint_2_3D = t2 * r2
        endpoint_2_3D - endpoint_1_3D = L * d_axis
        => [-r1, r2, -d_axis] @ [t1; t2; L] = 0    (homogeneous 3x3 system)
    The system is rank-2 because, for a true projection of a segment parallel
    to d_axis, the vector d_axis lies in the plane spanned by r1, r2. The
    1-D null space gives (t1, t2, L) up to scale; we fix L=1.

    Returns dict {t1, t2, P1, P2} (depths and 3D endpoints in cam frame), or
    None if degenerate.
    """
    r1 = ray_through_pixel(seg[0][0], seg[0][1], pp, f)
    r2 = ray_through_pixel(seg[1][0], seg[1][1], pp, f)
    d = np.asarray(axis_dir_cam, dtype=float)
    d = d / np.linalg.norm(d)

    M = np.column_stack([-r1, r2, -d])
    _, s, Vt = np.linalg.svd(M)
    sol = Vt[-1]
    t1, t2, L = sol
    # The SVD picks one of the two sign branches arbitrarily. The physical
    # constraint is that both endpoints are IN FRONT of the camera, i.e. t1>0
    # and t2>0. If they're negative, flip the whole null vector. L's sign just
    # tells us which way the user drew the segment (reverse order) — we use
    # |L| since the geometric line is the same either way.
    if t1 < 0 and t2 < 0:
        t1, t2, L = -t1, -t2, -L
    if t1 <= 0 or t2 <= 0:
        return None     # endpoints behind camera -> bad data
    L = abs(L)
    if L < 1e-9:
        return None

    t1, t2 = t1 / L, t2 / L
    return {"t1": float(t1), "t2": float(t2),
            "P1": t1 * r1, "P2": t2 * r2}


def plane_from_two_image_segments(seg_a, seg_b, axis_dir_cam, pp, f,
                                  scale_factor=1.0):
    """
    Build the 3D wall plane (in camera frame) containing two image segments
    that are parallel-in-reality and lie on the same physical wall.

    Algorithm:
        1. For each segment, recover its 3D endpoints up to overall scale via
           solve_segment_depths (the known axis direction makes this solvable).
           Both segments are normalised to 3D length 1; this implicitly assumes
           the two segments have similar real-world lengths, which holds for
           typical "top and bottom of a wall" edges.
        2. Wall contains both 3D segments, both parallel to d_axis. Its normal
           is perpendicular to d_axis AND to the vector between the two
           segments' midpoints.
        3. scale_factor multiplies the wall position (depth) and lets the user
           tie this wall's units to a known real-world length via Set Scale.

    Returns dict {n, p0, depth} or None if degenerate.
    """
    sol_a = solve_segment_depths(seg_a, axis_dir_cam, pp, f)
    sol_b = solve_segment_depths(seg_b, axis_dir_cam, pp, f)
    if sol_a is None or sol_b is None:
        return None

    Pa = (sol_a["P1"] + sol_a["P2"]) / 2.0 * scale_factor
    Pb = (sol_b["P1"] + sol_b["P2"]) / 2.0 * scale_factor
    connector = Pb - Pa
    if np.linalg.norm(connector) < EPS:
        return None

    d = np.asarray(axis_dir_cam, dtype=float)
    d = d / np.linalg.norm(d)

    n = np.cross(d, connector)
    nrm = np.linalg.norm(n)
    if nrm < EPS:
        return None
    n = n / nrm

    return {"n": n, "p0": Pa,
            "depth": float((sol_a["t1"] + sol_a["t2"]) / 2.0 * scale_factor)}


def line_plane_intersection(ray_origin, ray_dir, plane):
    """
    Intersect a parametric ray (origin + t*dir, t>0) with a plane dict {n,p0}.
    Returns the 3D point in the same frame, or None if (near-)parallel/behind.
    """
    n = plane["n"]
    p0 = plane["p0"]
    denom = float(np.dot(ray_dir, n))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(p0 - ray_origin, n) / denom)
    if t <= 1e-6:
        return None
    return ray_origin + t * ray_dir


def vp_residual_px(segments, vp_xy):
    """
    Mean perpendicular distance (in pixels) from each segment's infinite line
    to the vanishing point. Smaller = the user's parallel lines really do meet
    at this VP. Used as a confidence metric in the UI.
    """
    if len(segments) == 0:
        return 0.0
    dists = []
    for (a, b) in segments:
        l = line_through(a, b)
        n = np.linalg.norm(l[:2])
        if n < EPS:
            continue
        # signed distance from vp to the line
        d = abs(l[0] * vp_xy[0] + l[1] * vp_xy[1] + l[2]) / n
        dists.append(d)
    return float(np.mean(dists)) if dists else 0.0
