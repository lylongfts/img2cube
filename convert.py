# SPDX-License-Identifier: GPL-3.0-or-later
"""
Bridge between the vision-frame solver and Blender's camera.

CONVENTION NOTE (the thing to verify first on a real machine):
  solver frame  : x right, y DOWN, camera looks +Z   (OpenCV-style)
  blender camera: x right, y UP,   camera looks -Z
So a blender camera-local vector maps to the solver frame by flipping y and z:
  v_solver = diag(1,-1,-1) @ v_blender_local
The world axes stay the user's three real-world directions, mapped onto
Blender's global X/Y/Z, so a matched building's edges run along global axes.
"""

import numpy as np
import mathutils

# Flip matrix between blender-camera-local and vision-camera frames.
_FLIP = np.diag([1.0, -1.0, -1.0])


def focal_px_to_lens_mm(focal_px, image_w, sensor_width_mm=36.0):
    """Blender lens (mm) for a horizontal-fit sensor."""
    return focal_px * sensor_width_mm / image_w


def principal_to_shift(pp_x, pp_y, image_w, image_h):
    """
    Blender camera shift_x / shift_y for an off-centre principal point.
    Blender normalises shift by the LARGER image dimension. Image y is top-down,
    Blender shift_y is +up, hence the y sign flip.
    NOTE: sign/scale here is the most likely thing to need a tweak after testing.
    """
    big = max(image_w, image_h)
    shift_x = -(pp_x - image_w / 2.0) / big
    shift_y = (pp_y - image_h / 2.0) / big
    return shift_x, shift_y


def world_matrix_from_solution(R_cw_vision, cam_centre_world=None):
    """
    Build a Blender camera matrix_world from the solver's rotation.

    R_cw_vision: 3x3, columns are world X/Y/Z expressed in the vision camera
                 frame (i.e. camera_from_world).
    Derivation:
        v_cam_vision = R_cw_vision @ (p_world - C)
        v_cam_vision = FLIP @ v_cam_blender_local
      => p_world = C + R_cw_vision.T @ FLIP @ v_cam_blender_local
      => rotation (local->world) R_wb = R_cw_vision.T @ FLIP
    """
    R_wb = R_cw_vision.T @ _FLIP
    M = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = float(R_wb[i, j])
    if cam_centre_world is not None:
        for i in range(3):
            M[i][3] = float(cam_centre_world[i])
    return M


def apply_to_camera(cam_obj, solution, image_w, image_h):
    """
    solution: dict with keys 'R' (3x3 np), 'pp' (2,), 'f' (px),
              optionally 'centre' (3,) world camera position.
    Sets lens, sensor, shift and matrix_world on cam_obj.
    """
    cam = cam_obj.data
    cam.type = 'PERSP'
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 36.0
    cam.lens = focal_px_to_lens_mm(solution['f'], image_w, cam.sensor_width)

    sx, sy = principal_to_shift(solution['pp'][0], solution['pp'][1],
                                image_w, image_h)
    cam.shift_x, cam.shift_y = sx, sy

    centre = solution.get('centre')
    cam_obj.matrix_world = world_matrix_from_solution(solution['R'], centre)
