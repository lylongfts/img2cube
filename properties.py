# SPDX-License-Identifier: GPL-3.0-or-later
"""Data model: drawn segments + solve state + origin + scale, stored on Scene."""

import bpy
from bpy.props import (
    FloatProperty, IntProperty, StringProperty, EnumProperty,
    BoolProperty, CollectionProperty, PointerProperty,
)
from bpy.types import PropertyGroup

AXIS_ITEMS = [
    ('0', "X axis", "First real-world direction (red)", 'EVENT_X', 0),
    ('1', "Y axis", "Second real-world direction (green)", 'EVENT_Y', 1),
    ('2', "Z axis", "Third real-world direction (blue)", 'EVENT_Z', 2),
]

AXIS_COLORS = {
    0: (0.95, 0.25, 0.25, 1.0),
    1: (0.30, 0.85, 0.30, 1.0),
    2: (0.30, 0.55, 0.95, 1.0),
}


def _update_known_length(self, context):
    from .operators import _recompute_scale, _reposition_camera_from_origin
    props = context.scene.img2cube
    _recompute_scale(props)
    if props.origin_set:
        _reposition_camera_from_origin(context)


class Img2CubeSegment(PropertyGroup):
    """One drawn segment in IMAGE PIXEL coordinates (y measured from top)."""
    x1: FloatProperty()
    y1: FloatProperty()
    x2: FloatProperty()
    y2: FloatProperty()
    axis: IntProperty(default=0)
    known_length: FloatProperty(
        name="Real length", default=0.0, min=0.0, unit='LENGTH',
        description="Real-world length of this line (0 = unmeasured). Lines "
                    "with a known length feed the scene scale.",
        update=_update_known_length)


class Img2CubeProps(PropertyGroup):
    image_name: StringProperty(name="Image")
    image_width: IntProperty(default=0)
    image_height: IntProperty(default=0)

    active_axis: EnumProperty(name="Active axis", items=AXIS_ITEMS, default='0')
    segments: CollectionProperty(type=Img2CubeSegment)

    # currently-selected segment (click-to-select in the image, or panel click)
    selected_segment: IntProperty(default=-1)

    # ephemeral UI state for the modal operators
    is_drawing:     BoolProperty(default=False)
    is_picking_origin: BoolProperty(default=False)
    is_adding_snaps:   BoolProperty(default=False)
    is_selecting:   BoolProperty(default=False)

    # --- Step 3a: Solve results ------------------------------------------ #
    solved:             BoolProperty(default=False)
    solved_focal_px:    FloatProperty(default=0.0)
    solved_pp_x:        FloatProperty(default=0.0)
    solved_pp_y:        FloatProperty(default=0.0)
    solved_n_vps:       IntProperty(default=0)
    solve_residual_px:  FloatProperty(default=0.0)

    # --- Step 3b: Origin (clicked intersection) -------------------------- #
    origin_set:      BoolProperty(default=False)
    origin_pixel_x:  FloatProperty(default=0.0)
    origin_pixel_y:  FloatProperty(default=0.0)
    # the two segment indices whose intersection the user clicked
    origin_seg_a:    IntProperty(default=-1)
    origin_seg_b:    IntProperty(default=-1)

    # --- Step 4: Global scene scale (average of known_length values) ----- #
    scale_factor:    FloatProperty(default=1.0)
    n_measured:      IntProperty(default=0)

    show_dimensions: BoolProperty(
        name="Show Dimensions in Viewport",
        default=False,
        description="Overlay calculated lengths on all drawn lines in the viewport"
    )

    # --- Step 5: Snap point options -------------------------------------- #
    pass


CLASSES = (Img2CubeSegment, Img2CubeProps)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.img2cube = PointerProperty(type=Img2CubeProps)


def unregister():
    del bpy.types.Scene.img2cube
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
