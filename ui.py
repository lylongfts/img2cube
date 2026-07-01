# SPDX-License-Identifier: GPL-3.0-or-later
"""img2cube N-panel — 5-step workflow."""

import bpy
from bpy.types import Panel
from .properties import AXIS_ITEMS

_AXIS_LABEL = {int(it[0]): it[1] for it in AXIS_ITEMS}


def _modal_active(props):
    return (props.is_drawing or props.is_picking_origin
            or props.is_adding_snaps or props.is_selecting)


class IMG2CUBE_PT_main(Panel):
    bl_label = "img2cube"
    bl_idname = "IMG2CUBE_PT_main"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "img2cube"

    def draw(self, context):
        props = context.scene.img2cube
        layout = self.layout

        # If a modal is running, show a banner up top so user always sees how to stop.
        if _modal_active(props):
            banner = layout.box()
            banner.alert = True
            if props.is_drawing:
                msg = "DRAWING — click A then B"
            elif props.is_picking_origin:
                msg = "PICK ORIGIN — click near a corner"
            elif props.is_adding_snaps:
                msg = "ADDING SNAPS — click on any line or extension"
            else:
                msg = "PICK LINE — click on a line"
            banner.label(text=msg, icon='REC')
            banner.label(text="Press  Esc  or  right-click  to stop",
                         icon='CANCEL')

        # ============================================================== #
        # 1. IMAGE
        # ============================================================== #
        box = layout.box()
        box.label(text="1. Image", icon='IMAGE_DATA')
        box.operator("img2cube.setup", icon='RESTRICT_VIEW_OFF')
        if props.image_name:
            box.label(text=f"{props.image_name}  "
                          f"{props.image_width}x{props.image_height}",
                      icon='CHECKMARK')
        if not props.image_name:
            return

        # ============================================================== #
        # 2. DRAW PARALLEL LINES
        # ============================================================== #
        box = layout.box()
        box.label(text="2. Draw parallel lines", icon='GREASEPENCIL')
        row = box.row(align=True)
        row.prop(props, "active_axis", expand=True)
        row = box.row()
        row.operator("img2cube.draw_lines",
                     text=("Stop drawing" if props.is_drawing
                           else "Draw lines (click A, click B)"),
                     icon='PAUSE' if props.is_drawing else 'GREASEPENCIL',
                     depress=props.is_drawing)

        counts = {0: 0, 1: 0, 2: 0}
        for s in props.segments:
            counts[s.axis] = counts.get(s.axis, 0) + 1
        sub = box.column(align=True)
        for axis_id, label in _AXIS_LABEL.items():
            sub.label(text=f"{label}: {counts[axis_id]} line(s)")
        box.operator("img2cube.clear_axis", text="Clear active axis",
                     icon='TRASH')

        # ============================================================== #
        # 3. SOLVE CAMERA + SET ORIGIN
        # ============================================================== #
        box = layout.box()
        box.label(text="3. Solve & origin", icon='OUTLINER_OB_CAMERA')

        # 3a Solve
        box.operator("img2cube.solve", icon='PLAY')
        if props.solved:
            col = box.column(align=True)
            col.label(text=f"VPs: {props.solved_n_vps}    "
                          f"Focal: {props.solved_focal_px:.0f} px")
            col.label(text=f"Residual: {props.solve_residual_px:.2f} px",
                      icon=('CHECKMARK' if props.solve_residual_px < 2.0
                            else 'ERROR'))
            if props.solve_residual_px < 1.0:
                label = "Excellent precision."
            elif props.solve_residual_px < 3.0:
                label = "Good precision."
            elif props.solve_residual_px < 8.0:
                label = "Rough precision. Try adjusting lines to better match perspective."
            else:
                label = "Poor precision. Redraw lines and ensure they are parallel in 3D."
            col.label(text=label)
            col.label(text="Note: Fewer accurate lines are better than many rough lines.", icon='INFO')
            col.label(text="Extra lines only serve to create snap points.", icon='INFO')

        # 3b Origin
        if props.solved:
            row = box.row()
            row.operator("img2cube.set_origin",
                         text=("Cancel origin pick"
                               if props.is_picking_origin
                               else "Establish Origin (0,0,0)"),
                         icon='PIVOT_CURSOR',
                         depress=props.is_picking_origin)
            if props.origin_set:
                a_idx = props.origin_seg_a
                b_idx = props.origin_seg_b
                if (0 <= a_idx < len(props.segments)
                        and 0 <= b_idx < len(props.segments)):
                    sa = props.segments[a_idx]
                    sb = props.segments[b_idx]
                    box.label(
                        text=f"Origin: [{_AXIS_LABEL[sa.axis]}]#{a_idx} "
                             f"∩ [{_AXIS_LABEL[sb.axis]}]#{b_idx}",
                        icon='CHECKMARK')
            box.label(text="Note: Origin must be a real corner where two lines", icon='INFO')
            box.label(text="      meet in 3D (not just cross on the photo).")

        # ============================================================== #
        # 4. SCALE
        # ============================================================== #
        if props.solved:
            box = layout.box()
            box.label(text="4. Scale (measure lines)", icon='FIXED_SIZE')
            row = box.row()
            row.operator("img2cube.pick_segment",
                         text=("Cancel pick" if props.is_selecting
                               else "Pick a line in image"),
                         icon='RESTRICT_SELECT_OFF',
                         depress=props.is_selecting)

            sel = props.selected_segment

            # full segment list with click-to-select buttons
            if len(props.segments) > 0:
                inner = box.box()
                inner.label(text="Drawn & Measured Lines:", icon='SHORTDISPLAY')
                for i, s in enumerate(props.segments):
                    row = inner.row(align=True)
                    marker = "● " if i == sel else "  "
                    txt = f"{marker}[{_AXIS_LABEL[s.axis]}]#{i}"
                    
                    op = row.operator("img2cube.select_segment",
                                      text=txt, icon='RESTRICT_SELECT_OFF', depress=(i == sel))
                    op.index = i
                    
                    if s.known_length > 0:
                        btn_txt = f"{s.known_length:.3f}"
                    else:
                        btn_txt = "set length"
                    # Glow the "set length" button on the selected line so the
                    # user knows exactly where to type the size next.
                    op2 = row.operator("img2cube.set_segment_length",
                                       text=btn_txt,
                                       icon='GREASEPENCIL',
                                       depress=(i == sel))
                    op2.index = i
                    op3 = row.operator("img2cube.remove_segment", text="",
                                       icon='X')
                    op3.index = i

            # Scene scale info
            row = box.row()
            row.label(text=f"Scene scale: {props.scale_factor:.4f}  "
                          f"(avg of {props.n_measured} measure(s))")

            # Toggle button (only after at least 1 reference dimension is set)
            if props.n_measured > 0:
                row = box.row()
                row.prop(props, "show_dimensions", text="Show Estimated Dimensions", toggle=True, icon='VIS_SEL_11')

        # ============================================================== #
        # 5. ADD SNAP POINTS
        # ============================================================== #
        if props.origin_set:
            box = layout.box()
            box.label(text="5. Add snap points", icon='SNAP_VERTEX')
            box.label(text="Note: Click on any drawn line or its extension (thin guide).", icon='INFO')
            row = box.row()
            row.operator("img2cube.add_snap_points",
                         text=("Stop adding"
                               if props.is_adding_snaps
                               else "Add Snap Points along Lines"),
                         icon='PAUSE' if props.is_adding_snaps
                              else 'OUTLINER_OB_EMPTY',
                         depress=props.is_adding_snaps)
            box.operator("img2cube.clear_snap_points",
                         text="Clear all snap points", icon='TRASH')


CLASSES = (IMG2CUBE_PT_main,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
