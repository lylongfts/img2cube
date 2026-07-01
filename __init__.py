# SPDX-License-Identifier: GPL-3.0-or-later
"""img2cube — photo camera-match from drawn lines, then 3D snap points (v1.0.6)."""

from . import properties, operators, ui, draw


def register():
    properties.register()
    operators.register()
    ui.register()


def unregister():
    # make sure the overlay handler is gone before classes unregister
    try:
        draw.disable()
    except Exception:
        pass
    ui.unregister()
    operators.unregister()
    properties.unregister()
