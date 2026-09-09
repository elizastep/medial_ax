"""Build a lattice + Voronoi dual around a complex with Blender in the background.

Usage (macOS path shown; use `blender` on Linux/Windows):

    /Applications/Blender.app/Contents/MacOS/Blender -b --python blender_scripts/headless_lattice.py -- \
        blender_scripts/cubic_lattices.py examples/cylinder.obj out/cylinder_bcc.obj BCC side 0.15 [cull]

Arguments after `--`:
    1. path to cubic_lattices.py (the add-on)
    2. the complex .obj (imported with the axes below, so the export matches it)
    3. output .obj: one file with `<kind>_lattice` (points + edges) and `<kind>_voronoi` (walls)
    4. SC, BCC or FCC
    5. `side` or `density`, and 6. its value (side length a, or grid edges per average triangle edge)
    7. optional `cull` to drop lattice points outside the complex (closed meshes only)

The complex is imported and the lattice exported with X forward / Z up (FORWARD, UP below),
the way we export complexes from Blender, so the output is in the complex's coordinates.

Then: `mars-cli grid-check out/cylinder_bcc.obj`.
"""
import importlib.util
import sys

import bpy

FORWARD, UP = 'X', 'Z'

argv = sys.argv[sys.argv.index("--") + 1:]
if len(argv) < 6:
    raise SystemExit(__doc__)
addon, complex_path, out_path, kind, mode, value = argv[:6]
cull = len(argv) > 6 and argv[6] == "cull"

spec = importlib.util.spec_from_file_location("cubic_lattices", addon)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.register()

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=complex_path, forward_axis=FORWARD, up_axis=UP)
target = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = target
print("target:", target.name, "verts", len(target.data.vertices), "faces", len(target.data.polygons))

kwargs = dict(lattice_type=kind, fit_to_object=True, target_object=target.name,
              cull_outside=cull)
if mode == "side":
    kwargs.update(spacing_mode='SIDE', side=float(value))
else:
    kwargs.update(spacing_mode='DENSITY', density=float(value))
print("add_cubic_lattice:", bpy.ops.mesh.add_cubic_lattice(**kwargs))

lattice = bpy.data.objects["%s_lattice" % kind]
for o in bpy.context.view_layer.objects:
    o.select_set(o is lattice)
bpy.context.view_layer.objects.active = lattice
print("export:", bpy.ops.export_mesh.lattice_for_mars(filepath=out_path,
                                                        forward_axis=FORWARD, up_axis=UP))
