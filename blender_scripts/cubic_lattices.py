bl_info = {
    "name": "Cubic Lattices (SC / BCC / FCC) for medial-ax",
    "author": "medial-ax",
    "version": (1, 3, 1),
    "blender": (3, 3, 0),
    "location": "View3D > Add > Mesh > Cubic Lattice   |   Sidebar (N) > Lattice",
    "description": "Generate simple-cubic, body-centred and face-centred cubic lattices "
                   "together with their Voronoi tessellation, optionally fitted to an "
                   "object and culled to its inside, and export lattice + Voronoi dual "
                   "as one .obj for mars-cli.",
    "category": "Add Mesh",
}

# This add-on is two things: a general lattice visualiser (Delaunay mosaics, exploded cells,
# sphere instances, BCC + FCC side by side) and the way we produce grids for mars-cli.
#
# Workflow for mars-cli:
#   1. Select the input complex, then Add > Mesh > Cubic Lattice (or the N-panel "Lattice" tab).
#   2. Tick "Fit to object" (and "Cull to inside" for closed meshes), pick BCC/FCC/SC, adjust the
#      spacing.  The defaults are what mars-cli needs: Voronoi faces = All faces, Delaunay
#      faces = None, Bonds = Voronoi walls.  The other face modes are for looking, not exporting.
#   3. With the lattice selected, press "Export for mars (.obj)".  Both objects go into one file:
#      `<kind>_lattice` (points + edges) and `<kind>_voronoi` (faces).  mars-cli links every
#      Voronoi wall to the grid edge it bisects; run `mars-cli grid-check file.obj` to see how.
#      Keep both objects visible and without modifiers when exporting; the operator checks.

import itertools
import math

import bpy
import bmesh
import mathutils
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, StringProperty)
from mathutils.bvhtree import BVHTree


# =====================================================================
#  Pure geometry.  No Blender calls in this section, so it is testable
#  outside Blender and easy to reuse.
#
#  Two integer coordinate systems are used internally, both exact:
#
#    Y = 2 x fractional cell coordinates.  Every lattice point is an
#        integer triple:  SC all even; BCC all even or all odd;
#        FCC integer with even coordinate sum.
#    Z = 4 x fractional = 2Y.  Lattice points and Voronoi vertices are
#        both integer triples here, which lets the Voronoi walls be
#        welded by exact key lookup instead of distance tests.
#
#  Real coordinates are (a/2)*Y = (a/4)*Z, minus the centring offset.
# =====================================================================

BASIS_Y = {
    'SC':  ((0, 0, 0),),
    'BCC': ((0, 0, 0), (1, 1, 1)),
    'FCC': ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
}

SHELL_1 = {'SC': 1.0, 'BCC': math.sqrt(3.0) / 2.0, 'FCC': math.sqrt(0.5)}
SHELL_2 = {'SC': math.sqrt(2.0), 'BCC': 1.0, 'FCC': 1.0}

TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
OCT_FACES = tuple((i, j, k) for i in (0, 1) for j in (2, 3) for k in (4, 5))
CUBE_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def _orbit(v):
    """All coordinate permutations of v with all sign choices."""
    out = set()
    for p in itertools.permutations(v):
        for s in itertools.product((1, -1), repeat=3):
            out.add((s[0] * p[0], s[1] * p[1], s[2] * p[2]))
    return sorted(out)


# Vertices of the Voronoi (Wigner-Seitz) cell, in Z coordinates.
#   SC  -> cube,                 8 vertices, 6 facets
#   BCC -> truncated octahedron, 24 vertices, 14 facets
#   FCC -> rhombic dodecahedron, 14 vertices, 12 facets
WS_Z = {
    'SC': _orbit((2, 2, 2)),
    'BCC': _orbit((0, 1, 2)),
    'FCC': _orbit((2, 0, 0)) + _orbit((1, 1, 1)),
}

# Relevant vectors: the lattice vectors whose bisector planes carry a facet.
REL_Z = {
    'SC': _orbit((4, 0, 0)),
    'BCC': _orbit((2, 2, 2)) + _orbit((4, 0, 0)),
    'FCC': _orbit((2, 2, 0)),
}


def _y_points(kind, nx, ny, nz):
    """Lattice points of the closed nx*ny*nz block, in Y coordinates."""
    lim = (2 * nx, 2 * ny, 2 * nz)
    pts = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                for off in BASIS_Y[kind]:
                    p = (2 * i + off[0], 2 * j + off[1], 2 * k + off[2])
                    if p[0] <= lim[0] and p[1] <= lim[1] and p[2] <= lim[2]:
                        pts.append(p)
    return pts


def _offset(nx, ny, nz, a, centred):
    return (nx * a / 2.0, ny * a / 2.0, nz * a / 2.0) if centred else (0.0, 0.0, 0.0)


def lattice_points(kind, nx, ny, nz, a=1.0, centred=True):
    """Real-space points of a rectangular block of nx*ny*nz conventional cells.

    The block is closed: points on the far faces are included, so a single
    BCC cell gives 9 points (8 corners + centre) and a single FCC cell
    gives 14 (8 corners + 6 face centres).
    """
    s, o = a / 2.0, _offset(nx, ny, nz, a, centred)
    return [(s * p[0] - o[0], s * p[1] - o[1], s * p[2] - o[2])
            for p in _y_points(kind, nx, ny, nz)]


def neighbour_pairs(points, distance, tol=1e-4):
    """Index pairs (i, j), i < j, with |p_i - p_j| == distance.

    Uses a uniform spatial hash so the cost stays linear in the number of
    points; a KD-tree would do as well but this keeps the module free of
    Blender's mathutils.
    """
    if distance <= 0.0:
        return []
    cell = distance * (1.0 + tol)
    grid = {}
    for idx, p in enumerate(points):
        key = (int(math.floor(p[0] / cell)),
               int(math.floor(p[1] / cell)),
               int(math.floor(p[2] / cell)))
        grid.setdefault(key, []).append(idx)

    lo, hi = (distance - tol) ** 2, (distance + tol) ** 2
    pairs = []
    for key, bucket in grid.items():
        neigh = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    neigh.extend(grid.get((key[0] + di, key[1] + dj, key[2] + dk), ()))
        for i in bucket:
            pi = points[i]
            for j in neigh:
                if j <= i:
                    continue
                pj = points[j]
                d2 = ((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2)
                if lo <= d2 <= hi:
                    pairs.append((i, j))
    pairs.sort()
    return pairs


# ------------------------------------------------------------- Delaunay
def delaunay_cells(kind, nx, ny, nz):
    """Maximal cells of the Delaunay mosaic of the block.

    Returns a list of (vertex_indices, local_faces), indices referring to
    the order produced by lattice_points / _y_points.  Cells that stick
    out of the block are dropped.

        SC  : cubes (the mosaic is not simplicial: 8 cospherical points)
        BCC : 6 tetrahedra per lattice point, the affine image of the
              Freudenthal triangulation of the cube
        FCC : tetrahedra and octahedra, the tetrahedral-octahedral
              honeycomb; 2 tetrahedra + 1 octahedron per lattice point
    """
    pts = _y_points(kind, nx, ny, nz)
    index = {p: i for i, p in enumerate(pts)}
    lim = (2 * nx, 2 * ny, 2 * nz)
    cells = []

    def add(cell, faces):
        idx = [index.get(v) for v in cell]
        if None not in idx:
            cells.append((tuple(idx), faces))

    if kind == 'SC':
        for p in pts:
            add([(p[0] + 2 * dx, p[1] + 2 * dy, p[2] + 2 * dz)
                 for dx, dy, dz in itertools.product((0, 1), repeat=3)], CUBE_FACES)

    elif kind == 'BCC':
        b = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))     # primitive basis in Y units
        for p in pts:
            for perm in itertools.permutations(range(3)):
                v, acc = [p], p
                for t in perm[:2]:
                    acc = (acc[0] + b[t][0], acc[1] + b[t][1], acc[2] + b[t][2])
                    v.append(acc)
                v.append((p[0] + 1, p[1] + 1, p[2] + 1))   # shared space diagonal
                add(v, TET_FACES)

    elif kind == 'FCC':
        # tetrahedra: the four even-sum corners of every unit cube in Y space
        for i in range(lim[0]):
            for j in range(lim[1]):
                for k in range(lim[2]):
                    corners = [(i + dx, j + dy, k + dz)
                               for dx, dy, dz in itertools.product((0, 1), repeat=3)]
                    add([c for c in corners if sum(c) % 2 == 0], TET_FACES)
        # octahedra: the six neighbours of every octahedral hole (odd-sum point)
        for i in range(lim[0] + 1):
            for j in range(lim[1] + 1):
                for k in range(lim[2] + 1):
                    if (i + j + k) % 2 == 0:
                        continue
                    add([(i + 1, j, k), (i - 1, j, k), (i, j + 1, k),
                         (i, j - 1, k), (i, j, k + 1), (i, j, k - 1)], OCT_FACES)
    else:
        raise ValueError(kind)

    return cells


# -------------------------------------------------------------- Voronoi
def wigner_seitz_facets(kind):
    """Voronoi cell of the lattice as (vertices_in_Z, [(relevant_vector, facet), ...]).

    Facets are found exactly: a vertex v lies on the wall of the relevant
    vector r when 2 v.r == r.r, in integer arithmetic.  The vertices of a
    wall are then sorted by angle around r and wound so the normal points
    outwards.  The relevant vector is (in Z units) the offset to the
    neighbouring lattice point that shares the wall.
    """
    verts = WS_Z[kind]
    facets = []
    for r in REL_Z[kind]:
        rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
        on = [i for i, v in enumerate(verts)
              if 2 * (v[0] * r[0] + v[1] * r[1] + v[2] * r[2]) == rr]
        if len(on) < 3:
            continue                      # 'lax' relevant vector: not a facet
        c = _centroid([verts[i] for i in on])
        n = _unit(r)
        u = _unit(_sub(verts[on[0]], c))
        w = _cross(n, u)
        on.sort(key=lambda i: math.atan2(_dot(_sub(verts[i], c), w),
                                         _dot(_sub(verts[i], c), u)))
        facets.append((r, _orient(tuple(on), verts, _centroid(verts))))
    return verts, facets


def wigner_seitz_polytope(kind):
    """Voronoi cell of the lattice as (vertices_in_Z, ordered facet polygons)."""
    verts, facets = wigner_seitz_facets(kind)
    return verts, [f for _, f in facets]


def voronoi_walls(kind, nx, ny, nz, a=1.0, centred=True, keep=None, outer=True):
    """Voronoi walls of the block, each wall once, with the pair of sites it separates.

    Returns (verts, faces, pairs).  `faces` index into `verts`; pairs[k] is
    (i, j) with i, j indices into lattice_points(...), j being None for an
    outer wall, i.e. a wall whose second site is not in the block (or was
    culled).  `keep` optionally restricts to a set of lattice point indices,
    `outer` says whether to emit outer walls at all.  Walls shared by two
    cells use the same welded vertices.
    """
    pts = _y_points(kind, nx, ny, nz)
    index = {p: i for i, p in enumerate(pts) if keep is None or i in keep}
    off_z, facets = wigner_seitz_facets(kind)
    s, o = a / 4.0, _offset(nx, ny, nz, a, centred)
    verts, key, faces, pairs = [], {}, [], []
    for p, i in index.items():
        pz = (2 * p[0], 2 * p[1], 2 * p[2])
        for r, face in facets:
            j = index.get((p[0] + r[0] // 2, p[1] + r[1] // 2, p[2] + r[2] // 2))
            if j is None and not outer:
                continue
            if j is not None and j < i:
                continue                  # emitted from the other side
            ids = []
            for vi in face:
                d = off_z[vi]
                k = (pz[0] + d[0], pz[1] + d[1], pz[2] + d[2])
                vid = key.get(k)
                if vid is None:
                    vid = key[k] = len(verts)
                    verts.append((s * k[0] - o[0], s * k[1] - o[1], s * k[2] - o[2]))
                ids.append(vid)
            faces.append(tuple(ids))
            pairs.append((i, j))
    return verts, faces, pairs


def voronoi_cells(kind, nx, ny, nz, a=1.0, centred=True):
    """Voronoi tessellation of the block: one Wigner-Seitz cell per point.

    Returns (points, cells) in the same shape as delaunay_cells, so the
    same face and solid builders apply.  Walls shared by two cells use
    the same welded vertices.
    """
    off_z, faces = wigner_seitz_polytope(kind)
    s, o = a / 4.0, _offset(nx, ny, nz, a, centred)
    verts, key, cells = [], {}, []
    for p in _y_points(kind, nx, ny, nz):
        pz = (2 * p[0], 2 * p[1], 2 * p[2])
        idx = []
        for d in off_z:
            k = (pz[0] + d[0], pz[1] + d[1], pz[2] + d[2])
            i = key.get(k)
            if i is None:
                i = key[k] = len(verts)
                verts.append((s * k[0] - o[0], s * k[1] - o[1], s * k[2] - o[2]))
            idx.append(i)
        cells.append((tuple(idx), faces))
    return verts, cells


# ------------------------------------------------- shared face builders
def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _dot(p, q):
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


def _cross(p, q):
    return (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0])


def _unit(p):
    n = math.sqrt(_dot(p, p))
    return (p[0] / n, p[1] / n, p[2] / n)


def _centroid(vs):
    n = float(len(vs))
    return (sum(v[0] for v in vs) / n, sum(v[1] for v in vs) / n,
            sum(v[2] for v in vs) / n)


def _orient(face, coords, centre):
    """Reverse `face` if its normal points towards `centre`."""
    a, b, c = coords[face[0]], coords[face[1]], coords[face[2]]
    n = _cross(_sub(b, a), _sub(c, a))
    return tuple(face) if _dot(n, _sub(a, centre)) >= 0.0 else tuple(reversed(face))


def mosaic_faces(cells, points, boundary_only=False):
    """Faces of a cell complex, indexing into `points`.

    boundary_only keeps the faces used by a single cell, i.e. the outer
    shell; otherwise the whole 2-skeleton is returned, interior walls
    included but never duplicated.
    """
    count, first = {}, {}
    for verts, local in cells:
        centre = _centroid([points[i] for i in verts])
        for f in local:
            face = tuple(verts[i] for i in f)
            k = tuple(sorted(face))
            count[k] = count.get(k, 0) + 1
            if k not in first:
                first[k] = _orient(face, points, centre)
    if boundary_only:
        return [first[k] for k, n in count.items() if n == 1]
    return [first[k] for k in count]


def separate_cells(cells, points, shrink=0.0):
    """Cells as individual closed solids (loose parts).

    Vertices are duplicated per cell and pulled `shrink` of the way
    towards the cell centroid, which gives an exploded view.
    """
    verts, faces = [], []
    for vidx, local in cells:
        base = len(verts)
        vs = [points[i] for i in vidx]
        c = _centroid(vs)
        for v in vs:
            verts.append((v[0] + (c[0] - v[0]) * shrink,
                          v[1] + (c[1] - v[1]) * shrink,
                          v[2] + (c[2] - v[2]) * shrink))
        for f in local:
            faces.append(tuple(base + i for i in _orient(f, vs, c)))
    return verts, faces


def build_faces(cells, points, mode, shrink=0.0):
    """Dispatch a face mode to the right builder -> (verts, faces)."""
    if mode == 'CELLS':
        return separate_cells(cells, points, shrink)
    return points, mosaic_faces(cells, points, boundary_only=(mode == 'BOUNDARY'))


# =====================================================================
#  Blender side
# =====================================================================

def _world_bbox(ob, depsgraph=None):
    """Axis-aligned bounding box of an object in world space, modifiers applied: (min, max)."""
    if depsgraph is not None:
        ob = ob.evaluated_get(depsgraph)
    corners = [ob.matrix_world @ mathutils.Vector(c) for c in ob.bound_box]
    lo = tuple(min(c[k] for c in corners) for k in range(3))
    hi = tuple(max(c[k] for c in corners) for k in range(3))
    return lo, hi


def average_longest_edge_length(ob, depsgraph=None):
    """Mean over the (triangulated) faces of `ob` of their longest edge, in world units.

    Same heuristic as adaptive_grid.py: a density of 2 means two grid edges
    per average triangle edge.  Quads and n-gons are triangulated first, so
    any mesh yields a spacing.
    """
    bm = bmesh.new()
    if depsgraph is not None:
        bm.from_object(ob, depsgraph)
    else:
        bm.from_mesh(ob.data)
    bm.transform(ob.matrix_world)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    longest = [max(e.calc_length() for e in f.edges) for f in bm.faces]
    bm.free()
    return sum(longest) / len(longest) if longest else 0.0


class InsideTester:
    """Point-in-closed-mesh test against the evaluated geometry of `ob`, in world space.

    Rays are cast from the point along the three axes and their crossings
    counted; the point is inside if at least two of the three parities are
    odd, so one bad triangle cannot flip the verdict.  Points within a small
    distance of the surface count as inside, so a lattice point lying exactly
    on a face is treated the same on every side of the object.  The BVH tree
    is built once; the step past a hit is relative to the object's size.
    """

    def __init__(self, ob, depsgraph):
        self.inv = ob.matrix_world.inverted()
        self.tree = BVHTree.FromObject(ob, depsgraph)
        corners = [mathutils.Vector(c) for c in ob.evaluated_get(depsgraph).bound_box]
        diagonal = max((p - q).length for p in corners for q in corners)
        self.eps = 1e-6 * max(diagonal, 1e-12)
        self.directions = [(self.inv.to_3x3() @ mathutils.Vector(axis)).normalized()
                           for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]

    def __call__(self, point):
        origin = self.inv @ mathutils.Vector(point)
        _location, _normal, _index, distance = self.tree.find_nearest(origin)
        if distance is not None and distance <= self.eps:
            return True
        odd = 0
        for direction in self.directions:
            count, start = 0, origin.copy()
            while count < 100000:
                location, _normal, _index, _distance = self.tree.ray_cast(start, direction)
                if location is None:
                    break
                count += 1
                start = location + direction * self.eps
            odd += count % 2
        return odd >= 2


def _new_object(context, name, verts, edges=(), faces=(), location=(0.0, 0.0, 0.0)):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(list(verts), list(edges), list(faces))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    context.collection.objects.link(obj)
    return obj


def _sphere_instances(context, parent, radius, subdivisions=2):
    """Small icosphere parented to `parent`, instanced on its vertices."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius)
    mesh = bpy.data.meshes.new(parent.name + "_atom")
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    child = bpy.data.objects.new(parent.name + "_atom", mesh)
    context.collection.objects.link(child)
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    parent.instance_type = 'VERTS'
    return child


def build_lattice(context, kind, a, nx, ny, nz, bonds, second_shell,
                  delaunay_mode, voronoi_mode, cell_shrink,
                  spheres, sphere_scale, origin,
                  cull_object=None, outer_walls=True, delaunay_bonds=False,
                  depsgraph=None):
    """Create the lattice object, plus a Voronoi object if asked for.

    With `cull_object`, lattice points outside that (closed) mesh are
    dropped, together with every edge, Delaunay cell and Voronoi wall that
    needs them.  With `delaunay_bonds`, the edges are exactly the pairs of
    points that share a Voronoi wall, read off the wall construction (no
    distance tolerance involved); `bonds`/`second_shell` are then ignored.
    Returns (object or None, counts).
    """
    all_pts = lattice_points(kind, nx, ny, nz, a)
    if cull_object is not None:
        inside = InsideTester(cull_object, depsgraph or context.evaluated_depsgraph_get())
        keep = {i for i, p in enumerate(all_pts)
                if inside((p[0] + origin[0], p[1] + origin[1], p[2] + origin[2]))}
    else:
        keep = set(range(len(all_pts)))
    kept = sorted(keep)
    old_to_new = {i: n for n, i in enumerate(kept)}
    pts = [all_pts[i] for i in kept]
    counts = [len(pts), 0, 0, 0, len(all_pts) - len(pts)]
    if not pts:
        return None, counts

    def remap_cells(cells):
        out = []
        for vidx, local in cells:
            if all(i in old_to_new for i in vidx):
                out.append((tuple(old_to_new[i] for i in vidx), local))
        return out

    verts, edges, faces = pts, [], []

    walls = None
    if voronoi_mode == 'MOSAIC' or delaunay_bonds:
        # every wall once, welded, with the pair of sites it separates
        walls = voronoi_walls(kind, nx, ny, nz, a, keep=keep, outer=outer_walls)

    if delaunay_mode != 'NONE':
        verts, faces = build_faces(remap_cells(delaunay_cells(kind, nx, ny, nz)), pts,
                                   delaunay_mode, cell_shrink)

    if delaunay_mode != 'CELLS':
        if delaunay_bonds:
            # exactly the pairs that share a wall: the Delaunay edges
            edges += sorted((old_to_new[i], old_to_new[j])
                            for i, j in walls[2] if j is not None)
        elif bonds:
            edges += neighbour_pairs(pts, SHELL_1[kind] * a, tol=1e-4 * a)
            if second_shell:
                edges += neighbour_pairs(pts, SHELL_2[kind] * a, tol=1e-4 * a)
        if faces and edges:             # drop edges the faces already create
            used = set()
            for f in faces:
                for m in range(len(f)):
                    used.add(tuple(sorted((f[m], f[(m + 1) % len(f)]))))
            edges = [e for e in edges if e not in used]

    obj = _new_object(context, "%s_lattice" % kind, verts, edges, faces, location=origin)
    counts[1], counts[2] = len(edges), len(faces)

    if spheres and delaunay_mode != 'CELLS':
        _sphere_instances(context, obj, radius=sphere_scale * SHELL_1[kind] * a / 2.0)

    if voronoi_mode != 'NONE':
        if voronoi_mode == 'SINGLE':
            # one Wigner-Seitz cell, on the lattice point nearest the centre
            c = min(pts, key=lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
            vz, vf = wigner_seitz_polytope(kind)
            vv = [(c[0] + a * v[0] / 4.0, c[1] + a * v[1] / 4.0, c[2] + a * v[2] / 4.0)
                  for v in vz]
            vfaces = vf
        elif voronoi_mode == 'MOSAIC':
            vv, vfaces, _pairs = walls
        else:
            vpts, vcells = voronoi_cells(kind, nx, ny, nz, a)
            vcells = [cell for i, cell in enumerate(vcells) if i in keep]
            vv, vfaces = build_faces(vcells, vpts, voronoi_mode, cell_shrink)
        # The Voronoi object is a child of the lattice object and inherits its transform, so
        # it is created at the parent's local origin (not at `origin` again: the parent's
        # matrix_world is not evaluated yet at this point, so a parent-inverse would be stale
        # and the child would end up translated twice).
        vobj = _new_object(context, "%s_voronoi" % kind, vv, (), vfaces)
        if voronoi_mode == 'SINGLE':
            vobj.display_type = 'WIRE'
        vobj.parent = obj
        counts[3] = len(vfaces)

    return obj, counts


FACE_MODES = (
    ('NONE', "None", "Do not generate these faces"),
    ('MOSAIC', "All faces", "Every face, interior walls included, welded"),
    ('BOUNDARY', "Boundary only",
     "Only the faces used by a single cell: a closed shell around the block"),
    ('CELLS', "Separate cells", "Each cell as its own closed solid (loose parts)"),
)


class MESH_OT_add_cubic_lattice(bpy.types.Operator):
    """Add a body-centred and/or face-centred cubic lattice"""
    bl_idname = "mesh.add_cubic_lattice"
    bl_label = "Cubic Lattice"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    lattice_type: EnumProperty(
        name="Lattice",
        items=(('BCC', "BCC", "Body-centred cubic"),
               ('FCC', "FCC", "Face-centred cubic"),
               ('SC', "Simple cubic", "Simple cubic"),
               ('BOTH', "BCC + FCC", "Both, as two separate objects")),
        default='BCC',
    )
    side: FloatProperty(
        name="Side length a", description="Edge of the conventional cubic cell",
        default=1.0, min=1e-4, soft_max=10.0, unit='LENGTH',
    )
    nx: IntProperty(name="Cells X", default=3, min=1, soft_max=30)
    ny: IntProperty(name="Cells Y", default=3, min=1, soft_max=30)
    nz: IntProperty(name="Cells Z", default=3, min=1, soft_max=30)

    fit_to_object: BoolProperty(
        name="Fit to object",
        description="Size and place the block around the target object's bounding box "
                    "(plus padding) instead of using the cell counts and location",
        default=False,
    )
    target_object: StringProperty(
        name="Target object",
        description="Mesh object to fit the block to and, optionally, to cull the lattice to",
    )
    spacing_mode: EnumProperty(
        name="Spacing",
        items=(('SIDE', "Side length a", "Use the side length"),
               ('DENSITY', "Density",
                "Choose a so that the nearest-neighbour distance is the target's mean "
                "longest triangle edge divided by the density")),
        default='SIDE',
    )
    density: FloatProperty(
        name="Density", description="Grid edges per average triangle edge length",
        default=2.0, min=0.1, soft_max=10.0,
    )
    padding: FloatProperty(
        name="Padding",
        description="Extra room around the bounding box; 0 means half the "
                    "nearest-neighbour distance",
        default=0.0, min=0.0, unit='LENGTH',
    )
    cull_outside: BoolProperty(
        name="Cull to inside",
        description="Drop lattice points outside the target object (closed meshes only), "
                    "with their edges and Voronoi walls",
        default=False,
    )

    delaunay_bonds: BoolProperty(
        name="Bonds = Voronoi walls",
        description="Connect exactly the pairs of points that share a Voronoi wall (the "
                    "Delaunay edges): the first shell, plus the second shell for BCC. This is "
                    "what mars-cli needs; an edge without a wall is walked for nothing",
        default=True,
    )
    bonds: BoolProperty(
        name="Nearest-neighbour bonds",
        description="Edges to the first coordination shell "
                    "(8 per site for BCC, 12 for FCC, 6 for SC)",
        default=True,
    )
    second_shell: BoolProperty(
        name="Second shell (length a)",
        description="Also connect the 6 neighbours at distance a. Only for BCC do these "
                    "share a Voronoi wall (the 6 squares of the truncated octahedron)",
        default=False,
    )
    delaunay_faces: EnumProperty(
        name="Delaunay faces", items=FACE_MODES, default='NONE',
        description="Faces of the Delaunay mosaic: tetrahedra for BCC, "
                    "tetrahedra and octahedra for FCC, cubes for SC. "
                    "Written into the lattice object",
    )
    voronoi_faces: EnumProperty(
        name="Voronoi faces",
        items=(FACE_MODES[0],
               ('SINGLE', "Single cell",
                "One Wigner-Seitz cell at the centre of the block, as a wireframe"),
               ) + FACE_MODES[1:],
        default='MOSAIC',
        description="Faces of the Voronoi tessellation: truncated octahedra for "
                    "BCC, rhombic dodecahedra for FCC, cubes for SC. Written into "
                    "a separate <lattice>_voronoi object. mars-cli needs 'All faces'",
    )
    outer_walls: BoolProperty(
        name="Outer Voronoi walls",
        description="Also build the walls of the boundary cells that face away from the "
                    "block. mars-cli drops them, so this is only for looks",
        default=True,
    )
    cell_shrink: FloatProperty(
        name="Cell shrink",
        description="Pull each separate cell towards its centroid, for an "
                    "exploded view of the mosaic or of the foam",
        default=0.08, min=0.0, max=0.9,
    )
    spheres: BoolProperty(
        name="Sphere instances",
        description="Parent a small icosphere and instance it on the vertices",
        default=False,
    )
    sphere_scale: FloatProperty(
        name="Sphere scale", description="1.0 = touching spheres (close packing)",
        default=0.6, min=0.01, max=1.0,
    )
    dual_scaling: BoolProperty(
        name="Dual scaling for FCC",
        description="With BCC + FCC, build the FCC lattice at side 2/a instead "
                    "of a, so that it is the reciprocal (dual) lattice of the BCC one",
        default=False,
    )
    side_by_side: BoolProperty(
        name="Side by side",
        description="Offset the second lattice along X instead of overlapping them",
        default=True,
    )
    location: FloatVectorProperty(name="Location", subtype='TRANSLATION')

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "lattice_type")

        box = layout.box()
        box.prop(self, "fit_to_object")
        sub = box.column()
        sub.enabled = self.fit_to_object or self.cull_outside
        sub.prop_search(self, "target_object", bpy.data, "objects")
        if self.fit_to_object:
            box.prop(self, "spacing_mode")
            if self.spacing_mode == 'DENSITY':
                box.prop(self, "density")
            box.prop(self, "padding")
        box.prop(self, "cull_outside")

        sub = layout.column()
        sub.enabled = not (self.fit_to_object and self.spacing_mode == 'DENSITY')
        sub.prop(self, "side")
        col = layout.column(align=True)
        col.enabled = not self.fit_to_object
        col.prop(self, "nx"); col.prop(self, "ny"); col.prop(self, "nz")

        layout.separator()
        layout.prop(self, "delaunay_faces")
        layout.prop(self, "voronoi_faces")
        sub = layout.column()
        sub.enabled = self.voronoi_faces == 'MOSAIC'
        sub.prop(self, "outer_walls")
        sub = layout.column()
        sub.enabled = 'CELLS' in (self.delaunay_faces, self.voronoi_faces)
        sub.prop(self, "cell_shrink")

        layout.separator()
        col = layout.column()
        col.enabled = self.delaunay_faces != 'CELLS'
        col.prop(self, "delaunay_bonds")
        sub = col.column()
        sub.enabled = not self.delaunay_bonds
        sub.prop(self, "bonds")
        sub = sub.column()
        sub.enabled = self.bonds
        sub.prop(self, "second_shell")
        col.prop(self, "spheres")
        sub = col.column()
        sub.enabled = self.spheres
        sub.prop(self, "sphere_scale")

        if self.lattice_type == 'BOTH':
            layout.separator()
            layout.prop(self, "dual_scaling")
            layout.prop(self, "side_by_side")

    def invoke(self, context, event):
        self.location = context.scene.cursor.location
        ob = context.active_object
        if (ob is not None and ob.type == 'MESH' and not self.target_object
                and not ob.name.split(".")[0].endswith(("_lattice", "_voronoi"))):
            self.target_object = ob.name
        return self.execute(context)

    def execute(self, context):
        kinds = ['BCC', 'FCC'] if self.lattice_type == 'BOTH' else [self.lattice_type]
        target = bpy.data.objects.get(self.target_object) if self.target_object else None
        if (self.fit_to_object or self.cull_outside) and (target is None or target.type != 'MESH'):
            self.report({'ERROR'}, "Fit to object / Cull to inside need a target mesh object")
            return {'CANCELLED'}
        cull = target if self.cull_outside else None
        depsgraph = context.evaluated_depsgraph_get()

        base = tuple(self.location)
        created, report = [], []

        for n, kind in enumerate(kinds):
            a = self.side
            if kind == 'FCC' and self.lattice_type == 'BOTH' and self.dual_scaling:
                a = 2.0 / self.side
            nx, ny, nz = self.nx, self.ny, self.nz
            origin = list(base)
            if self.fit_to_object:
                if self.spacing_mode == 'DENSITY':
                    nn = average_longest_edge_length(target, depsgraph) / self.density
                    if nn <= 0.0:
                        self.report({'ERROR'}, "Target has no faces; cannot derive a spacing")
                        return {'CANCELLED'}
                    a = nn / SHELL_1[kind]
                lo, hi = _world_bbox(target, depsgraph)
                pad = self.padding if self.padding > 0.0 else 0.5 * SHELL_1[kind] * a
                nx, ny, nz = [max(1, int(math.ceil((hi[k] - lo[k] + 2.0 * pad) / a)))
                              for k in range(3)]
                origin = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
            elif n and self.side_by_side:
                origin[0] += self.nx * self.side + max(self.side, a)

            obj, c = build_lattice(
                context, kind, a, nx, ny, nz,
                self.bonds, self.second_shell, self.delaunay_faces,
                self.voronoi_faces, self.cell_shrink,
                self.spheres, self.sphere_scale, tuple(origin),
                cull_object=cull, outer_walls=self.outer_walls,
                delaunay_bonds=self.delaunay_bonds, depsgraph=depsgraph)
            if obj is None:
                self.report({'WARNING'}, "%s: no lattice point is inside %r" % (kind, target.name))
                continue
            created.append(obj)
            report.append("%s (a=%.4g, %dx%dx%d cells): %d points, %d edges, %d Delaunay faces, "
                          "%d Voronoi faces, %d points culled"
                          % (kind, a, nx, ny, nz, c[0], c[1], c[2], c[3], c[4]))

        if not created:
            return {'CANCELLED'}
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in created:
            obj.select_set(True)
        context.view_layer.objects.active = created[-1]

        self.report({'INFO'}, "   ".join(report))
        return {'FINISHED'}


class EXPORT_OT_lattice_for_mars(bpy.types.Operator):
    """Export the selected lattice and its Voronoi object as one .obj for mars-cli"""
    bl_idname = "export_mesh.lattice_for_mars"
    bl_label = "Export lattice for mars"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'})

    AXES = (('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", ""),
            ('NEGATIVE_X', "-X", ""), ('NEGATIVE_Y', "-Y", ""), ('NEGATIVE_Z', "-Z", ""))
    forward_axis: EnumProperty(
        name="Forward", items=AXES, default='X',
        description="Forward axis of the .obj file. Use the same axes as for the complex")
    up_axis: EnumProperty(
        name="Up", items=AXES, default='Z',
        description="Up axis of the .obj file. Use the same axes as for the complex")

    @staticmethod
    def _objects(context):
        """The meshes to export: the selected ones, their `_voronoi` children and `_lattice`
        parents, and the `<name>_voronoi` / `<name>_lattice` partner by name, in case the
        parenting was cleared."""
        objs = {o for o in context.selected_objects if o.type == 'MESH'}
        for o in list(objs):
            objs.update(c for c in o.children if c.type == 'MESH' and "_voronoi" in c.name)
            if o.parent is not None and o.parent.type == 'MESH':
                objs.add(o.parent)
            for a, b in (("_lattice", "_voronoi"), ("_voronoi", "_lattice")):
                if a in o.name:
                    partner = context.scene.objects.get(o.name.replace(a, b, 1))
                    if partner is not None and partner.type == 'MESH':
                        objs.add(partner)
        return objs

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "forward_axis")
        layout.prop(self, "up_axis")
        layout.label(text="Same axes as the complex", icon='INFO')

    def invoke(self, context, event):
        if not self.filepath:
            ob = context.active_object
            name = ob.name.split(".")[0].replace("_voronoi", "") if ob else "lattice"
            self.filepath = bpy.path.ensure_ext(name, ".obj")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        objs = self._objects(context)
        if not objs:
            self.report({'ERROR'}, "Select the lattice object first")
            return {'CANCELLED'}
        names = ", ".join(sorted(o.name for o in objs))
        if not any(len(o.data.polygons) for o in objs):
            self.report({'ERROR'},
                        "No Voronoi object among [%s]. Regenerate the lattice with Voronoi "
                        "faces = All faces (F9 panel), or select the <kind>_voronoi object "
                        "as well" % names)
            return {'CANCELLED'}
        for o in objs:
            if "_lattice" in o.name and len(o.data.polygons):
                self.report({'WARNING'},
                            "%s has faces; mars-cli rejects a lattice object with faces and "
                            "Blender drops the edges they cover. Regenerate with Delaunay "
                            "faces = None" % o.name)
            if "_lattice" in o.name and o.modifiers:
                self.report({'ERROR'},
                            "%s has modifiers (%s). They would be applied on export and turn "
                            "the grid into faces; disable or remove them first"
                            % (o.name, ", ".join(m.name for m in o.modifiers)))
                return {'CANCELLED'}
        context.view_layer.update()
        for o in objs:
            if o.parent in objs:
                drift = max(abs(x) for row in (o.matrix_world - o.parent.matrix_world)
                            for x in row)
                if drift > 1e-6:
                    self.report({'WARNING'},
                                "%s is not aligned with %s (moved on its own?); mars-cli "
                                "will drop the misaligned walls" % (o.name, o.parent.name))
        # Hidden objects cannot be selected and the exporter skips them, which would silently
        # leave the Voronoi object out of the file.  Unhide for the export, restore afterwards.
        def visibility(o):
            try:
                return o.hide_get(), o.hide_viewport
            except RuntimeError:                        # not in this view layer
                return None
        saved = {o: visibility(o) for o in objs}
        missing = sorted(o.name for o, v in saved.items() if v is None)
        if missing:
            self.report({'ERROR'}, "%s is not in the current view layer (excluded collection?); "
                                   "cannot export it" % ", ".join(missing))
            return {'CANCELLED'}
        try:
            for o in objs:
                o.hide_viewport = False
                o.hide_set(False)
            context.view_layer.update()
            for o in context.view_layer.objects:
                o.select_set(o in objs)
            unselected = sorted(o.name for o in objs if not o.select_get())
            if unselected:
                self.report({'ERROR'}, "Cannot select %s for export" % ", ".join(unselected))
                return {'CANCELLED'}

            path = bpy.path.ensure_ext(self.filepath, ".obj")
            # No normals, UVs or materials, and no triangulation: mars-cli needs the walls
            # whole.  The axes default to X forward / Z up, the way we export complexes; both
            # objects must be in the same frame as the complex.  Modifiers are applied, which
            # is why a lattice with modifiers was refused above.
            bpy.ops.wm.obj_export(
                filepath=path, export_selected_objects=True, apply_modifiers=True,
                export_normals=False, export_uv=False, export_materials=False,
                export_triangulated_mesh=False,
                forward_axis=self.forward_axis, up_axis=self.up_axis)
        finally:
            for o, (hidden, hide_viewport) in saved.items():
                o.hide_viewport = hide_viewport
                o.hide_set(hidden)

        n_faces = sum(len(o.data.polygons) for o in objs)
        self.report({'INFO'}, "Wrote %s (%s forward, %s up): %s, %d Voronoi faces. Check it "
                              "with `mars-cli grid-check`"
                    % (path, self.forward_axis.replace("NEGATIVE_", "-"),
                       self.up_axis.replace("NEGATIVE_", "-"), names, n_faces))
        return {'FINISHED'}


class VIEW3D_PT_cubic_lattice(bpy.types.Panel):
    bl_label = "Cubic Lattices"
    bl_idname = "VIEW3D_PT_cubic_lattice"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lattice"

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        col = layout.column(align=True)
        for kind, text in (('BCC', "BCC"), ('FCC', "FCC"), ('SC', "Simple cubic"),
                           ('BOTH', "BCC + FCC")):
            op = col.operator(MESH_OT_add_cubic_lattice.bl_idname, text=text)
            op.lattice_type = kind
        layout.label(text="Adjust in the redo panel (F9)", icon='INFO')
        layout.separator()
        layout.operator(EXPORT_OT_lattice_for_mars.bl_idname,
                        text="Export for mars (.obj)", icon='EXPORT')


def menu_func(self, context):
    self.layout.operator(MESH_OT_add_cubic_lattice.bl_idname,
                         text="Cubic Lattice (SC/BCC/FCC)", icon='MESH_GRID')


classes = (MESH_OT_add_cubic_lattice, EXPORT_OT_lattice_for_mars, VIEW3D_PT_cubic_lattice)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
