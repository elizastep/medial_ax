use std::collections::{HashMap, HashSet};

use rstar::{RTree, RTreeObject, AABB};
use tracing::{instrument, trace};

use crate::{
    complex::{Complex, Pos},
    reduce_from_scratch, vineyards_step, Reduction, Swaps,
};

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Index(pub [isize; 3]);

impl Index {
    /// Make a fake index. Used for [VineyardsGridMesh], where we don't have real [Index]es for the
    /// points, but to be API compatible with [VineyardsGrid] we pretend that we do.
    pub fn fake(n: isize) -> Self {
        Self([n, 0, 0])
    }

    /// Get the first component of the index.
    fn x(&self) -> isize {
        self.0[0]
    }

    fn y(&self) -> isize {
        self.0[1]
    }

    fn z(&self) -> isize {
        self.0[2]
    }
}

impl std::ops::Add<Index> for Index {
    type Output = Index;
    fn add(self, rhs: Index) -> Index {
        let mut arr = [0; 3];
        for i in 0..3 {
            arr[i] = self.0[i] + rhs.0[i];
        }
        Index(arr)
    }
}

impl std::ops::AddAssign<Index> for Index {
    fn add_assign(&mut self, rhs: Index) {
        for i in 0..3 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl std::fmt::Debug for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "[{}, {}, {}]",
            self.0[0], self.0[1], self.0[2]
        ))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VineyardsGrid {
    pub corner: Pos,
    pub size: f64,
    pub shape: Index,
    /// Should always be `"grid"`. Used for serialization stuff.
    pub r#type: String,
}

impl VineyardsGrid {
    pub fn new(corner: Pos, size: f64, shape: [isize; 3]) -> Self {
        VineyardsGrid {
            corner,
            size,
            shape: Index(shape),
            r#type: "grid".to_string(),
        }
    }

    /// Construct a new [Grid] around the given [Complex]. The grid cells are of
    /// size `size`, and `buffer` is the smallest distance from the grid
    /// boundary to the complex.  This is tight at the min corner of the grid,
    /// and is up to `buffer + size` as the max corner of the grid.
    pub fn around_complex(complex: &Complex, size: f64, buffer: f64) -> Self {
        let (xmin, ymin, zmin) = complex.simplices_per_dim[0].iter().fold(
            (f64::MAX, f64::MAX, f64::MAX),
            |acc, simplex| {
                let [x, y, z] = simplex.coords.unwrap().0;
                (acc.0.min(x), acc.1.min(y), acc.2.min(z))
            },
        );

        let (xmax, ymax, zmax) = complex.simplices_per_dim[0].iter().fold(
            (f64::MIN, f64::MIN, f64::MIN),
            |acc, simplex| {
                let [x, y, z] = simplex.coords.unwrap().0;
                (acc.0.max(x), acc.1.max(y), acc.2.max(z))
            },
        );

        let corner = Pos([xmin - buffer, ymin - buffer, zmin - buffer]);

        let shape = [
            ((xmax - xmin + 2.0 * buffer) / size).ceil() as isize,
            ((ymax - ymin + 2.0 * buffer) / size).ceil() as isize,
            ((zmax - zmin + 2.0 * buffer) / size).ceil() as isize,
        ];

        Self::new(corner, size, shape)
    }

    pub fn is_on_boundary(&self, i: Index) -> bool {
        for j in 0..3 {
            if i.0[j] == 0 || i.0[j] == self.shape.0[j] - 1 {
                return true;
            }
        }
        false
    }

    pub fn center_index(&self) -> Index {
        let mut arr = [0; 3];
        for j in 0..3 {
            arr[j] = (self.shape.0[j] - 1) / 2;
        }
        Index(arr)
    }

    pub fn closest_index_of(&self, p: Pos) -> Index {
        let mut arr = [0; 3];
        for j in 0..3 {
            arr[j] = ((p.0[j] - self.corner.0[j]) / self.size).round() as isize;
        }
        Index(arr)
    }

    /// Returns the coordinate of the lower corner of the cell.
    pub fn coordinate(&self, i: Index) -> Pos {
        let mut arr = [0.0; 3];
        for j in 0..3 {
            arr[j] = self.corner.0[j] + self.size * i.0[j] as f64;
        }
        Pos(arr)
    }

    pub fn volume(&self) -> isize {
        self.shape.0[0] * self.shape.0[1] * self.shape.0[2]
    }

    pub fn dual_quad_points(&self, a: Index, b: Index) -> [Pos; 4] {
        let pa = self.coordinate(a);
        let pb = self.coordinate(b);
        let middle = (pa + pb) / 2.0;

        let size = self.size / 2.0;

        if a.x() != b.x() {
            [
                Pos([middle.x(), middle.y() - size, middle.z() - size]), // ll
                Pos([middle.x(), middle.y() - size, middle.z() + size]), // lr
                Pos([middle.x(), middle.y() + size, middle.z() + size]), // ur
                Pos([middle.x(), middle.y() + size, middle.z() - size]), // ul
            ]
        } else if a.y() != b.y() {
            [
                Pos([middle.x() - size, middle.y(), middle.z() - size]),
                Pos([middle.x() - size, middle.y(), middle.z() + size]),
                Pos([middle.x() + size, middle.y(), middle.z() + size]),
                Pos([middle.x() + size, middle.y(), middle.z() - size]),
            ]
        } else if a.z() != b.z() {
            [
                Pos([middle.x() - size, middle.y() - size, middle.z()]),
                Pos([middle.x() - size, middle.y() + size, middle.z()]),
                Pos([middle.x() + size, middle.y() + size, middle.z()]),
                Pos([middle.x() + size, middle.y() - size, middle.z()]),
            ]
        } else {
            panic!("dual_quad_face: a == b");
        }
    }

    /// Splits the grid into two along the longest axis.
    /// The [Index] returned is the offset of the second [Grid] wrt. the first [Grid].
    pub fn split_with_overlap(&self) -> (Self, Self, Index) {
        let [w, h, d] = self.shape.0;
        if h <= w && d <= w {
            let wmin = w / 2;
            let wmax = w - wmin;
            (
                Self::new(self.corner, self.size, [wmin + 1, h, d]),
                Self::new(
                    self.coordinate(Index([wmin, 0, 0])),
                    self.size,
                    [wmax, h, d],
                ),
                Index([wmin, 0, 0]),
            )
        } else if w <= h && d <= h {
            let hmin = h / 2;
            let hmax = h - hmin;
            (
                Self::new(self.corner, self.size, [w, hmin + 1, d]),
                Self::new(
                    self.coordinate(Index([0, hmin, 0])),
                    self.size,
                    [w, hmax, d],
                ),
                Index([0, hmin, 0]),
            )
        } else {
            let dmin = d / 2;
            let dmax = d - dmin;
            (
                Self::new(self.corner, self.size, [w, h, dmin + 1]),
                Self::new(
                    self.coordinate(Index([0, 0, dmin])),
                    self.size,
                    [w, h, dmax],
                ),
                Index([0, 0, dmin]),
            )
        }
    }

    pub fn number_of_grid_edges(&self) -> isize {
        self.shape.0[0] * self.shape.0[1] * (self.shape.0[2] - 1)
            + self.shape.0[0] * (self.shape.0[1] - 1) * self.shape.0[2]
            + (self.shape.0[0] - 1) * self.shape.0[1] * self.shape.0[2]
    }

    /// Run vineyards across all edges of the grid.
    pub fn run_vineyards_in_grid<F: Fn(usize, usize)>(
        &self,
        complex: &Complex,
        i0: Index,
        state: Reduction,
        require_hom_birth_to_be_first: bool,
        on_visit: F,
    ) -> (HashMap<Index, Reduction>, Vec<(Index, Index, Swaps)>) {
        let mut hm = HashMap::new();
        let mut all_swaps = Vec::new();

        let num_grid_edges = self.number_of_grid_edges() as usize;
        let mut edge_i = 0;

        self.visit_edges(i0, |new_cell, old_cell| {
            edge_i += 1;
            on_visit(edge_i, num_grid_edges);

            if let Some(old_cell) = old_cell {
                let old_state = hm
                    .get(&old_cell)
                    .expect("prev_cell should have state in the map.");
                let p = self.coordinate(new_cell);
                let (new_state, swaps) =
                    vineyards_step(complex, old_state, p, require_hom_birth_to_be_first);
                all_swaps.push((old_cell, new_cell, swaps));
                hm.insert(new_cell, new_state);
            } else {
                hm.insert(new_cell, state.clone());
            }
        });
        (hm, all_swaps)
    }

    /// True if the index is contained in the grid.
    fn contains(&self, index: &Index) -> bool {
        let [x, y, z] = index.0;
        x >= 0
            && x < self.shape.0[0]
            && y >= 0
            && y < self.shape.0[1]
            && z >= 0
            && z < self.shape.0[2]
    }

    /// Iterate over the at most 6 neighbors of a cell.
    fn iter_neighbors(&self, index: &Index) -> impl Iterator<Item = Index> + '_ {
        let [x, y, z] = index.0;
        [
            Index([x + 1, y, z]),
            Index([x - 1, y, z]),
            Index([x, y + 1, z]),
            Index([x, y - 1, z]),
            Index([x, y, z + 1]),
            Index([x, y, z - 1]),
        ]
        .into_iter()
        .filter(|i| self.contains(i))
    }

    /// Visit each edge of the grid.
    ///
    /// The two endpoints of each edge is passed as parameters to `f`.  The first [Index] is a
    /// potentially new [Index] that we haven't visited before.  The second parameter is the
    /// [Index] of the vertex that we have visited before, unless it is the very first edge we
    /// visit.
    pub fn visit_edges<F: FnMut(Index, Option<Index>)>(&self, start: Index, mut f: F) {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start, None));

        let mut visited = std::collections::HashSet::new();

        while let Some((next, prev)) = queue.pop_front() {
            let was_visited = visited.contains(&next);
            visited.insert(next);

            f(next, prev);

            if !was_visited {
                for neighbor in self.iter_neighbors(&next) {
                    if !visited.contains(&neighbor) {
                        queue.push_back((neighbor, Some(next)));
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Bbox(Pos, Pos);

impl Bbox {
    fn span_x(&self) -> f64 {
        (self.1 - self.0).x()
    }
    fn span_y(&self) -> f64 {
        (self.1 - self.0).y()
    }
    fn span_z(&self) -> f64 {
        (self.1 - self.0).z()
    }

    fn mid_x(&self) -> f64 {
        self.0.x() + self.span_x() / 2.0
    }
    fn mid_y(&self) -> f64 {
        self.0.y() + self.span_y() / 2.0
    }
    fn mid_z(&self) -> f64 {
        self.0.z() + self.span_z() / 2.0
    }
}

/// A single triangle of a [DualMesh] face, in the form used by the intersection test.
#[derive(Clone, Debug)]
struct Tri {
    /// The [DualMesh] face this triangle was cut from.
    face: u32,
    /// First corner.
    a: Pos,
    /// Edge from `a` to the second corner.
    ab: Pos,
    /// Edge from `a` to the third corner.
    ac: Pos,
}

impl RTreeObject for Tri {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        let b = self.a + self.ab;
        let c = self.a + self.ac;
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for j in 0..3 {
            lo[j] = self.a.0[j].min(b.0[j]).min(c.0[j]);
            hi[j] = self.a.0[j].max(b.0[j]).max(c.0[j]);
        }
        AABB::from_corners(lo, hi)
    }
}

/// Barycentric slack in the segment/triangle test.  We assume a grid edge never passes very close
/// to the boundary of a dual face, so this only needs to be big enough that an edge crossing the
/// seam between two triangles of the same fan is caught by at least one of them.
const TRI_EPS: f64 = 1e-9;

impl Tri {
    /// Fan-triangulate every face of a dual mesh.  Valid because we require the faces to be
    /// convex.
    fn tree_for(points: &[Pos], faces: &[Vec<u32>]) -> RTree<Tri> {
        let mut tris = Vec::new();
        for (fi, face) in faces.iter().enumerate() {
            for k in 1..face.len().saturating_sub(1) {
                let a = points[face[0] as usize];
                let b = points[face[k] as usize];
                let c = points[face[k + 1] as usize];
                tris.push(Tri {
                    face: fi as u32,
                    a,
                    ab: b - a,
                    ac: c - a,
                });
            }
        }
        RTree::bulk_load(tris)
    }

    /// Möller-Trumbore, restricted to the segment `p + t * d` for `t` in
    /// `(0, 1)`, and two-sided, since the winding of the dual faces is
    /// arbitrary.
    fn is_crossed_by(&self, p: Pos, d: Pos) -> bool {
        let pvec = d.cross(&self.ac);
        let det = self.ab.dot(&pvec);
        let parallel = det.abs() < 1e-30;
        if parallel {
            return false;
        }
        let inv = 1.0 / det;

        let tvec = p - self.a;
        let u = tvec.dot(&pvec) * inv;
        if u < -TRI_EPS || 1.0 + TRI_EPS < u {
            return false;
        }

        let qvec = tvec.cross(&self.ab);
        let v = d.dot(&qvec) * inv;
        if v < -TRI_EPS || 1.0 + TRI_EPS < u + v {
            return false;
        }

        let t = self.ac.dot(&qvec) * inv;
        0.0 < t && t < 1.0
    }
}

/// The dual mesh of the input grid is used for more complex faces than regular
/// quads. We require faces of the dual to be convex.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DualMesh {
    pub points: Vec<Pos>,
    pub faces: Vec<Vec<u32>>,
    #[serde(skip)]
    tree: Option<RTree<Tri>>,
}

impl DualMesh {
    /// Read a dual mesh from the contents of an .obj file.
    pub fn read_from_obj_string(s: &str) -> Result<Self, String> {
        let mut points: Vec<Pos> = Vec::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();

        for (line_no, line) in s.lines().enumerate() {
            let mut tokens = line.trim().split_ascii_whitespace();
            let Some(kind) = tokens.next() else {
                continue;
            };

            let err = |what: &str| format!("dual mesh, line {}: {}", line_no + 1, what);

            match kind {
                "v" => {
                    let mut c = [0.0; 3];
                    for j in 0..3 {
                        c[j] = tokens
                            .next()
                            .ok_or_else(|| err("vertex needs three coordinates"))?
                            .parse::<f64>()
                            .map_err(|e| err(&e.to_string()))?;
                    }
                    points.push(Pos(c));
                }
                "f" => {
                    let mut face = Vec::new();
                    for tok in tokens {
                        // `f 1`, `f 1/2`, `f 1/2/3` and `f 1//3` all mean vertex 1 here.
                        let vs = tok.split('/').next().unwrap_or(tok);
                        let i = vs.parse::<isize>().map_err(|e| err(&e.to_string()))?;
                        // .obj indices are 1-based, and negative ones count back from the end.
                        let i = if i < 0 {
                            points.len() as isize + i
                        } else {
                            i - 1
                        };
                        if i < 0 || points.len() as isize <= i {
                            return Err(err(&format!("face index {} out of range", i + 1)));
                        }
                        face.push(i as u32);
                    }
                    if face.len() < 3 {
                        return Err(err("face needs at least three vertices"));
                    }
                    faces.push(face);
                }
                _ => continue,
            }
        }

        if faces.is_empty() {
            return Err("dual mesh has no faces".to_string());
        }

        Ok(Self {
            points,
            faces,
            tree: None,
        })
    }

    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    fn tree(&mut self) -> &RTree<Tri> {
        self.tree
            .get_or_insert_with(|| Tri::tree_for(&self.points, &self.faces))
    }

    /// The corners of the dual face crossed by the segment from `p` to `q`, wound so that the
    /// polygon normal points from `p` towards `q`.
    pub fn face_points_crossed_by(&mut self, p: Pos, q: Pos) -> Option<Vec<Pos>> {
        let d = q - p;
        let envelope = AABB::from_corners(
            [p.x().min(q.x()), p.y().min(q.y()), p.z().min(q.z())],
            [p.x().max(q.x()), p.y().max(q.y()), p.z().max(q.z())],
        );
        // We assume a grid edge crosses at most one dual face, so the first hit wins.
        let fi = self
            .tree()
            .locate_in_envelope_intersecting(envelope)
            .find(|t| t.is_crossed_by(p, d))?
            .face as usize;
        let mut pts: Vec<Pos> = self.faces[fi]
            .iter()
            .map(|i| self.points[*i as usize])
            .collect();

        // Newell's method for the normal of a polygon.
        let mut n = Pos([0.0; 3]);
        for k in 0..pts.len() {
            let a = pts[k];
            let b = pts[(k + 1) % pts.len()];
            n = n + Pos([
                (a.y() - b.y()) * (a.z() + b.z()),
                (a.z() - b.z()) * (a.x() + b.x()),
                (a.x() - b.x()) * (a.y() + b.y()),
            ]);
        }
        if n.dot(&(q - p)) < 0.0 {
            pts.reverse();
        }

        Some(pts)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VineyardsGridMesh {
    pub points: Vec<Pos>,
    /// Map from vertex index to the list of its neighbors.
    pub neighbors: Vec<Vec<isize>>,
    /// Should always be `"meshgrid"`. Used for serialization stuff.
    pub r#type: String,

    /// Grid distances along the three axes, assuming the mesh is a subset of a regular grid.
    #[serde(skip)]
    pub dim_dist: Option<(f64, f64, f64)>,

    #[serde(default)]
    pub dual: Option<DualMesh>,
}

impl VineyardsGridMesh {
    pub fn empty() -> Self {
        Self {
            points: Vec::new(),
            neighbors: Vec::new(),
            r#type: "meshgrid".to_string(),
            dim_dist: None,
            dual: None,
        }
    }

    pub fn coordinate(&self, index: Index) -> Pos {
        self.points[index.0[0] as usize]
    }

    pub fn dual_quad_points(&self, a: Index, b: Index) -> [Pos; 4] {
        let a = self.points[a.0[0] as usize];
        let b = self.points[b.0[0] as usize];
        let (dx, dy, dz) = self.dim_dist.unwrap_or_else(|| {
            let dist = a.dist(&b);
            (dist, dist, dist)
        });

        let [ax, ay, az] = a.0;
        let [bx, by, bz] = b.0;
        let middle = a + (b - a) / 2.0;
        let ret: [Pos; 4] = if (ax - bx).abs() > 1e-3 {
            let p = Pos([0.0, dy / 2.0, 0.0]);
            let q = Pos([0.0, 0.0, dz / 2.0]);
            [
                middle - p - q,
                middle - p + q,
                middle + p + q,
                middle + p - q,
            ]
        } else if (ay - by).abs() > 1e-3 {
            let p = Pos([dx / 2.0, 0.0, 0.0]);
            let q = Pos([0.0, 0.0, dz / 2.0]);
            [
                middle - p - q,
                middle - p + q,
                middle + p + q,
                middle + p - q,
            ]
        } else if (az - bz).abs() > 1e-3 {
            let p = Pos([dx / 2.0, 0.0, 0.0]);
            let q = Pos([0.0, dy / 2.0, 0.0]);
            [
                middle - p - q,
                middle - p + q,
                middle + p + q,
                middle + p - q,
            ]
        } else {
            panic!("bad points {:?} {:?}", a, b);
        };
        ret
    }

    /// The corners of the dual face between the two adjacent grid points `a` and `b`.
    ///
    /// If the grid has a real dual mesh attached, this is the polygon of that mesh crossed by the
    /// grid edge.  Otherwise we fall back to [Self::dual_quad_points].
    ///
    /// [None] means we had a dual mesh but no face of it was crossed by this edge, which happens
    /// for edges on the boundary of the region the dual mesh covers.
    pub fn dual_face_points(&mut self, a: Index, b: Index) -> Option<Vec<Pos>> {
        let p = self.coordinate(a);
        let q = self.coordinate(b);
        match self.dual {
            Some(ref mut dual) => dual.face_points_crossed_by(p, q),
            None => Some(self.dual_quad_points(a, b).to_vec()),
        }
    }

    /// Lower- and upper corner of the bounding box.
    pub fn bbox_without_singletons(&self) -> Bbox {
        let mut minx = f64::MAX;
        let mut miny = f64::MAX;
        let mut minz = f64::MAX;
        let mut maxx = f64::MIN;
        let mut maxy = f64::MIN;
        let mut maxz = f64::MIN;
        for (i, p) in self.points.iter().enumerate() {
            if self
                .neighbors
                .get(i)
                .map(|ref v| v.len() == 0)
                .unwrap_or(true)
            {
                continue;
            }
            let [x, y, z] = p.0;
            minx = minx.min(x);
            miny = miny.min(y);
            minz = minz.min(z);
            maxx = maxx.max(x);
            maxy = maxy.max(y);
            maxz = maxz.max(z);
        }

        Bbox(Pos([minx, miny, minz]), Pos([maxx, maxy, maxz]))
    }

    #[instrument(skip_all)]
    pub fn split_in_half(&self) -> (Self, Self) {
        let bbox = self.bbox_without_singletons();
        let (dim_i, lim) = {
            let dx = bbox.span_x();
            let dy = bbox.span_y();
            let dz = bbox.span_z();
            if dy <= dx && dz <= dx {
                (0, bbox.mid_x())
            } else if dx <= dy && dz <= dy {
                (1, bbox.mid_y())
            } else {
                (2, bbox.mid_z())
            }
        };

        let mut lower_edges: Vec<Vec<isize>> = vec![Vec::new(); self.neighbors.len()];
        let mut upper_edges: Vec<Vec<isize>> = vec![Vec::new(); self.neighbors.len()];

        for (v, neighbors) in self.neighbors.iter().enumerate() {
            let v_is_lower = self.points[v as usize].0[dim_i] <= lim;
            for &w in neighbors {
                let w_is_lower = self.points[w as usize].0[dim_i] <= lim;
                if v_is_lower || w_is_lower {
                    lower_edges[v].push(w);
                } else {
                    upper_edges[v].push(w);
                }
            }
        }

        trace!("split {} / {}", lower_edges.len(), upper_edges.len());

        (
            VineyardsGridMesh {
                points: self.points.clone(),
                neighbors: lower_edges,
                r#type: self.r#type.clone(),
                dim_dist: None,
                dual: None,
            },
            VineyardsGridMesh {
                points: self.points.clone(),
                neighbors: upper_edges,
                r#type: self.r#type.clone(),
                dim_dist: None,
                dual: None,
            },
        )
    }

    pub fn run_vineyards<F: Fn(usize, usize)>(
        &self,
        complex: &Complex,
        require_hom_birth_to_be_first: bool,
        record_progress: F,
    ) -> (HashMap<Index, Reduction>, Vec<(Index, Index, Swaps)>) {
        let mut reductions: HashMap<Index, Reduction> = HashMap::new();
        let mut all_swaps: Vec<(Index, Index, Swaps)> = Vec::new();

        if self.points.len() == 0 {
            return (HashMap::new(), Vec::new());
        }
        let mut seen_vx = HashSet::<isize>::new();

        // Find a component in the meshgrid that we haven't reached yet.
        // `i0` is any node in this component.
        while let Some(i0) = self
            .neighbors
            .iter()
            .enumerate()
            .filter(|(v, _)| !seen_vx.contains(&(*v as isize)))
            .find(|(_, n)| n.len() > 0)
            .map(|(v, _)| v as isize)
        {
            seen_vx.insert(i0);
            let i0 = Index::fake(i0 as isize);
            let reduction_at_0 = reduce_from_scratch(&complex, self.points[i0.x() as usize], false);
            reductions.insert(i0, reduction_at_0);

            let mut stack = self
                .neighbors
                .get(i0.x() as usize)
                .unwrap()
                .iter()
                .map(|n| (Index::fake(*n), i0))
                .collect::<Vec<_>>();

            let mut loop_i = 0;
            let num_edges = self
                .neighbors
                .iter()
                .map(|v| v.len() as usize)
                .sum::<usize>()
                / 2;

            while let Some((next, from)) = stack.pop() {
                seen_vx.insert(next.x());
                loop_i += 1;
                record_progress(loop_i, num_edges);

                let old_state = reductions.get(&from).expect("from should be in the map");
                let p = self.coordinate(next);
                let (new_state, swaps) =
                    vineyards_step(complex, old_state, p, require_hom_birth_to_be_first);
                all_swaps.push((from, next, swaps));

                if !reductions.contains_key(&next) {
                    reductions.insert(next, new_state);
                    for neighbor in self.neighbors.get(next.x() as usize).unwrap() {
                        if !seen_vx.contains(neighbor) {
                            stack.push((Index::fake(*neighbor), next));
                        }
                    }
                }
            }
        }

        (reductions, all_swaps)
    }

    pub fn run_vineyards_slim<
        F: Fn(usize, usize),
        G: FnMut(Index, Index, &Reduction, &Reduction, Swaps),
    >(
        &self,
        complex: &Complex,
        require_hom_birth_to_be_first: bool,
        record_progress: F,
        mut on_edge: G,
    ) {
        let mut reductions: HashMap<Index, Reduction> = HashMap::new();

        if self.points.len() == 0 {
            return;
        }
        let mut seen_vx = HashSet::<isize>::new();
        let mut visits_left = HashMap::<isize, usize>::new();
        for (k, ns) in self.neighbors.iter().enumerate() {
            visits_left.insert(k as isize, ns.len());
        }

        // Find a component in the meshgrid that we haven't reached yet.
        // `i0` is any node in this component.
        while let Some(i0) = self
            .neighbors
            .iter()
            .enumerate()
            .filter(|(v, _)| !seen_vx.contains(&(*v as isize)))
            .find(|(_, n)| n.len() > 0)
            .map(|(v, _)| v as isize)
        {
            seen_vx.insert(i0);
            let i0 = Index::fake(i0 as isize);
            let reduction_at_0 = reduce_from_scratch(&complex, self.points[i0.x() as usize], false);
            reductions.insert(i0, reduction_at_0);

            let mut stack = self
                .neighbors
                .get(i0.x() as usize)
                .unwrap()
                .iter()
                .map(|n| (Index::fake(*n), i0))
                .collect::<Vec<_>>();

            let mut loop_i = 0;
            let num_edges = self
                .neighbors
                .iter()
                .map(|v| v.len() as usize)
                .sum::<usize>()
                / 2;

            while stack.len() > 0 {
                let (next, from) = stack.remove(0);
                seen_vx.insert(next.x());
                loop_i += 1;
                record_progress(loop_i, num_edges);

                let old_state = reductions.get(&from).expect("from should be in the map");
                let p = self.coordinate(next);
                let (new_state, swaps) =
                    vineyards_step(complex, old_state, p, require_hom_birth_to_be_first);

                on_edge(from, next, old_state, &new_state, swaps);

                if !reductions.contains_key(&next) {
                    reductions.insert(next, new_state);
                    for neighbor in self.neighbors.get(next.x() as usize).unwrap() {
                        if !seen_vx.contains(neighbor) {
                            stack.push((Index::fake(*neighbor), next));
                        }
                    }
                }

                let from_counter = visits_left
                    .get_mut(&from.x())
                    .expect("All vertices should be in the map");
                *from_counter -= 1;
                if *from_counter == 0 {
                    reductions.remove(&from).expect("missing reduction in from");
                }

                let next_counter = visits_left
                    .get_mut(&next.x())
                    .expect("All vertices should be in the map");
                *next_counter -= 1;
                if *next_counter == 0 {
                    reductions.remove(&next).expect("missing reduction in next");
                }
            }
        }
    }

    pub fn read_from_obj_string(s: &str) -> Result<Self, String> {
        let mut points: Vec<Pos> = Vec::new();
        let mut edges: Vec<(isize, isize)> = Vec::new();

        for line in s.lines() {
            let line = line.trim();
            if line.starts_with("#")
                || line.starts_with("mtllib")
                || line.starts_with("o")
                || line.starts_with("s")
            {
                continue;
            } else if line.starts_with("v") {
                let groups = line.split_ascii_whitespace().collect::<Vec<_>>();
                let x = groups
                    .get(1)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<f64>().map_err(|e| e.to_string()))?;
                let y = groups
                    .get(2)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<f64>().map_err(|e| e.to_string()))?;
                let z = groups
                    .get(3)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<f64>().map_err(|e| e.to_string()))?;
                let coords = Pos([x, y, z]);
                points.push(coords);
            } else if line.starts_with("l") {
                let groups = line.split_ascii_whitespace().collect::<Vec<_>>();
                let from = groups
                    .get(1)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<isize>().map_err(|e| e.to_string()))?;
                let to = groups
                    .get(2)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<isize>().map_err(|e| e.to_string()))?;
                edges.push((from - 1, to - 1));
            }
        }

        // Check that no two vertices are actually the same vertex
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let p = points[i];
                let q = points[j];
                let dist = p.dist(&q);
                if dist < 1e-5 {
                    return Err(format!(
                        "Two grid vertices are too close together: {} and {}",
                        i, j
                    ));
                }
            }
        }

        let xs = edges
            .iter()
            .cloned()
            .find(|(i, j)| points[*i as usize].x() != points[*j as usize].x());
        let ys = edges
            .iter()
            .cloned()
            .find(|(i, j)| points[*i as usize].y() != points[*j as usize].y());
        let zs = edges
            .iter()
            .cloned()
            .find(|(i, j)| points[*i as usize].z() != points[*j as usize].z());

        let dim_dist = if let (Some((xi, xj)), Some((yi, yj)), Some((zi, zj))) = (xs, ys, zs) {
            Some((
                (points[xi as usize].dist(&points[xj as usize])),
                (points[yi as usize].dist(&points[yj as usize])),
                (points[zi as usize].dist(&points[zj as usize])),
            ))
        } else {
            None
        };

        let mut neighbors: Vec<Vec<isize>> = vec![Vec::new(); points.len()];
        for (from, to) in edges {
            neighbors[from as usize].push(to);
            neighbors[to as usize].push(from);
        }

        Ok(Self {
            points,
            neighbors,
            r#type: "meshgrid".to_string(),
            dim_dist,
            dual: None,
        })
    }

    pub fn recompute_dim_dist(&mut self) {
        if self.dim_dist.is_some() {
            return;
        }
        let xs = self
            .neighbors
            .iter()
            .enumerate()
            .flat_map(|(i, n)| {
                n.iter()
                    .find(|j| self.points[i].x() != self.points[**j as usize].x())
                    .map(|j| (i, *j as usize))
            })
            .next();
        let ys = self
            .neighbors
            .iter()
            .enumerate()
            .flat_map(|(i, n)| {
                n.iter()
                    .find(|j| self.points[i].y() != self.points[**j as usize].y())
                    .map(|j| (i, *j as usize))
            })
            .next();
        let zs = self
            .neighbors
            .iter()
            .enumerate()
            .flat_map(|(i, n)| {
                n.iter()
                    .find(|j| self.points[i].z() != self.points[**j as usize].z())
                    .map(|j| (i, *j as usize))
            })
            .next();

        let dim_dist = if let (Some((xi, xj)), Some((yi, yj)), Some((zi, zj))) = (xs, ys, zs) {
            Some((
                (self.points[xi as usize].dist(&self.points[xj as usize])),
                (self.points[yi as usize].dist(&self.points[yj as usize])),
                (self.points[zi as usize].dist(&self.points[zj as usize])),
            ))
        } else {
            None
        };

        self.dim_dist = dim_dist;
    }

    pub fn write_as_obj<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        writeln!(w, "o grid")?;

        for pt in &self.points {
            writeln!(w, "v {} {} {}", pt.x(), pt.y(), pt.z())?;
        }

        for (a, neighs) in self.neighbors.iter().enumerate() {
            for b in neighs {
                // Skip these to avoid double outputting
                if *b < a as isize {
                    continue;
                }

                writeln!(w, "l {} {}", a + 1, b + 1)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod dual_tests {
    use super::*;

    const A: f64 = 1.0;

    fn norm(p: Pos) -> Pos {
        let l = (p.dot(&p)).sqrt();
        p / l
    }

    fn centroid(pts: &[Pos]) -> Pos {
        let mut c = Pos([0.0; 3]);
        for p in pts {
            c = c + *p;
        }
        c / pts.len() as f64
    }

    /// The 24 corners of the Voronoi cell of a BCC lattice with lattice constant `a`: a truncated
    /// octahedron, whose corners are the permutations of `(0, ±a/4, ±a/2)`.
    fn truncated_octahedron(a: f64) -> Vec<Pos> {
        let mut v = Vec::new();
        for (i, j, k) in [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ] {
            for s1 in [-1.0, 1.0] {
                for s2 in [-1.0, 1.0] {
                    let mut p = [0.0; 3];
                    p[i] = 0.0;
                    p[j] = s1 * a / 4.0;
                    p[k] = s2 * a / 2.0;
                    let p = Pos(p);
                    if !v.iter().any(|q: &Pos| q.dist(&p) < 1e-12) {
                        v.push(p);
                    }
                }
            }
        }
        assert_eq!(v.len(), 24);
        v
    }

    /// The 14 neighbours of a BCC site: 8 along the body diagonals (which share the hexagonal
    /// faces of the cell) and 6 along the axes (the square faces).
    fn bcc_offsets(a: f64) -> Vec<Pos> {
        let mut o = Vec::new();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    o.push(Pos([sx * a / 2.0, sy * a / 2.0, sz * a / 2.0]));
                }
            }
        }
        for j in 0..3 {
            for s in [-1.0, 1.0] {
                let mut p = [0.0; 3];
                p[j] = s * a;
                o.push(Pos(p));
            }
        }
        assert_eq!(o.len(), 14);
        o
    }

    /// Put the corners of a planar convex polygon into boundary order.
    fn sort_around(pts: &mut Vec<Pos>, n: Pos) {
        let c = centroid(pts);
        let n = norm(n);
        // Any direction in the plane will do as the zero angle.
        let seed = if n.x().abs() < 0.9 {
            Pos([1.0, 0.0, 0.0])
        } else {
            Pos([0.0, 1.0, 0.0])
        };
        let u = norm(seed - n * seed.dot(&n));
        let v = n.cross(&u);
        pts.sort_by(|p, q| {
            let ap = (p.dot(&v) - c.dot(&v)).atan2(p.dot(&u) - c.dot(&u));
            let aq = (q.dot(&v) - c.dot(&v)).atan2(q.dot(&u) - c.dot(&u));
            ap.partial_cmp(&aq).unwrap()
        });
    }

    /// A BCC lattice over `[0, n]^3` together with its Voronoi mesh, built from the geometry of
    /// the truncated octahedron so that the correspondence is known by construction.
    ///
    /// The dual is emitted un-welded, one closed polyhedron per site, so interior faces appear
    /// twice at the same place -- the harder of the two layouts a user might hand us.
    fn bcc_lattice(n: isize, triangulate: bool) -> VineyardsGridMesh {
        let mut sites: Vec<Pos> = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    sites.push(Pos([i as f64 * A, j as f64 * A, k as f64 * A]));
                    if i < n && j < n && k < n {
                        sites.push(Pos([
                            (i as f64 + 0.5) * A,
                            (j as f64 + 0.5) * A,
                            (k as f64 + 0.5) * A,
                        ]));
                    }
                }
            }
        }

        // Bonds: every pair of sites separated by one of the 14 neighbour offsets.
        let offsets = bcc_offsets(A);
        let mut neighbors: Vec<Vec<isize>> = vec![Vec::new(); sites.len()];
        for (i, s) in sites.iter().enumerate() {
            for d in &offsets {
                let t = *s + *d;
                if let Some(j) = sites.iter().position(|q| q.dist(&t) < 1e-9) {
                    neighbors[i].push(j as isize);
                }
            }
        }

        // Dual: the truncated octahedron around every site.  The face for offset `d` is made of
        // the cell corners lying on the bisector plane between the site and its neighbour.
        let cell = truncated_octahedron(A);
        let mut points: Vec<Pos> = Vec::new();
        let mut faces: Vec<Vec<u32>> = Vec::new();
        for s in &sites {
            for d in &offsets {
                let dh = norm(*d);
                let half = (d.dot(d)).sqrt() / 2.0;
                let mut face: Vec<Pos> = cell
                    .iter()
                    .filter(|v| (v.dot(&dh) - half).abs() < 1e-9)
                    .map(|v| *v + *s)
                    .collect();
                assert!(
                    face.len() == 4 || face.len() == 6,
                    "a truncated octahedron has square and hexagonal faces, got {}",
                    face.len()
                );
                sort_around(&mut face, dh);

                let base = points.len() as u32;
                points.extend_from_slice(&face);
                let idx: Vec<u32> = (0..face.len() as u32).map(|k| base + k).collect();
                if triangulate {
                    for k in 1..idx.len() - 1 {
                        faces.push(vec![idx[0], idx[k], idx[k + 1]]);
                    }
                } else {
                    faces.push(idx);
                }
            }
        }

        VineyardsGridMesh {
            points: sites,
            neighbors,
            r#type: "meshgrid".to_string(),
            dim_dist: None,
            dual: Some(DualMesh {
                points,
                faces,
                tree: None,
            }),
        }
    }

    #[test]
    fn bcc_edges_find_their_own_voronoi_face() {
        let mut grid = bcc_lattice(2, false);
        let sites = grid.points.clone();
        let neighbors = grid.neighbors.clone();

        let mut hexagons = 0;
        let mut squares = 0;

        for (i, ns) in neighbors.iter().enumerate() {
            for &j in ns {
                let (a, b) = (Index::fake(i as isize), Index::fake(j));
                let (p, q) = (sites[i], sites[j as usize]);
                let pts = grid
                    .dual_face_points(a, b)
                    .unwrap_or_else(|| panic!("no dual face for bond {:?} -> {:?}", p, q));

                let d = q - p;
                let dh = norm(d);
                let half = (d.dot(&d)).sqrt() / 2.0;

                // Every corner lies on the perpendicular bisector plane of the bond.
                for c in &pts {
                    let signed = (*c - p).dot(&dh);
                    assert!(
                        (signed - half).abs() < 1e-9,
                        "corner {:?} is not on the bisector of {:?} -> {:?}",
                        c,
                        p,
                        q
                    );
                }

                // The face is centred on the bond, and is the right shape: a hexagon for the 8
                // body-diagonal bonds, a square for the 6 axis-aligned ones.
                let mid = p + d / 2.0;
                assert!(centroid(&pts).dist(&mid) < 1e-9);
                let diagonal = d.x().abs() > 1e-9 && d.y().abs() > 1e-9 && d.z().abs() > 1e-9;
                if diagonal {
                    assert_eq!(pts.len(), 6, "body-diagonal bonds share hexagons");
                    hexagons += 1;
                } else {
                    assert_eq!(pts.len(), 4, "axis-aligned bonds share squares");
                    squares += 1;
                }

                // Wound so that the normal points from `p` towards `q`.
                let n = (pts[1] - pts[0]).cross(&(pts[2] - pts[0]));
                assert!(n.dot(&d) > 0.0, "face is wound the wrong way round");
            }
        }

        assert!(0 < hexagons && 0 < squares, "expected both kinds of face");
    }

    #[test]
    fn bcc_edges_match_a_triangulated_dual() {
        let mut grid = bcc_lattice(2, true);
        let sites = grid.points.clone();
        let neighbors = grid.neighbors.clone();

        for (i, ns) in neighbors.iter().enumerate() {
            for &j in ns {
                let (p, q) = (sites[i], sites[j as usize]);
                let pts = grid
                    .dual_face_points(Index::fake(i as isize), Index::fake(j))
                    .expect("triangulated dual should still be hit");

                // A fan triangle of the face, rather than the whole face, but still on the
                // bisector plane of the bond.
                assert_eq!(pts.len(), 3);
                let dh = norm(q - p);
                let half = p.dist(&q) / 2.0;
                for c in &pts {
                    assert!(((*c - p).dot(&dh) - half).abs() < 1e-9);
                }
            }
        }
    }

    /// On a cubic grid the inferred quad *is* the dual face, so the two paths have to agree.
    #[test]
    fn cubic_dual_agrees_with_the_inferred_quad() {
        let n = 3;
        let mut sites = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    sites.push(Pos([i as f64, j as f64, k as f64]));
                }
            }
        }
        let mut neighbors: Vec<Vec<isize>> = vec![Vec::new(); sites.len()];
        for (i, s) in sites.iter().enumerate() {
            for (j, t) in sites.iter().enumerate() {
                if (s.dist(t) - 1.0).abs() < 1e-9 {
                    neighbors[i].push(j as isize);
                }
            }
        }

        // The dual of a cubic grid: the six faces of the unit cube around every site.
        let mut points = Vec::new();
        let mut faces = Vec::new();
        for s in &sites {
            for j in 0..3 {
                for sign in [-1.0, 1.0] {
                    let (u, v) = ((j + 1) % 3, (j + 2) % 3);
                    let mut face = Vec::new();
                    for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                        let mut c = [0.0; 3];
                        c[j] = sign * 0.5;
                        c[u] = su * 0.5;
                        c[v] = sv * 0.5;
                        face.push(*s + Pos(c));
                    }
                    let base = points.len() as u32;
                    points.extend_from_slice(&face);
                    faces.push((0..4).map(|k| base + k).collect());
                }
            }
        }

        let plain = VineyardsGridMesh {
            points: sites.clone(),
            neighbors: neighbors.clone(),
            r#type: "meshgrid".to_string(),
            dim_dist: Some((1.0, 1.0, 1.0)),
            dual: None,
        };
        let mut with_dual = VineyardsGridMesh {
            dual: Some(DualMesh {
                points,
                faces,
                tree: None,
            }),
            ..plain.clone()
        };

        let sort_key = |pts: &[Pos]| {
            let mut v: Vec<String> = pts.iter().map(|p| format!("{:?}", p)).collect();
            v.sort();
            v
        };

        for (i, ns) in neighbors.iter().enumerate() {
            for &j in ns {
                let (a, b) = (Index::fake(i as isize), Index::fake(j));
                let quad = plain.dual_quad_points(a, b);
                let face = with_dual.dual_face_points(a, b).expect("cube face");
                assert_eq!(
                    sort_key(&quad),
                    sort_key(&face),
                    "inferred quad and cube dual face differ for {:?} -> {:?}",
                    sites[i],
                    sites[j as usize]
                );
            }
        }
    }

    #[test]
    fn dual_obj_parser_handles_blender_output() {
        // Normals and texture coordinates must not be mistaken for vertices, faces may name them,
        // and negative indices count back from the end.
        let obj = "\
# a comment
mtllib whatever.mtl
o Cell
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0.5 0.5
vn 0.0 0.0 1.0
usemtl Material
s off
f 1/1/1 2/1/1 3/1/1 4/1/1
f -4 -3 -2
";
        let dual = DualMesh::read_from_obj_string(obj).expect("should parse");
        assert_eq!(dual.points.len(), 4, "vn/vt must not become points");
        assert_eq!(dual.faces.len(), 2);
        assert_eq!(dual.faces[0], vec![0, 1, 2, 3]);
        assert_eq!(
            dual.faces[1],
            vec![0, 1, 2],
            "negative indices are relative"
        );

        assert!(
            DualMesh::read_from_obj_string("v 0 0 0\n").is_err(),
            "no faces"
        );
        assert!(
            DualMesh::read_from_obj_string("v 0 0 0\nf 1 2\n").is_err(),
            "2-gon"
        );
        assert!(
            DualMesh::read_from_obj_string("v 0 0 0\nf 1 2 9\n").is_err(),
            "out of range"
        );
    }
}
