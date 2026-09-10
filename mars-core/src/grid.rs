use std::collections::{HashMap, HashSet};

use tracing::{info, instrument, trace, warn};

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

/// A face of the dual grid: the Voronoi facet that separates grid vertices `a` and `b`, i.e. the
/// dual of the grid edge `(a, b)`.  Dual faces are supplied together with the grid (see
/// [VineyardsGridMesh::read_from_obj_string]).  The algorithm never needs them; only the output
/// does.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DualFace {
    /// Grid vertex id on one side.  Always `a < b`.
    pub a: isize,
    /// Grid vertex id on the other side.
    pub b: isize,
    /// Polygon vertices in cyclic order.
    pub vertices: Vec<Pos>,
}

/// What [VineyardsGridMesh::read_from_obj_string_with_report] found.  Printed by
/// `mars-cli grid-check`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GridReport {
    /// Grid points.
    pub points: usize,
    /// Undirected grid edges, after merging the `l` lines with the matched dual faces.
    pub edges: usize,
    /// Undirected edges that came from `l` lines.
    pub edges_from_lines: usize,
    /// Faces found in dual objects.
    pub faces_total: usize,
    /// Faces linked to a grid edge.
    pub faces_matched: usize,
    /// Faces whose vertices are not equidistant from their two nearest grid points.  These are the
    /// outer walls of boundary cells (the second site is not in the grid), or garbage.
    pub faces_dropped: usize,
    /// Faces that repeat an already matched face (e.g. one copy per cell).
    pub faces_duplicate: usize,
    /// Faces with fewer than three distinct vertices.
    pub faces_degenerate: usize,
    /// Edges without a dual face.  Swaps found along them have no geometry in the output.  Zero
    /// when no dual was supplied at all.
    pub edges_without_face: usize,
    /// Matched faces whose edge was not among the `l` lines.  Zero when there were no `l` lines.
    pub faces_without_line_edge: usize,
    /// `(number of vertices, count)` over the matched faces.
    pub face_vertex_histogram: Vec<(usize, usize)>,
    /// `(degree, count)` over the grid points.
    pub degree_histogram: Vec<(usize, usize)>,
}

impl GridReport {
    /// True if a dual was supplied.
    pub fn has_dual(&self) -> bool {
        self.faces_total > 0
    }
}

impl std::fmt::Display for GridReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn hist(h: &[(usize, usize)]) -> String {
            h.iter()
                .map(|(k, n)| format!("{}:{}", k, n))
                .collect::<Vec<_>>()
                .join(" ")
        }
        writeln!(
            f,
            "grid: {} points, {} edges ({} from l lines)",
            self.points, self.edges, self.edges_from_lines
        )?;
        writeln!(
            f,
            "  degree histogram (degree:count): {}",
            hist(&self.degree_histogram)
        )?;
        if self.has_dual() {
            writeln!(
                f,
                "dual: {} faces: {} matched, {} dropped (outer walls), {} duplicate, {} degenerate",
                self.faces_total,
                self.faces_matched,
                self.faces_dropped,
                self.faces_duplicate,
                self.faces_degenerate
            )?;
            writeln!(
                f,
                "  face size histogram (vertices:count): {}",
                hist(&self.face_vertex_histogram)
            )?;
            write!(
                f,
                "  edges without a face: {}; matched faces without an l edge: {}",
                self.edges_without_face, self.faces_without_line_edge
            )
        } else {
            write!(
                f,
                "dual: none supplied (legacy grid: dual faces are axis-aligned squares)"
            )
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VineyardsGridMesh {
    pub points: Vec<Pos>,
    /// Map from vertex index to the list of its neighbors.
    pub neighbors: Vec<Vec<isize>>,
    /// Should always be `"meshgrid"`. Used for serialization stuff.
    pub r#type: String,

    /// Dual faces supplied with the grid, at most one per edge.  Empty for legacy grids, whose
    /// dual faces are the axis-aligned squares of [Self::dual_quad_points].
    ///
    /// Serialized last, with a default, so that state files written before this field existed
    /// still load (rmp_serde encodes structs positionally).  Private because the lookup index
    /// below is built from it once; see [Self::dual_faces] and [Self::set_dual_faces].
    #[serde(default)]
    dual_faces: Vec<DualFace>,

    /// Lazily built lookup from `(a, b)` with `a < b` into `dual_faces`.
    #[serde(skip)]
    dual_index: std::sync::OnceLock<HashMap<(isize, isize), usize>>,

    /// Lazily inferred grid spacing along the three axes, for the legacy square rule (which
    /// assumes the mesh is a subset of a regular grid).  See [Self::dim_dist].
    #[serde(skip)]
    dim_dist: std::sync::OnceLock<Option<(f64, f64, f64)>>,
}

impl VineyardsGridMesh {
    pub fn empty() -> Self {
        Self::with_parts(Vec::new(), Vec::new(), Vec::new())
    }

    fn with_parts(points: Vec<Pos>, neighbors: Vec<Vec<isize>>, dual_faces: Vec<DualFace>) -> Self {
        Self {
            points,
            neighbors,
            r#type: "meshgrid".to_string(),
            dual_faces,
            dual_index: Default::default(),
            dim_dist: Default::default(),
        }
    }

    pub fn coordinate(&self, index: Index) -> Pos {
        self.points[index.0[0] as usize]
    }

    /// The supplied dual faces, one per edge that has one.
    pub fn dual_faces(&self) -> &[DualFace] {
        &self.dual_faces
    }

    /// Replace the dual faces (and reset the lookup index built from them).
    pub fn set_dual_faces(&mut self, faces: Vec<DualFace>) {
        self.dual_faces = faces;
        self.dual_index = Default::default();
    }

    /// Grid spacing along the three axes for the legacy square rule: the length of the first
    /// edge that differs in x, in y and in z (in adjacency order).  `None` if some axis has no
    /// such edge.  Inferred once from the adjacency, so it is available however the grid was
    /// constructed or loaded.
    pub fn dim_dist(&self) -> Option<(f64, f64, f64)> {
        *self.dim_dist.get_or_init(|| {
            let first = |axis: usize| {
                self.neighbors.iter().enumerate().find_map(|(i, ns)| {
                    ns.iter()
                        .find(|&&j| self.points[i].0[axis] != self.points[j as usize].0[axis])
                        .map(|&j| self.points[i].dist(&self.points[j as usize]))
                })
            };
            match (first(0), first(1), first(2)) {
                (Some(dx), Some(dy), Some(dz)) => Some((dx, dy, dz)),
                _ => None,
            }
        })
    }

    /// True if the grid came with dual faces.
    pub fn has_dual(&self) -> bool {
        !self.dual_faces.is_empty()
    }

    fn dual_index(&self) -> &HashMap<(isize, isize), usize> {
        self.dual_index.get_or_init(|| {
            self.dual_faces
                .iter()
                .enumerate()
                .map(|(k, f)| ((f.a.min(f.b), f.a.max(f.b)), k))
                .collect()
        })
    }

    /// The supplied dual face of the edge `(a, b)`, if any.
    pub fn dual_face(&self, a: Index, b: Index) -> Option<&DualFace> {
        let key = (a.x().min(b.x()), a.x().max(b.x()));
        self.dual_index().get(&key).map(|&k| &self.dual_faces[k])
    }

    /// Polygon of the dual face of the grid edge `(a, b)`, in cyclic order.
    ///
    /// This is the supplied Voronoi facet if the grid came with a dual, otherwise the legacy
    /// axis-aligned square of [Self::dual_quad_points].  Returns an empty polygon if the edge has
    /// no face; callers should skip (and count) such faces.
    pub fn dual_face_points(&self, a: Index, b: Index) -> Vec<Pos> {
        if let Some(face) = self.dual_face(a, b) {
            return face.vertices.clone();
        }
        // Callers count and report the skipped faces; per-edge messages stay at trace level.
        if self.has_dual() {
            trace!(
                "grid edge ({}, {}) has no dual face; its swaps are not drawn",
                a.x(),
                b.x()
            );
            return Vec::new();
        }
        match self.dual_quad_points(a, b) {
            Some(quad) => quad.to_vec(),
            None => {
                trace!(
                    "grid edge ({}, {}) is not axis-aligned and no dual was supplied; its swaps are not drawn",
                    a.x(),
                    b.x()
                );
                Vec::new()
            }
        }
    }

    /// Legacy dual face: the axis-aligned rectangle bisecting an axis-parallel edge of a cubic
    /// grid, with side lengths from `dim_dist`.  `None` if the edge is not parallel to a
    /// coordinate axis (within an absolute 1e-3 on the other two coordinates).
    pub fn dual_quad_points(&self, a: Index, b: Index) -> Option<[Pos; 4]> {
        let a = self.points[a.0[0] as usize];
        let b = self.points[b.0[0] as usize];
        let (dx, dy, dz) = self.dim_dist().unwrap_or_else(|| {
            let dist = a.dist(&b);
            (dist, dist, dist)
        });

        let moving = [
            (a.x() - b.x()).abs() > 1e-3,
            (a.y() - b.y()).abs() > 1e-3,
            (a.z() - b.z()).abs() > 1e-3,
        ];
        let middle = a + (b - a) / 2.0;
        let (p, q) = match moving {
            [true, false, false] => (Pos([0.0, dy / 2.0, 0.0]), Pos([0.0, 0.0, dz / 2.0])),
            [false, true, false] => (Pos([dx / 2.0, 0.0, 0.0]), Pos([0.0, 0.0, dz / 2.0])),
            [false, false, true] => (Pos([dx / 2.0, 0.0, 0.0]), Pos([0.0, dy / 2.0, 0.0])),
            _ => return None,
        };
        Some([
            middle - p - q,
            middle - p + q,
            middle + p + q,
            middle + p - q,
        ])
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

        // The halves only drive the traversal; dual faces are read from the full grid at export.
        (
            Self::with_parts(self.points.clone(), lower_edges, Vec::new()),
            Self::with_parts(self.points.clone(), upper_edges, Vec::new()),
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

    /// Read a grid, and optionally its dual, from the text of an `.obj` file.
    ///
    /// The file may contain several objects (`o` or `g` lines).  Objects with `f` lines are the
    /// **dual**: the Voronoi tessellation of the grid points.  Every other object contributes grid
    /// points (`v`) and edges (`l`).  An object with both faces and edges is an error, because
    /// either reading of it would be silently wrong.  A file without faces is a plain grid,
    /// exactly as before.
    ///
    /// Each dual face is linked to the grid edge it bisects by geometry: the two grid points
    /// nearest to the face's centroid are the edge, and every vertex of the face must be
    /// equidistant from them.  Faces that fail this (the outer walls of boundary cells) are
    /// dropped, duplicate faces are merged.  The grid's edges are the `l` lines followed by the
    /// edges implied by the matched faces.
    pub fn read_from_obj_string(s: &str) -> Result<Self, String> {
        let (grid, report) = Self::read_from_obj_string_with_report(s)?;
        info!("read grid\n{}", report);
        Ok(grid)
    }

    /// [Self::read_from_obj_string], also returning what was found.
    pub fn read_from_obj_string_with_report(s: &str) -> Result<(Self, GridReport), String> {
        let obj = ObjFile::parse(s)?;

        let mut point_of_vertex: Vec<Option<usize>> = vec![None; obj.vertices.len()];
        let mut points: Vec<Pos> = Vec::new();
        let mut line_edges: Vec<(usize, usize)> = Vec::new();
        let mut faces: Vec<Vec<Pos>> = Vec::new();

        for o in &obj.objects {
            if !o.faces.is_empty() {
                // A dual object.  Faces and edges in one object would be misread either way (a
                // grid without its dual, or a dual whose corners become grid points), so refuse.
                if !o.lines.is_empty() {
                    return Err(format!(
                        "object {:?} has both {} faces and {} edges. The dual (faces) and the grid \
                         (edges) must be separate objects: if this is the lattice, regenerate it \
                         with Delaunay faces = None; if this is the Voronoi object, delete its \
                         loose edges; if Blender merged the two, export them as separate objects.",
                        o.name,
                        o.faces.len(),
                        o.lines.len()
                    ));
                }
                faces.extend(
                    o.faces
                        .iter()
                        .map(|f| f.iter().map(|&v| obj.vertices[v]).collect::<Vec<_>>()),
                );
                continue;
            }
            for &v in &o.vertices {
                point_of_vertex[v] = Some(points.len());
                points.push(obj.vertices[v]);
            }
            for &(i, j) in &o.lines {
                match (point_of_vertex[i], point_of_vertex[j]) {
                    (Some(a), Some(b)) => line_edges.push((a, b)),
                    _ => {
                        return Err(format!(
                            "object {:?}: an l line refers to a vertex of a dual object",
                            o.name
                        ))
                    }
                }
            }
        }

        Self::assemble(points, line_edges, faces)
    }

    /// Add the dual from a second `.obj` file: all of its faces are taken as dual faces of this
    /// grid's points.  Replaces any dual faces read before, and adds the edges the faces imply.
    pub fn merge_dual_from_obj_string(&mut self, s: &str) -> Result<GridReport, String> {
        let obj = ObjFile::parse(s)?;
        let faces: Vec<Vec<Pos>> = obj
            .objects
            .iter()
            .flat_map(|o| {
                o.faces
                    .iter()
                    .map(|f| f.iter().map(|&v| obj.vertices[v]).collect::<Vec<_>>())
            })
            .collect();
        if faces.is_empty() {
            return Err("the dual file contains no faces".to_string());
        }

        let mut edges = Vec::new();
        for (a, ns) in self.neighbors.iter().enumerate() {
            for &b in ns {
                if (a as isize) < b {
                    edges.push((a, b as usize));
                }
            }
        }
        if self.has_dual() {
            warn!(
                "the grid already had {} dual faces; the separate dual file replaces them, but \
                 the edges they implied stay",
                self.dual_faces.len()
            );
        }
        let (grid, report) = Self::assemble(self.points.clone(), edges, faces)?;
        *self = grid;
        Ok(report)
    }

    /// Common tail of the readers: check the points, match the faces, build the adjacency.
    fn assemble(
        points: Vec<Pos>,
        line_edges: Vec<(usize, usize)>,
        faces: Vec<Vec<Pos>>,
    ) -> Result<(Self, GridReport), String> {
        check_distinct_points(&points)?;

        let mut report = GridReport {
            points: points.len(),
            ..Default::default()
        };
        let dual_faces = match_dual_faces(&points, &faces, &mut report)?;

        // The `l` lines first, in file order (this keeps the traversal of legacy grids exactly as
        // it was), then the edges implied by the dual faces.
        let mut neighbors: Vec<Vec<isize>> = vec![Vec::new(); points.len()];
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut add_edge = |a: usize, b: usize| {
            if a != b && seen.insert((a.min(b), a.max(b))) {
                neighbors[a].push(b as isize);
                neighbors[b].push(a as isize);
            }
        };
        for &(a, b) in &line_edges {
            add_edge(a, b);
        }
        for f in &dual_faces {
            add_edge(f.a as usize, f.b as usize);
        }

        let line_set: HashSet<(usize, usize)> = line_edges
            .iter()
            .filter(|(a, b)| a != b)
            .map(|&(a, b)| (a.min(b), a.max(b)))
            .collect();
        let face_keys: HashSet<(usize, usize)> = dual_faces
            .iter()
            .map(|f| (f.a as usize, f.b as usize))
            .collect();
        report.edges = seen.len();
        report.edges_from_lines = line_set.len();
        if !face_keys.is_empty() {
            report.edges_without_face = seen.iter().filter(|e| !face_keys.contains(e)).count();
        }
        if !line_set.is_empty() {
            report.faces_without_line_edge =
                face_keys.iter().filter(|k| !line_set.contains(k)).count();
        }
        report.degree_histogram = histogram(neighbors.iter().map(|n| n.len()));

        Ok((Self::with_parts(points, neighbors, dual_faces), report))
    }

    /// Write the grid as an `.obj`: an object `grid` with the points and edges, and, if there are
    /// dual faces, an object `dual` with one polygon per face.  Readable by
    /// [Self::read_from_obj_string].
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

        if !self.dual_faces.is_empty() {
            writeln!(w, "o dual")?;
            let mut vi = self.points.len();
            for face in &self.dual_faces {
                write_face(&mut w, &face.vertices, &mut vi)?;
            }
        }

        Ok(())
    }
}

/// Write one polygon as `.obj`: its vertices as `v` lines and one `f` line referring to them.
/// `vi` is the number of vertices written to `w` so far and is advanced.  Polygons with fewer
/// than three vertices are skipped, returning `false`.
pub fn write_face<W: std::io::Write>(
    w: &mut W,
    pts: &[Pos],
    vi: &mut usize,
) -> std::io::Result<bool> {
    if pts.len() < 3 {
        return Ok(false);
    }
    for p in pts {
        writeln!(w, "v {} {} {}", p.x(), p.y(), p.z())?;
    }
    write!(w, "f")?;
    for k in 1..=pts.len() {
        write!(w, " {}", *vi + k)?;
    }
    writeln!(w)?;
    *vi += pts.len();
    Ok(true)
}

// ---------------------------------------------------------------------------------------------
// .obj parsing and dual-face matching
// ---------------------------------------------------------------------------------------------

/// Relative tolerance (in units of the edge length) for a face vertex to count as equidistant
/// from the two grid points of its edge.  Blender writes six decimals, i.e. ~1e-5 relative for
/// typical grid spacings, so this is generous but far below any real lattice feature.
const EQUIDISTANT_TOL: f64 = 1e-3;
/// Relative tolerance for merging consecutive near-identical face vertices.
const MERGE_TOL: f64 = 1e-6;
/// Relative tolerance for two faces of the same edge to count as the same face.
const DUPLICATE_TOL: f64 = 1e-5;

/// One `o` object of an `.obj` file.  Vertex ids index [ObjFile::vertices].
#[derive(Debug, Default)]
struct ObjObject {
    name: String,
    vertices: Vec<usize>,
    lines: Vec<(usize, usize)>,
    faces: Vec<Vec<usize>>,
}

impl ObjObject {
    fn is_empty(&self) -> bool {
        self.vertices.is_empty() && self.lines.is_empty() && self.faces.is_empty()
    }
}

/// The parts of an `.obj` file we care about: vertices, and per object the edges and faces.
/// Normals, texture coordinates, materials and smoothing groups are skipped.
#[derive(Debug, Default)]
struct ObjFile {
    vertices: Vec<Pos>,
    objects: Vec<ObjObject>,
}

impl ObjFile {
    fn parse(s: &str) -> Result<Self, String> {
        let mut file = ObjFile::default();
        file.objects.push(ObjObject::default());

        for (n, raw) in s.lines().enumerate() {
            let lineno = n + 1;
            let mut toks = raw.split_ascii_whitespace();
            let Some(kw) = toks.next() else { continue };
            match kw {
                "v" => {
                    let mut c = [0.0; 3];
                    for coord in c.iter_mut() {
                        let t = toks.next().ok_or_else(|| {
                            format!("line {}: a vertex needs three coordinates", lineno)
                        })?;
                        *coord = t
                            .parse::<f64>()
                            .map_err(|e| format!("line {}: {}", lineno, e))?;
                    }
                    if !c.iter().all(|x| x.is_finite()) {
                        return Err(format!(
                            "line {}: vertex with a non-finite coordinate: {:?}",
                            lineno,
                            raw.trim()
                        ));
                    }
                    let id = file.vertices.len();
                    file.vertices.push(Pos(c));
                    file.objects.last_mut().unwrap().vertices.push(id);
                }
                "l" => {
                    let ids = parse_indices(toks, file.vertices.len(), lineno)?;
                    let obj = file.objects.last_mut().unwrap();
                    for w in ids.windows(2) {
                        obj.lines.push((w[0], w[1]));
                    }
                }
                "f" => {
                    let ids = parse_indices(toks, file.vertices.len(), lineno)?;
                    file.objects.last_mut().unwrap().faces.push(ids);
                }
                // Blender writes `o` per object, or `g` with "Objects as OBJ Groups".
                "o" | "g" => {
                    let name = toks.collect::<Vec<_>>().join(" ");
                    let last = file.objects.last_mut().unwrap();
                    if last.is_empty() {
                        last.name = name;
                    } else {
                        file.objects.push(ObjObject {
                            name,
                            ..Default::default()
                        });
                    }
                }
                // Comments, `vn`, `vt`, `s`, `usemtl`, `mtllib`, ...
                _ => {}
            }
        }

        Ok(file)
    }
}

/// Parse the vertex indices of an `l` or `f` line: `a`, `a/t`, `a//n` and `a/t/n` forms, 1-based,
/// negative meaning relative to the last vertex defined so far.  Returns 0-based ids.
fn parse_indices<'a>(
    toks: impl Iterator<Item = &'a str>,
    nverts: usize,
    lineno: usize,
) -> Result<Vec<usize>, String> {
    toks.map(|t| {
        let first = t.split('/').next().unwrap_or("");
        let n: isize = first
            .parse()
            .map_err(|_| format!("line {}: bad vertex index {:?}", lineno, t))?;
        let id = if n > 0 {
            n - 1
        } else if n < 0 {
            nverts as isize + n
        } else {
            return Err(format!("line {}: vertex index 0 is not valid in .obj", lineno));
        };
        if id < 0 || id as usize >= nverts {
            return Err(format!(
                "line {}: vertex index {} refers to a vertex that does not exist (yet)",
                lineno, n
            ));
        }
        Ok(id as usize)
    })
    .collect()
}

/// Error if two points are closer than 1e-5.  Points are bucketed on a 1e-5 lattice and compared
/// with the 27 surrounding buckets, so this is linear in the number of points (a sort-and-sweep
/// along one axis is quadratic per slab on axis-aligned lattices).
fn check_distinct_points(points: &[Pos]) -> Result<(), String> {
    const TOL: f64 = 1e-5;
    let key = |p: &Pos| {
        let f = |k: usize| (p.0[k] / TOL).round() as i64;
        (f(0), f(1), f(2))
    };
    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let (kx, ky, kz) = key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = buckets.get(&(kx + dx, ky + dy, kz + dz)) {
                        for &j in bucket {
                            if points[j].dist(p) < TOL {
                                return Err(format!(
                                    "Two grid vertices are too close together: {} and {}",
                                    j, i
                                ));
                            }
                        }
                    }
                }
            }
        }
        buckets.entry((kx, ky, kz)).or_default().push(i);
    }
    Ok(())
}

/// `(value, count)` pairs, sorted by value.
fn histogram(values: impl Iterator<Item = usize>) -> Vec<(usize, usize)> {
    let mut h = std::collections::BTreeMap::new();
    for v in values {
        *h.entry(v).or_insert(0) += 1;
    }
    h.into_iter().collect()
}

fn centroid(vs: &[Pos]) -> Pos {
    let mut c = Pos([0.0; 3]);
    for v in vs {
        c = c + *v;
    }
    c / vs.len() as f64
}

/// Drop vertices that repeat an earlier one within `tol`, consecutive or not, keeping the first
/// occurrence.  A polygon that lists a vertex twice is degenerate; removing the repeat is the
/// most useful reading of it.
fn dedupe_vertices(vs: &[Pos], tol: f64) -> Vec<Pos> {
    let mut out: Vec<Pos> = Vec::with_capacity(vs.len());
    for &v in vs {
        if !out.iter().any(|o| o.dist(&v) <= tol) {
            out.push(v);
        }
    }
    out
}

/// True if `p` and `q` hold the same vertices, as multisets, within `tol`.
fn same_vertex_set(p: &[Pos], q: &[Pos], tol: f64) -> bool {
    if p.len() != q.len() {
        return false;
    }
    let mut used = vec![false; q.len()];
    p.iter().all(|a| {
        match (0..q.len()).find(|&k| !used[k] && a.dist(&q[k]) <= tol) {
            Some(k) => {
                used[k] = true;
                true
            }
            None => false,
        }
    })
}

/// Nearest-neighbour queries over the grid points through a uniform spatial hash.
struct PointLocator<'a> {
    points: &'a [Pos],
    lo: Pos,
    hi: Pos,
    /// Hash cell size, about one nearest-neighbour spacing.
    cell: f64,
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl<'a> PointLocator<'a> {
    /// `points` must hold at least two finite points.
    fn new(points: &'a [Pos]) -> Self {
        let mut lo = Pos([f64::MAX; 3]);
        let mut hi = Pos([f64::MIN; 3]);
        for p in points {
            for k in 0..3 {
                lo.0[k] = lo.0[k].min(p.0[k]);
                hi.0[k] = hi.0[k].max(p.0[k]);
            }
        }
        let cell = estimate_spacing(points)
            .max(lo.dist(&hi) * 1e-9)
            .max(f64::MIN_POSITIVE);
        let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            cells.entry(hash_key(p, &lo, cell)).or_default().push(i);
        }
        Self {
            points,
            lo,
            hi,
            cell,
            cells,
        }
    }

    /// True if `p` lies more than `margin` outside the bounding box of the points.
    fn is_far_outside(&self, p: Pos, margin: f64) -> bool {
        (0..3).any(|k| p.0[k] < self.lo.0[k] - margin || p.0[k] > self.hi.0[k] + margin)
    }

    /// The two grid points nearest to `c`, nearest first.  `None` only if fewer than two points
    /// have a finite distance to `c`.
    fn two_nearest(&self, c: Pos) -> Option<(usize, usize)> {
        fn pick(points: &[Pos], c: &Pos, best: &mut [(f64, usize); 2], i: usize) {
            let d = points[i].dist2(c);
            if d < best[0].0 {
                best[1] = best[0];
                best[0] = (d, i);
            } else if d < best[1].0 {
                best[1] = (d, i);
            }
        }
        let n = self.points.len();
        let mut best = [(f64::INFINITY, usize::MAX); 2];
        let (kx, ky, kz) = hash_key(&c, &self.lo, self.cell);

        // Grow a cube of cells around `c`, visiting each shell once.  Every point within
        // `r * cell` of `c` lies in the (2r+1)^3 cube, so once the second-nearest candidate is
        // closer than that the answer is final.  When the cube holds more cells than there are
        // points, scanning the points directly is cheaper (and always correct).
        let mut r: i64 = 0;
        while (2 * r + 1).pow(3) <= n as i64 {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs() != r && dy.abs() != r && dz.abs() != r {
                            continue; // strictly inside the shell: seen in an earlier round
                        }
                        if let Some(bucket) = self.cells.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &i in bucket {
                                pick(self.points, &c, &mut best, i);
                            }
                        }
                    }
                }
            }
            if best[1].1 != usize::MAX && best[1].0.sqrt() <= r as f64 * self.cell {
                return Some((best[0].1, best[1].1));
            }
            r += 1;
        }
        // Scan everything, from scratch: re-picking a point already in `best` would enter it twice.
        let mut best = [(f64::INFINITY, usize::MAX); 2];
        for i in 0..n {
            pick(self.points, &c, &mut best, i);
        }
        (best[1].1 != usize::MAX).then(|| (best[0].1, best[1].1))
    }
}

fn hash_key(p: &Pos, lo: &Pos, cell: f64) -> (i64, i64, i64) {
    let f = |k: usize| ((p.0[k] - lo.0[k]) / cell).floor() as i64;
    (f(0), f(1), f(2))
}

/// Typical nearest-neighbour distance between grid points, from a sample.
fn estimate_spacing(points: &[Pos]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 1.0;
    }
    let samples = 32.min(n);
    let mut ds: Vec<f64> = (0..samples)
        .map(|s| {
            let i = s * n / samples;
            let p = points[i];
            points
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, q)| p.dist2(q))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .collect();
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ds[ds.len() / 2]
}

/// Link each face to the grid edge it bisects.
///
/// Every point in the relative interior of a Voronoi facet has exactly two nearest sites, the
/// two the facet separates.  So the two grid points nearest to the face's centroid are its edge,
/// and all its vertices must be equidistant from them.  Faces failing that are outer walls of
/// boundary cells and are dropped; repeated faces are merged; two different faces on one edge
/// (a triangulated dual) are an error.
fn match_dual_faces(
    points: &[Pos],
    faces: &[Vec<Pos>],
    report: &mut GridReport,
) -> Result<Vec<DualFace>, String> {
    report.faces_total = faces.len();
    if faces.is_empty() {
        return Ok(Vec::new());
    }
    if points.len() < 2 {
        return Err("a dual needs at least two grid points".to_string());
    }

    let locator = PointLocator::new(points);
    // Blender stores float32 and writes six decimals, so far from the origin the absolute error
    // grows with the coordinates: the tolerances get a floor proportional to them.
    let coord_scale = points
        .iter()
        .flat_map(|p| p.0)
        .fold(0.0f64, |m, x| m.max(x.abs()));
    let mut out: Vec<DualFace> = Vec::new();
    let mut index: HashMap<(usize, usize), usize> = HashMap::new();

    for (fi, face) in faces.iter().enumerate() {
        if face.len() < 3 {
            report.faces_degenerate += 1;
            continue;
        }
        let c = centroid(face);
        // A wall between two grid points has its centroid between them, hence inside their
        // bounding box.  Anything well outside is an outer wall of some other cell.
        if locator.is_far_outside(c, 2.0 * locator.cell) {
            report.faces_dropped += 1;
            continue;
        }
        let Some((a, b)) = locator.two_nearest(c) else {
            report.faces_degenerate += 1;
            continue;
        };
        let (pa, pb) = (points[a], points[b]);
        let len = pa.dist(&pb);

        let vertices = dedupe_vertices(face, MERGE_TOL * len);
        if vertices.len() < 3 {
            report.faces_degenerate += 1;
            continue;
        }

        let tol = (EQUIDISTANT_TOL * len).max(1e-6 * coord_scale);
        let equidistant = vertices
            .iter()
            .all(|x| (x.dist(&pa) - x.dist(&pb)).abs() <= tol);
        if !equidistant {
            trace!(face = fi, a, b, "dropped: not equidistant from its two nearest grid points");
            report.faces_dropped += 1;
            continue;
        }

        let key = (a.min(b), a.max(b));
        if let Some(&k) = index.get(&key) {
            let dup_tol = (DUPLICATE_TOL * len).max(1e-6 * coord_scale);
            if same_vertex_set(&out[k].vertices, &vertices, dup_tol) {
                report.faces_duplicate += 1;
                continue;
            }
            let mismatch = vertices
                .iter()
                .map(|v| {
                    out[k]
                        .vertices
                        .iter()
                        .map(|w| v.dist(w))
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(0.0, f64::max);
            return Err(format!(
                "dual faces {} and {} both bisect grid edge ({}, {}) but differ: vertices up to \
                 {:.2e} apart for an edge of length {:.2e}. Either the dual is triangulated (export \
                 it without triangulation), or two copies of the same wall disagree by more than \
                 {:.0e} of the edge length.",
                fi, k, key.0, key.1, mismatch, len, DUPLICATE_TOL
            ));
        }
        index.insert(key, out.len());
        out.push(DualFace {
            a: key.0 as isize,
            b: key.1 as isize,
            vertices,
        });
    }

    report.faces_matched = out.len();
    report.face_vertex_histogram = histogram(out.iter().map(|f| f.vertices.len()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::*;

    fn read(s: &str) -> (VineyardsGridMesh, GridReport) {
        VineyardsGridMesh::read_from_obj_string_with_report(s).expect("parse grid")
    }

    fn edge_count(g: &VineyardsGridMesh) -> usize {
        g.neighbors.iter().map(|n| n.len()).sum::<usize>() / 2
    }

    fn fake(i: usize) -> Index {
        Index::fake(i as isize)
    }

    /// Every inner wall matched, every outer wall dropped, nothing left unexplained.
    fn assert_walls(r: &GridReport, block: &LatticeBlock, histogram: &[(usize, usize)]) {
        assert_eq!(r.faces_matched, block.inner_walls());
        assert_eq!(r.faces_dropped, block.outer_walls());
        assert_eq!(r.faces_duplicate, 0);
        assert_eq!(r.faces_degenerate, 0);
        assert_eq!(r.edges_without_face, 0);
        assert_eq!(r.faces_without_line_edge, 0);
        assert_eq!(r.face_vertex_histogram, histogram);
    }

    #[test]
    fn legacy_cylinder_grid_parses_as_before() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/cylinder_grid.obj");
        let s = std::fs::read_to_string(path).expect("read example grid");
        let (g, r) = read(&s);
        assert_eq!(g.points.len(), 2584);
        assert_eq!(edge_count(&g), 7141);
        assert_eq!(r.edges, 7141);
        assert_eq!(r.edges_from_lines, 7141);
        assert!(!g.has_dual());
        assert!(!r.has_dual());
        assert_eq!(r.degree_histogram, vec![(3, 8), (4, 152), (5, 894), (6, 1530)]);
        let (dx, dy, dz) = g.dim_dist().expect("dim_dist for a legacy grid");
        for d in [dx, dy, dz] {
            assert!((d - 0.13215).abs() < 1e-5, "spacing {}", d);
        }
        // Every edge is axis-aligned, so every dual face is a square of the lattice spacing.
        for (a, ns) in g.neighbors.iter().enumerate() {
            for &b in ns {
                let f = g.dual_face_points(fake(a), Index::fake(b));
                assert_eq!(f.len(), 4);
                for k in 0..4 {
                    let side = f[k].dist(&f[(k + 1) % 4]);
                    assert!((side - 0.13215).abs() < 1e-5, "side {}", side);
                }
            }
        }
    }

    #[test]
    fn sc_block_faces_equal_legacy_squares() {
        let block = lattice_block(Lattice::Sc, [3, 3, 3], 0.4, Pos([0.1, -0.2, 0.3]));
        let opts = ObjOptions {
            lines: Lines::Shell1,
            ..Default::default()
        };
        let (g, r) = read(&block_to_obj(&block, &opts));
        let (legacy, _) = read(&block_to_obj(
            &block,
            &ObjOptions {
                dual: false,
                ..opts
            },
        ));
        assert_eq!(g.points.len(), 64);
        assert_walls(&r, &block, &[(4, block.inner_walls())]);
        assert_eq!(edge_count(&g), edge_count(&legacy));
        for f in &g.dual_faces {
            let quad = legacy
                .dual_quad_points(Index::fake(f.a), Index::fake(f.b))
                .expect("axis-aligned edge");
            assert!(
                same_vertex_set(&f.vertices, &quad, 1e-9),
                "{:?} vs {:?}",
                f.vertices,
                quad
            );
            assert!(same_vertex_set(
                &g.dual_face_points(Index::fake(f.b), Index::fake(f.a)),
                &quad,
                1e-9
            ));
        }
    }

    #[test]
    fn bcc_block_matches_all_faces() {
        let block = lattice_block(Lattice::Bcc, [3, 3, 3], 1.0, Pos([0.0; 3]));
        let (g, r) = read(&block_to_obj(&block, &ObjOptions::default()));
        assert_eq!(g.points.len(), 64 + 27);
        assert_eq!(r.faces_total, block.walls.len());
        assert_walls(
            &r,
            &block,
            &[(4, block.shell2.len()), (6, block.shell1.len())],
        );
        assert_eq!(r.edges, block.shell1.len() + block.shell2.len());

        // The block is centred on a body centre with all 14 neighbours present.
        let centre = g
            .points
            .iter()
            .position(|p| p.norm() < 1e-9)
            .expect("centre point");
        assert_eq!(g.neighbors[centre].len(), 14);
        let side = 2f64.sqrt() / 4.0; // truncated octahedron edge for a = 1
        for &nb in &g.neighbors[centre] {
            let f = g.dual_face(fake(centre), Index::fake(nb)).expect("face");
            let n = f.vertices.len();
            for k in 0..n {
                let e = f.vertices[k].dist(&f.vertices[(k + 1) % n]);
                assert!((e - side).abs() < 1e-9, "edge {}", e);
            }
            let c = centroid(&f.vertices);
            let d = g.points[nb as usize].dist(&g.points[centre]);
            if (d - 3f64.sqrt() / 2.0).abs() < 1e-9 {
                assert_eq!(n, 6, "body-diagonal edge -> hexagon");
                for v in &f.vertices {
                    assert!((v.dist(&c) - side).abs() < 1e-9);
                }
            } else {
                assert!((d - 1.0).abs() < 1e-9);
                assert_eq!(n, 4, "axis edge -> square");
            }
            // The face lies in the bisector plane of the edge.
            let (pa, pb) = (g.points[centre], g.points[nb as usize]);
            for v in &f.vertices {
                assert!((v.dist(&pa) - v.dist(&pb)).abs() < 1e-9);
            }
        }

        // With only first-shell l lines the six square faces per point have no l edge, but they
        // are still grid edges.
        let (g1, r1) = read(&block_to_obj(
            &block,
            &ObjOptions {
                lines: Lines::Shell1,
                ..Default::default()
            },
        ));
        assert_eq!(edge_count(&g1), edge_count(&g));
        assert_eq!(r1.edges_from_lines, block.shell1.len());
        assert_eq!(r1.faces_without_line_edge, block.shell2.len());
        assert_eq!(r1.edges_without_face, 0);

        // Without any l lines the edges come from the faces alone.
        let (g0, r0) = read(&block_to_obj(
            &block,
            &ObjOptions {
                lines: Lines::None,
                ..Default::default()
            },
        ));
        assert_eq!(edge_count(&g0), edge_count(&g));
        assert_eq!(r0.edges_from_lines, 0);
        assert_eq!(r0.faces_without_line_edge, 0);
    }

    #[test]
    fn fcc_block_faces_are_rhombi() {
        let block = lattice_block(Lattice::Fcc, [2, 2, 2], 1.0, Pos([0.0; 3]));
        let (g, r) = read(&block_to_obj(
            &block,
            &ObjOptions {
                lines: Lines::Shell1,
                ..Default::default()
            },
        ));
        assert_walls(&r, &block, &[(4, block.inner_walls())]);

        let centre = g
            .points
            .iter()
            .position(|p| p.norm() < 1e-9)
            .expect("centre point");
        assert_eq!(g.neighbors[centre].len(), 12);
        let side = 3f64.sqrt() / 4.0;
        for &nb in &g.neighbors[centre] {
            let f = g.dual_face(fake(centre), Index::fake(nb)).expect("face");
            assert_eq!(f.vertices.len(), 4);
            for k in 0..4 {
                let e = f.vertices[k].dist(&f.vertices[(k + 1) % 4]);
                assert!((e - side).abs() < 1e-9, "edge {}", e);
            }
            let d1 = f.vertices[0].dist(&f.vertices[2]);
            let d2 = f.vertices[1].dist(&f.vertices[3]);
            let (lo, hi) = (d1.min(d2), d1.max(d2));
            assert!((lo - 0.5).abs() < 1e-9 && (hi - 0.5f64.sqrt()).abs() < 1e-9);
        }
    }

    #[test]
    fn duplicate_faces_are_merged() {
        let block = lattice_block(Lattice::Bcc, [2, 2, 2], 1.0, Pos([0.0; 3]));
        let (_, r) = read(&block_to_obj(
            &block,
            &ObjOptions {
                duplicate_walls: true,
                ..Default::default()
            },
        ));
        assert_eq!(r.faces_total, 2 * block.walls.len());
        assert_eq!(r.faces_matched, block.inner_walls());
        assert_eq!(r.faces_duplicate, block.inner_walls());
        assert_eq!(r.faces_dropped, 2 * block.outer_walls());
    }

    #[test]
    fn triangulated_dual_is_rejected() {
        let block = lattice_block(Lattice::Bcc, [2, 2, 2], 1.0, Pos([0.0; 3]));
        let err = VineyardsGridMesh::read_from_obj_string(&block_to_obj(
            &block,
            &ObjOptions {
                triangulate: true,
                ..Default::default()
            },
        ))
        .err()
        .expect("a triangulated dual must be rejected");
        assert!(err.contains("triangulated"), "{}", err);
    }

    #[test]
    fn blender_style_normals_rounding_and_slash_syntax_are_tolerated() {
        let block = lattice_block(Lattice::Bcc, [2, 2, 2], 0.13215, Pos([0.4, -1.1, 0.9]));
        let (g, r) = read(&block_to_obj(&block, &ObjOptions::default()));
        let (g6, r6) = read(&block_to_obj(
            &block,
            &ObjOptions {
                normals: true,
                decimals: Some(6),
                ..Default::default()
            },
        ));
        assert_eq!(g6.points.len(), g.points.len(), "vn lines are not vertices");
        assert_eq!(r6.faces_matched, r.faces_matched);
        assert_eq!(r6.faces_dropped, r.faces_dropped);
        assert_eq!(r6.face_vertex_histogram, r.face_vertex_histogram);
        assert_eq!(edge_count(&g6), edge_count(&g));
    }

    #[test]
    fn separate_dual_file_merges() {
        let block = lattice_block(Lattice::Bcc, [2, 2, 2], 1.0, Pos([0.0; 3]));
        let (combined, rc) = read(&block_to_obj(&block, &ObjOptions::default()));
        let (mut grid, rg) = read(&block_to_obj(
            &block,
            &ObjOptions {
                dual: false,
                ..Default::default()
            },
        ));
        assert!(!rg.has_dual());
        let dual_only = block_to_obj(
            &block,
            &ObjOptions {
                lines: Lines::None,
                ..Default::default()
            },
        );
        let rm = grid.merge_dual_from_obj_string(&dual_only).expect("merge");
        assert_eq!(rm.faces_matched, rc.faces_matched);
        assert_eq!(rm.faces_dropped, rc.faces_dropped);
        assert_eq!(rm.edges, rc.edges);
        assert_eq!(edge_count(&grid), edge_count(&combined));
        for f in &combined.dual_faces {
            let m = grid
                .dual_face(Index::fake(f.a), Index::fake(f.b))
                .expect("merged face");
            assert!(same_vertex_set(&m.vertices, &f.vertices, 1e-12));
        }
    }

    #[test]
    fn legacy_fallback_without_dual() {
        let points = vec![
            Pos([0.0, 0.0, 0.0]),
            Pos([1.0, 0.0, 0.0]),
            Pos([1.0, 1.0, 1.0]),
        ];
        let (g, _) = VineyardsGridMesh::assemble(points, vec![(0, 1), (0, 2)], Vec::new()).unwrap();
        assert!(!g.has_dual());
        assert_eq!(g.dual_face_points(fake(0), fake(1)).len(), 4);
        assert!(g.dual_quad_points(fake(0), fake(2)).is_none());
        assert!(g.dual_face_points(fake(0), fake(2)).is_empty());
    }

    #[test]
    fn edge_without_a_face_in_a_grid_with_dual_yields_nothing() {
        let block = lattice_block(Lattice::Bcc, [2, 2, 2], 1.0, Pos([0.0; 3]));
        let (g, _) = read(&block_to_obj(&block, &ObjOptions::default()));
        assert!(g.has_dual());
        // Two block corners are not neighbours and share no wall.
        let corner = |x: f64, y: f64, z: f64| {
            g.points
                .iter()
                .position(|p| p.dist(&Pos([x, y, z])) < 1e-9)
                .expect("corner")
        };
        let (a, b) = (corner(-1.0, -1.0, -1.0), corner(1.0, 1.0, 1.0));
        assert!(!g.neighbors[a].contains(&(b as isize)));
        assert!(g.dual_face(fake(a), fake(b)).is_none());
        assert!(g.dual_face_points(fake(a), fake(b)).is_empty());
    }

    #[test]
    fn culled_block_keeps_only_walls_between_kept_points() {
        // Keep the points inside a sphere: the walls to culled points become outer walls.
        let block = lattice_block_culled(Lattice::Bcc, [4, 4, 4], 1.0, Pos([0.0; 3]), |p| {
            p.norm() < 1.7
        });
        assert!(block.points.len() < 125 + 64 && block.points.len() > 20);
        assert!(block.outer_walls() > 0);
        let (g, r) = read(&block_to_obj(&block, &ObjOptions::default()));
        assert_eq!(g.points.len(), block.points.len());
        assert_eq!(r.faces_total, block.walls.len());
        assert_eq!(r.faces_matched, block.inner_walls());
        assert_eq!(r.faces_dropped, block.outer_walls());
        assert_eq!(r.edges_without_face, 0);
        assert_eq!(r.faces_without_line_edge, 0);
        assert_eq!(edge_count(&g), block.shell1.len() + block.shell2.len());
        // Every matched face joins two kept points that are lattice neighbours.
        for f in g.dual_faces() {
            let d = g.points[f.a as usize].dist(&g.points[f.b as usize]);
            assert!((d - 1.0).abs() < 1e-9 || (d - 3f64.sqrt() / 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn dual_in_the_wrong_frame_matches_nothing_and_finishes_quickly() {
        let block = lattice_block(Lattice::Bcc, [6, 6, 6], 1.0, Pos([0.0; 3]));
        let (mut grid, _) = read(&block_to_obj(
            &block,
            &ObjOptions {
                dual: false,
                ..Default::default()
            },
        ));
        // The same dual, but as exported with different axes: rotated a quarter turn about z.
        let mut rotated = lattice_block(Lattice::Bcc, [6, 6, 6], 1.0, Pos([0.0; 3]));
        for v in &mut rotated.dual_vertices {
            *v = Pos([-v.y() + 0.37, v.x() + 0.11, v.z() + 0.23]);
        }
        let dual = block_to_obj(
            &rotated,
            &ObjOptions {
                lines: Lines::None,
                ..Default::default()
            },
        );
        let start = std::time::Instant::now();
        let r = grid.merge_dual_from_obj_string(&dual).expect("a mismatch is not a parse error");
        assert_eq!(r.faces_matched, 0);
        assert_eq!(r.faces_dropped, r.faces_total);
        assert!(start.elapsed().as_secs() < 5, "took {:?}", start.elapsed());
        // The grid still has its points and edges.
        assert_eq!(grid.points.len(), block.points.len());
        assert_eq!(edge_count(&grid), block.shell1.len() + block.shell2.len());
    }

    #[test]
    fn non_finite_coordinates_are_rejected() {
        for bad in ["nan", "inf", "-inf"] {
            let s = format!("o grid\nv 0 0 0\nv 1 0 0\nv {} 0 0\nl 1 2\nl 2 3\n", bad);
            let err = VineyardsGridMesh::read_from_obj_string(&s).err().expect(bad);
            assert!(err.contains("non-finite"), "{}", err);
        }
    }

    #[test]
    fn object_with_faces_and_edges_is_rejected() {
        // Lattice and Voronoi merged into one object (Blender "Objects as OBJ Groups", or a
        // lattice exported with Delaunay faces on).
        let block = lattice_block(Lattice::Sc, [2, 2, 2], 1.0, Pos([0.0; 3]));
        let opts = ObjOptions {
            lines: Lines::Shell1, // SC's second shell (face diagonals) shares no wall
            ..Default::default()
        };
        let s = block_to_obj(&block, &opts).replace("o SC_voronoi\n", "");
        let err = VineyardsGridMesh::read_from_obj_string(&s).err().expect("must be rejected");
        assert!(err.contains("both") && err.contains("faces"), "{}", err);
        // With `g` groups instead of `o` objects it parses like before.
        let s = block_to_obj(&block, &opts).replace("\no ", "\ng ");
        let (_, r) = read(&s);
        assert_walls(&r, &block, &[(4, block.inner_walls())]);
    }

    #[test]
    fn two_nearest_handles_small_sets_far_points_and_the_brute_force_fallback() {
        // Two points: always those two.
        let two = [Pos([0.0; 3]), Pos([1.0, 0.0, 0.0])];
        let loc = PointLocator::new(&two);
        assert_eq!(loc.two_nearest(Pos([0.9, 5.0, -3.0])), Some((1, 0)));

        // A small cubic block: the shell scan cannot conclude for a point just outside the
        // block, so the brute-force fallback runs.  It must return two distinct points, the
        // nearest first.
        let pts: Vec<Pos> = (0..4)
            .flat_map(|i| (0..4).flat_map(move |j| (0..4).map(move |k| Pos([i as f64, j as f64, k as f64]))))
            .collect();
        let loc = PointLocator::new(&pts);
        let (a, b) = loc.two_nearest(Pos([-0.5, 0.0, 0.0])).unwrap();
        assert_ne!(a, b);
        assert!(pts[a].dist(&Pos([0.0; 3])) < 1e-12, "nearest is the corner");
        // second nearest: (0,1,0) or (0,0,1), at sqrt(0.25 + 1)
        assert!((pts[b].dist(&Pos([-0.5, 0.0, 0.0])) - 1.25f64.sqrt()).abs() < 1e-12);
        for c in [Pos([50.0, -20.0, 3.0]), Pos([1.5, 1.5, 1.5]), Pos([0.25, 3.0, 3.0])] {
            let (a, b) = loc.two_nearest(c).unwrap();
            assert_ne!(a, b);
            let mut sorted: Vec<f64> = pts.iter().map(|p| p.dist(&c)).collect();
            sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            assert!((pts[a].dist(&c) - sorted[0]).abs() < 1e-12);
            assert!((pts[b].dist(&c) - sorted[1]).abs() < 1e-12);
        }

        // Collinear points.
        let line: Vec<Pos> = (0..40).map(|i| Pos([i as f64 * 0.1, 0.0, 0.0])).collect();
        let loc = PointLocator::new(&line);
        let (a, b) = loc.two_nearest(Pos([2.04, 0.3, 0.0])).unwrap();
        assert_eq!((a.min(b), a.max(b)), (20, 21));
    }

    #[test]
    fn same_vertex_set_is_a_multiset_comparison() {
        let (a, b, c, d) = (
            Pos([0.0; 3]),
            Pos([1.0, 0.0, 0.0]),
            Pos([0.0, 1.0, 0.0]),
            Pos([0.0, 0.0, 1.0]),
        );
        assert!(same_vertex_set(&[a, b, c], &[c, a, b], 1e-9));
        assert!(!same_vertex_set(&[a, a, b, c], &[a, b, c, d], 1e-9));
        assert!(!same_vertex_set(&[a, b, c, d], &[a, a, b, c], 1e-9));
        let deduped = dedupe_vertices(&[a, b, a, c], 1e-9);
        assert_eq!(deduped.len(), 3);
        assert!(same_vertex_set(&deduped, &[a, b, c], 1e-9));
    }

    #[test]
    fn state_files_without_dual_faces_still_load() {
        // The struct as it was serialized before `dual_faces` existed.
        #[derive(serde::Serialize)]
        struct V0 {
            points: Vec<Pos>,
            neighbors: Vec<Vec<isize>>,
            r#type: String,
        }
        let v0 = V0 {
            points: vec![Pos([0.0, 0.0, 0.0]), Pos([1.0, 0.0, 0.0])],
            neighbors: vec![vec![1], vec![0]],
            r#type: "meshgrid".to_string(),
        };
        let bytes = rmp_serde::to_vec(&v0).unwrap();
        let g: VineyardsGridMesh = rmp_serde::from_slice(&bytes).expect("legacy layout loads");
        assert_eq!(g.points.len(), 2);
        assert_eq!(g.neighbors, vec![vec![1], vec![0]]);
        assert!(g.dual_faces.is_empty());

        // And the new layout round-trips through msgpack and JSON.
        let block = lattice_block(Lattice::Bcc, [1, 1, 1], 1.0, Pos([0.0; 3]));
        let (g, _) = read(&block_to_obj(&block, &ObjOptions::default()));
        let g2: VineyardsGridMesh = rmp_serde::from_slice(&rmp_serde::to_vec(&g).unwrap()).unwrap();
        let g3: VineyardsGridMesh =
            serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(g2.dual_faces.len(), g.dual_faces.len());
        assert_eq!(g3.dual_faces.len(), g.dual_faces.len());
        let f = &g.dual_faces[0];
        assert!(g2.dual_face(Index::fake(f.a), Index::fake(f.b)).is_some());
        assert!(g3.dual_face(Index::fake(f.a), Index::fake(f.b)).is_some());
    }

    #[test]
    fn write_as_obj_round_trips_the_dual() {
        let block = lattice_block(Lattice::Fcc, [2, 2, 2], 0.5, Pos([0.0; 3]));
        let (g, r) = read(&block_to_obj(&block, &ObjOptions::default()));
        let mut out = Vec::new();
        g.write_as_obj(&mut out).unwrap();
        let (g2, r2) = read(std::str::from_utf8(&out).unwrap());
        assert_eq!(g2.points.len(), g.points.len());
        assert_eq!(edge_count(&g2), edge_count(&g));
        assert_eq!(r2.faces_matched, r.faces_matched);
        assert_eq!(r2.faces_dropped, 0, "only matched faces were written");
    }
}
