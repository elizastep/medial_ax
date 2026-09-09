use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use crate::*;

// =============================================================================================
// Cubic lattices with their Voronoi tessellation, mirroring `blender_scripts/cubic_lattices.py`.
//
// Two exact integer coordinate systems: Y = 2 x fractional cell coordinates for lattice points,
// Z = 4 x fractional for Voronoi vertices.  Real coordinates are (a/2) Y = (a/4) Z, shifted so
// the block is centred where asked.
// =============================================================================================

/// Which cubic lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lattice {
    Sc,
    Bcc,
    Fcc,
}

impl Lattice {
    pub fn name(self) -> &'static str {
        match self {
            Lattice::Sc => "SC",
            Lattice::Bcc => "BCC",
            Lattice::Fcc => "FCC",
        }
    }

    /// Basis offsets of the conventional cell, in Y coordinates.
    fn basis_y(self) -> &'static [[i64; 3]] {
        match self {
            Lattice::Sc => &[[0, 0, 0]],
            Lattice::Bcc => &[[0, 0, 0], [1, 1, 1]],
            Lattice::Fcc => &[[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        }
    }

    /// Vertices of the Voronoi cell around the origin, in Z coordinates.
    /// SC: cube (8), BCC: truncated octahedron (24), FCC: rhombic dodecahedron (14).
    fn voronoi_vertices_z(self) -> Vec<[i64; 3]> {
        match self {
            Lattice::Sc => orbit([2, 2, 2]),
            Lattice::Bcc => orbit([0, 1, 2]),
            Lattice::Fcc => {
                let mut v = orbit([2, 0, 0]);
                v.extend(orbit([1, 1, 1]));
                v
            }
        }
    }

    /// Lattice vectors whose bisector planes carry a facet, in Z coordinates.
    fn relevant_z(self) -> Vec<[i64; 3]> {
        match self {
            Lattice::Sc => orbit([4, 0, 0]),
            Lattice::Bcc => {
                let mut v = orbit([2, 2, 2]);
                v.extend(orbit([4, 0, 0]));
                v
            }
            Lattice::Fcc => orbit([2, 2, 0]),
        }
    }

    /// Offsets to the first and second coordination shells, in Y coordinates.
    fn shell_offsets_y(self) -> (Vec<[i64; 3]>, Vec<[i64; 3]>) {
        match self {
            Lattice::Sc => (orbit([2, 0, 0]), orbit([2, 2, 0])),
            Lattice::Bcc => (orbit([1, 1, 1]), orbit([2, 0, 0])),
            Lattice::Fcc => (orbit([1, 1, 0]), orbit([2, 0, 0])),
        }
    }
}

/// All coordinate permutations of `v` with all sign choices, deduplicated and sorted.
fn orbit(v: [i64; 3]) -> Vec<[i64; 3]> {
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut out = BTreeSet::new();
    for p in perms {
        for s in 0..8u32 {
            let sign = |k: u32| if (s >> k) & 1 == 1 { -1 } else { 1 };
            out.insert([sign(0) * v[p[0]], sign(1) * v[p[1]], sign(2) * v[p[2]]]);
        }
    }
    out.into_iter().collect()
}

fn to_pos(v: [i64; 3]) -> Pos {
    Pos([v[0] as f64, v[1] as f64, v[2] as f64])
}

/// Facets of the Voronoi cell: `(relevant vector, vertex ids in cyclic order, outward)`.
fn voronoi_facets(kind: Lattice) -> (Vec<[i64; 3]>, Vec<([i64; 3], Vec<usize>)>) {
    let verts = kind.voronoi_vertices_z();
    let mut facets = Vec::new();
    for r in kind.relevant_z() {
        let rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let mut on: Vec<usize> = (0..verts.len())
            .filter(|&i| 2 * (verts[i][0] * r[0] + verts[i][1] * r[1] + verts[i][2] * r[2]) == rr)
            .collect();
        if on.len() < 3 {
            continue; // not a facet
        }
        let mut c = Pos([0.0; 3]);
        for &i in &on {
            c = c + to_pos(verts[i]);
        }
        let c = c / on.len() as f64;
        let n = to_pos(r) / to_pos(r).norm();
        let u = to_pos(verts[on[0]]) - c;
        let u = u / u.norm();
        let w = n.cross(&u);
        let angle = |i: usize| {
            let d = to_pos(verts[i]) - c;
            d.dot(&w).atan2(d.dot(&u))
        };
        on.sort_by(|&i, &j| angle(i).partial_cmp(&angle(j)).unwrap());
        let (a, b, c3) = (
            to_pos(verts[on[0]]),
            to_pos(verts[on[1]]),
            to_pos(verts[on[2]]),
        );
        if (b - a).cross(&(c3 - a)).dot(&to_pos(r)) < 0.0 {
            on.reverse();
        }
        facets.push((r, on));
    }
    (verts, facets)
}

/// A rectangular block of a cubic lattice together with its Voronoi tessellation.
pub struct LatticeBlock {
    pub kind: Lattice,
    pub points: Vec<Pos>,
    /// Nearest-neighbour edges (`i < j`).
    pub shell1: Vec<(usize, usize)>,
    /// Second-shell edges of length `a` (`i < j`); for BCC these are the six axis neighbours.
    pub shell2: Vec<(usize, usize)>,
    pub dual_vertices: Vec<Pos>,
    /// Every Voronoi wall once: `(site, other site or None for an outer wall, dual vertex ids)`.
    pub walls: Vec<(usize, Option<usize>, Vec<usize>)>,
}

impl LatticeBlock {
    pub fn inner_walls(&self) -> usize {
        self.walls.iter().filter(|w| w.1.is_some()).count()
    }
    pub fn outer_walls(&self) -> usize {
        self.walls.len() - self.inner_walls()
    }
}

/// `n = [nx, ny, nz]` conventional cells of side `a`, block centred at `centre`.  The block is
/// closed: points on its far faces are included.
pub fn lattice_block(kind: Lattice, n: [usize; 3], a: f64, centre: Pos) -> LatticeBlock {
    let lim = [2 * n[0] as i64, 2 * n[1] as i64, 2 * n[2] as i64];
    let mut ys: Vec<[i64; 3]> = Vec::new();
    for i in 0..=n[0] as i64 {
        for j in 0..=n[1] as i64 {
            for k in 0..=n[2] as i64 {
                for off in kind.basis_y() {
                    let p = [2 * i + off[0], 2 * j + off[1], 2 * k + off[2]];
                    if p[0] <= lim[0] && p[1] <= lim[1] && p[2] <= lim[2] {
                        ys.push(p);
                    }
                }
            }
        }
    }
    let index: HashMap<[i64; 3], usize> = ys.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let offset = Pos([
        n[0] as f64 * a / 2.0,
        n[1] as f64 * a / 2.0,
        n[2] as f64 * a / 2.0,
    ]);
    let real_y = |p: [i64; 3]| to_pos(p) * (a / 2.0) - offset + centre;
    let real_z = |k: [i64; 3]| to_pos(k) * (a / 4.0) - offset + centre;

    let points: Vec<Pos> = ys.iter().map(|&p| real_y(p)).collect();

    let (s1, s2) = kind.shell_offsets_y();
    let pairs = |offs: &[[i64; 3]]| {
        let mut v = Vec::new();
        for (i, p) in ys.iter().enumerate() {
            for o in offs {
                if let Some(&j) = index.get(&[p[0] + o[0], p[1] + o[1], p[2] + o[2]]) {
                    if i < j {
                        v.push((i, j));
                    }
                }
            }
        }
        v.sort();
        v
    };
    let shell1 = pairs(&s1);
    let shell2 = pairs(&s2);

    let (vz, facets) = voronoi_facets(kind);
    let mut dual_vertices: Vec<Pos> = Vec::new();
    let mut key: HashMap<[i64; 3], usize> = HashMap::new();
    let mut walls = Vec::new();
    for (i, p) in ys.iter().enumerate() {
        let pz = [2 * p[0], 2 * p[1], 2 * p[2]];
        for (r, face) in &facets {
            let q = [p[0] + r[0] / 2, p[1] + r[1] / 2, p[2] + r[2] / 2];
            let j = index.get(&q).copied();
            if matches!(j, Some(j) if j < i) {
                continue; // emitted from the other side
            }
            let ids = face
                .iter()
                .map(|&vi| {
                    let k = [pz[0] + vz[vi][0], pz[1] + vz[vi][1], pz[2] + vz[vi][2]];
                    match key.get(&k) {
                        Some(&id) => id,
                        None => {
                            let id = dual_vertices.len();
                            dual_vertices.push(real_z(k));
                            key.insert(k, id);
                            id
                        }
                    }
                })
                .collect();
            walls.push((i, j, ids));
        }
    }

    LatticeBlock {
        kind,
        points,
        shell1,
        shell2,
        dual_vertices,
        walls,
    }
}

/// Which `l` lines to write.
#[derive(Clone, Copy, Debug)]
pub enum Lines {
    None,
    Shell1,
    Both,
}

/// How to write a [LatticeBlock] as an `.obj`, mimicking what Blender exports.
pub struct ObjOptions {
    pub lines: Lines,
    /// Write the dual object at all.
    pub dual: bool,
    /// Include the outer walls of the boundary cells.
    pub include_outer: bool,
    /// Write `vn` lines and `f a//n` syntax, as Blender does for objects with faces.
    pub normals: bool,
    /// Round coordinates like Blender's exporter.
    pub decimals: Option<usize>,
    /// Write every wall twice (once per cell), as Cell Fracture would.
    pub duplicate_walls: bool,
    /// Fan-triangulate the walls (a mistake we want to detect).
    pub triangulate: bool,
}

impl Default for ObjOptions {
    fn default() -> Self {
        Self {
            lines: Lines::Both,
            dual: true,
            include_outer: true,
            normals: false,
            decimals: None,
            duplicate_walls: false,
            triangulate: false,
        }
    }
}

pub fn block_to_obj(block: &LatticeBlock, o: &ObjOptions) -> String {
    let fmt = |p: &Pos| match o.decimals {
        Some(d) => format!("v {:.*} {:.*} {:.*}\n", d, p.x(), d, p.y(), d, p.z()),
        None => format!("v {} {} {}\n", p.x(), p.y(), p.z()),
    };
    let mut s = String::from("# test lattice\n");
    s += &format!("o {}_lattice\n", block.kind.name());
    for p in &block.points {
        s += &fmt(p);
    }
    let edges: Vec<(usize, usize)> = match o.lines {
        Lines::None => Vec::new(),
        Lines::Shell1 => block.shell1.clone(),
        Lines::Both => block.shell1.iter().chain(&block.shell2).cloned().collect(),
    };
    for (i, j) in edges {
        s += &format!("l {} {}\n", i + 1, j + 1);
    }
    if o.dual {
        s += &format!("o {}_voronoi\n", block.kind.name());
        let base = block.points.len();
        for p in &block.dual_vertices {
            s += &fmt(p);
        }
        if o.normals {
            s += "vn 0 0 1\n";
        }
        let face_line = |ids: &[usize]| {
            let toks: Vec<String> = ids
                .iter()
                .map(|&v| {
                    if o.normals {
                        format!("{}//1", base + v + 1)
                    } else {
                        format!("{}", base + v + 1)
                    }
                })
                .collect();
            format!("f {}\n", toks.join(" "))
        };
        for (_, j, ids) in &block.walls {
            if j.is_none() && !o.include_outer {
                continue;
            }
            if o.triangulate {
                for k in 1..ids.len() - 1 {
                    s += &face_line(&[ids[0], ids[k], ids[k + 1]]);
                }
            } else {
                s += &face_line(ids);
                if o.duplicate_walls {
                    let mut rev = ids.clone();
                    rev.reverse();
                    s += &face_line(&rev);
                }
            }
        }
    }
    s
}

/// Construct a test [Complex]  of the cube.
pub fn test_complex_cube() -> Complex {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("test/cube.obj");
    Complex::read_from_obj_path(&d).expect("Failed to read test input")
}

pub fn test_grid_for_cube() -> grid::VineyardsGrid {
    grid::VineyardsGrid {
        corner: Pos([-0.5, -0.5, -0.5]),
        size: 0.4,
        shape: grid::Index([5, 5, 5]),
        r#type: "grd".to_string(),
    }
}

pub fn default_pruning_param(dim: usize) -> PruningParam {
    match dim {
        0 => default_pruning_param_dim0(),
        1 => default_pruning_param_dim1(),
        2 => default_pruning_param_dim2(),
        _ => panic!("Dimension too high: {}", dim),
    }
}

pub fn default_pruning_param_dim0() -> PruningParam {
    PruningParam {
        euclidean: true,
        euclidean_distance: Some(0.01),
        coface: true,
        face: false,
        persistence: false,
        persistence_threshold: None,
    }
}

pub fn default_pruning_param_dim1() -> PruningParam {
    PruningParam {
        euclidean: true,
        euclidean_distance: Some(0.01),
        coface: false,
        face: true,
        persistence: true,
        persistence_threshold: Some(0.01),
    }
}

pub fn default_pruning_param_dim2() -> PruningParam {
    PruningParam {
        euclidean: true,
        euclidean_distance: Some(0.01),
        coface: false,
        face: true,
        persistence: false,
        persistence_threshold: None,
    }
}
