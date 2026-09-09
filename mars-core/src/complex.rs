use std::collections::{HashMap, HashSet};

use crate::{sneaky_matrix::CI, SneakyMatrix};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Pos(pub [f64; 3]);

impl Pos {
    pub fn x(&self) -> f64 {
        self.0[0]
    }

    pub fn y(&self) -> f64 {
        self.0[1]
    }

    pub fn z(&self) -> f64 {
        self.0[2]
    }

    pub fn dist2(&self, other: &Pos) -> f64 {
        let mut sum = 0.0;
        for i in 0..3 {
            sum += (self.0[i] - other.0[i]).powi(2);
        }
        sum
    }

    pub fn dist(&self, other: &Pos) -> f64 {
        self.dist2(other).sqrt()
    }

    pub fn dot(&self, other: &Pos) -> f64 {
        self.0[0] * other.0[0] + self.0[1] * other.0[1] + self.0[2] * other.0[2]
    }

    pub fn cross(&self, other: &Pos) -> Pos {
        let [ax, ay, az] = self.0;
        let [bx, by, bz] = other.0;
        Pos([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
    }
}

impl std::fmt::Debug for Pos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "[{}, {}, {}]",
            self.0[0], self.0[1], self.0[2]
        ))
    }
}

impl std::ops::Add for Pos {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut arr = [0.0; 3];
        for i in 0..3 {
            arr[i] = self.0[i] + rhs.0[i];
        }
        Pos(arr)
    }
}

impl std::ops::Sub for Pos {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut arr = [0.0; 3];
        for i in 0..3 {
            arr[i] = self.0[i] - rhs.0[i];
        }
        Pos(arr)
    }
}

impl std::ops::Mul<f64> for Pos {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut arr = [0.0; 3];
        for i in 0..3 {
            arr[i] = self.0[i] * rhs;
        }
        Pos(arr)
    }
}

impl std::ops::Div<f64> for Pos {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        let mut arr = [0.0; 3];
        for i in 0..3 {
            arr[i] = self.0[i] / rhs;
        }
        Pos(arr)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Simplex {
    /// Unique identifier of the simplex.  This is only unique within the dimension for the complex it is in.
    pub id: CI,
    /// Coordinates of the simplex, if any.
    pub coords: Option<Pos>,
    /// The boundary of the simplex, i.e. the indices of the faces.
    pub boundary: Vec<CI>,
}

impl Simplex {
    pub fn dim(&self) -> isize {
        self.boundary.len() as isize - 1
    }

    /// The center point is:
    /// - point: itself
    /// - edge: midpoint of the edge
    /// - face: centroid of the face
    pub fn center_point(&self, complex: &Complex) -> Pos {
        match self.dim() {
            0 => self.coords.unwrap(),
            1 => {
                let a = &complex.simplices_per_dim[0][self.boundary[0] as usize];
                let b = &complex.simplices_per_dim[0][self.boundary[1] as usize];
                (a.coords.unwrap() + b.coords.unwrap()) * 0.5
            }
            2 => {
                let a = &complex.simplices_per_dim[1][self.boundary[0] as usize];
                let b = &complex.simplices_per_dim[1][self.boundary[1] as usize];
                let c = &complex.simplices_per_dim[1][self.boundary[2] as usize];
                (a.center_point(complex) + b.center_point(complex) + c.center_point(complex)) / 3.0
            }
            _ => panic!("Missing arms for simplex of dimension {}", self.dim()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Complex {
    pub simplices_per_dim: Vec<Vec<Simplex>>,
}

impl Complex {
    pub fn num_simplices_of_dim(&self, dim: isize) -> usize {
        if dim == -1 {
            return 1;
        }
        self.simplices_per_dim[dim as usize].len()
    }

    /// Returns a [SneakyMatrix] of the boundary map from dimension `dim` to `dim - 1`.
    pub fn boundary_matrix(&self, dim: isize) -> SneakyMatrix {
        assert!(0 <= dim);
        let n = self.num_simplices_of_dim(dim);
        let m = self.num_simplices_of_dim(dim - 1);
        assert!(
            n < CI::MAX as usize,
            "Too many simplices of dim {}: {} < {}",
            dim,
            n,
            CI::MAX
        );
        assert!(
            m < CI::MAX as usize,
            "Too many simplices of dim {}: {} < {}",
            dim - 1,
            m,
            CI::MAX
        );
        let mut sm = SneakyMatrix::zeros(m as CI, n as CI);
        for s in self.simplices_per_dim[dim as usize].iter() {
            for j in &s.boundary {
                sm.set((*j) as CI, s.id, true);
            }
        }
        sm
    }

    /// Return the vertex indices for each triangle. Sorts the indices, so any
    /// ordering information of the edges is lost.
    pub fn triangle_indices(&self) -> Vec<[CI; 3]> {
        let mut tris = Vec::new();
        for t in &self.simplices_per_dim[2] {
            let mut s = HashSet::new();
            for &e in &t.boundary {
                let e = &self.simplices_per_dim[1][e as usize];
                s.insert(e.boundary[0]);
                s.insert(e.boundary[1]);
            }
            let mut v = s.into_iter().collect::<Vec<_>>();
            v.sort();
            assert!(v.len() == 3, "A triangle should be three vertices");
            tris.push([v[0], v[1], v[2]]);
        }
        tris
    }

    pub fn read_from_obj_path<P: AsRef<std::path::Path>>(p: P) -> Result<Self, String> {
        let input_str =
            std::fs::read_to_string(p).map_err(|e| format!("Error reading file: {}", e))?;
        Self::read_from_obj_string(&input_str)
    }

    pub fn read_from_obj_string(input_str: &str) -> Result<Self, String> {
        let mut vertices: Vec<Simplex> = Vec::new();
        let mut edges: Vec<Simplex> = Vec::new();
        let mut triangles: Vec<Simplex> = Vec::new();

        // Map (i, j) to edge simplex index.
        let mut edge_map = HashMap::<(CI, CI), CI>::new();

        for line in input_str.lines() {
            let line = line.trim();
            if line.starts_with("#")
                || line.starts_with("mtllib")
                || line.starts_with("o")
                || line.starts_with("s")
            {
                continue;
            } else if line.starts_with("v") {
                // v -0.039375 1.021144 0.000000
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
                assert!(
                    vertices.len() <= CI::MAX as usize,
                    "Too many vertices in input complex (got {}, max {})",
                    vertices.len(),
                    CI::MAX
                );
                vertices.push(Simplex {
                    id: vertices.len() as CI,
                    coords: Some(coords),
                    boundary: vec![0],
                });
            } else if line.starts_with("l") {
                // l 1 2
                let groups = line.split_ascii_whitespace().collect::<Vec<_>>();
                if groups.len() != 3 {
                    return Err("An edge should have two vertices".into());
                }
                let a = groups
                    .get(1)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<CI>().map_err(|e| e.to_string()))?
                    - 1; // NOTE: .obj is 1-indexed

                let b = groups
                    .get(2)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<CI>().map_err(|e| e.to_string()))?
                    - 1; // NOTE: .obj is 1-indexed

                assert!(
                    edges.len() <= CI::MAX as usize,
                    "Too many edges in input complex (got {}, max {})",
                    edges.len(),
                    CI::MAX
                );
                let id = edges.len() as CI;

                let (a, b) = (a.min(b), a.max(b));

                if edge_map.contains_key(&(a, b)) {
                    panic!("Duplicate edge entries in the .obj");
                }

                edge_map.insert((a, b), id);

                edges.push(Simplex {
                    id,
                    coords: None,
                    boundary: vec![a, b],
                });
            } else if line.starts_with("f") {
                // f 20 27 19
                let groups = line.split_ascii_whitespace().collect::<Vec<_>>();

                if groups.len() != 4 {
                    return Err("A triangle should have three vertices".into());
                }
                let a = groups
                    .get(1)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<CI>().map_err(|e| e.to_string()))?
                    - 1;
                let b = groups
                    .get(2)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<CI>().map_err(|e| e.to_string()))?
                    - 1;
                let c = groups
                    .get(3)
                    .ok_or("missing field".to_string())
                    .and_then(|n| n.parse::<CI>().map_err(|e| e.to_string()))?
                    - 1;

                assert!(
                    triangles.len() <= CI::MAX as usize,
                    "Too many triangles in input complex (got {}, max {})",
                    triangles.len(),
                    CI::MAX
                );
                triangles.push(Simplex {
                    id: triangles.len() as CI,
                    coords: None,
                    boundary: vec![a, b, c], // NOTE: we insert vertex indices here, and fix them up later.
                });
            }
        }

        // Check that no two vertices are actually the same vertex
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let p = vertices[i].coords.ok_or("Vertex should have coordinates")?;
                let q = vertices[j].coords.ok_or("Vertex should have coordinates")?;
                let dist = p.dist(&q);
                if dist < 1e-5 {
                    return Err(format!(
                        "Two vertices are too close together: {} and {}",
                        i, j
                    ));
                }
            }
        }

        // Replace vertex indices with the correct edge indices.
        // If the edge does not exist, create it.
        for tri in triangles.iter_mut() {
            let bnd = tri.boundary.clone();
            for i in 0..3 {
                let a = bnd[i];
                let b = bnd[(i + 1) % 3];
                let (a, b) = (a.min(b), a.max(b));
                let id = edge_map
                    .entry((a, b))
                    .or_insert_with(|| {
                        let id = edges.len() as CI;
                        edges.push(Simplex {
                            id,
                            coords: None,
                            boundary: vec![a, b],
                        });
                        id
                    })
                    .clone();
                tri.boundary[i] = id;
            }
        }

        Ok(Self {
            simplices_per_dim: vec![vertices, edges, triangles],
        })
    }

    pub fn write_as_obj<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        writeln!(w, "o complex")?;

        for s in &self.simplices_per_dim[0] {
            let c = s.coords.expect("simplex of dim=0 should have coords set");
            writeln!(w, "v {} {} {}", c.x(), c.y(), c.z())?;
        }

        for s in &self.simplices_per_dim[2] {
            let mut vx = Vec::new();
            for ei in &s.boundary {
                vx.extend_from_slice(&self.simplices_per_dim[1][*ei as usize].boundary);
            }
            vx.sort();
            vx.dedup();
            assert_eq!(vx.len(), 3, "a face should have =3 simplex indices");
            writeln!(w, "f {} {} {}", vx[0] + 1, vx[1] + 1, vx[2] + 1)?;
        }

        Ok(())
    }

    /// Compute the distances from a given point to all vertices, edges, and faces.
    ///
    /// The distance to a point is the squared euclidean distance.
    ///
    /// The distance to a higher simplex is the largest of the distances to its faces.
    pub fn distances_to(&self, key_point: Pos) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let vertex_distances = self.simplices_per_dim[0]
            .iter()
            .map(|v| v.coords.unwrap().dist2(&key_point))
            .collect::<Vec<_>>();

        let edge_distances = self.simplices_per_dim[1]
            .iter()
            .map(|e| {
                vertex_distances[e.boundary[0] as usize]
                    .max(vertex_distances[e.boundary[1] as usize])
            })
            .collect::<Vec<_>>();

        let triangle_distances = self.simplices_per_dim[2]
            .iter()
            .map(|f| {
                edge_distances[f.boundary[0] as usize]
                    .max(edge_distances[f.boundary[1] as usize])
                    .max(edge_distances[f.boundary[2] as usize])
            })
            .collect::<Vec<_>>();

        (vertex_distances, edge_distances, triangle_distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_obj() {
        let path = "../web/inputs/cube-subdiv-2.obj";
        let c = Complex::read_from_obj_path(path).unwrap();

        let num_verts = c.num_simplices_of_dim(0) as CI;
        for e in &c.simplices_per_dim[1] {
            for &vi in &e.boundary {
                assert!(vi < num_verts);
            }
        }

        let num_edges = c.num_simplices_of_dim(1) as CI;
        for t in &c.simplices_per_dim[2] {
            for &vi in &t.boundary {
                assert!(vi < num_edges);
            }
        }
    }

    #[test]
    fn boundary_matrix() {
        let complex = crate::test::test_complex_cube();
        for dim in 0..3 {
            let boundary = complex.boundary_matrix(dim);
            insta::assert_snapshot!(boundary.__str__());
        }
    }
}
