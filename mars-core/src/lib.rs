#![allow(dead_code, unreachable_code, non_snake_case)]
use std::{
    collections::{HashMap, HashSet},
    iter::zip,
};

use complex::{Complex, Pos};
use grid::{Index, VineyardsGrid, VineyardsGridMesh};
use permutation::Permutation;
use serde::{Deserialize, Serialize};
use sneaky_matrix::{SneakyMatrix, CI};
use tracing::{info, warn};

pub mod complex;
pub mod grid;
pub mod json;
pub mod permutation;
pub mod sneaky_matrix;
pub mod stats;
#[cfg(test)]
pub mod test;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Mars {
    pub complex: Option<Complex>,
    pub grid: Option<Grid>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Grid {
    Regular(VineyardsGrid),
    Mesh(VineyardsGridMesh),
}

impl Mars {
    /// Load a complex from an .obj string into the state.
    pub fn load_from_obj_str(&mut self, obj_str: &str) -> Result<(), String> {
        info!("load_from_obj_str");
        let cplx = Complex::read_from_obj_string(obj_str)?;
        info!(
            "read complex #v={} #e={} #t={}",
            cplx.simplices_per_dim[0].len(),
            cplx.simplices_per_dim[1].len(),
            cplx.simplices_per_dim[2].len()
        );
        self.complex = Some(cplx);
        Ok(())
    }

    /// Load a mesh grid from an .obj string into the state.
    pub fn load_meshgrid_from_obj_str(&mut self, obj_str: &str) -> Result<(), String> {
        self.grid = Some(Grid::Mesh(VineyardsGridMesh::read_from_obj_string(
            obj_str,
        )?));
        Ok(())
    }

    /// Return [Err] if we don't have a grid.
    pub fn split_into_4(&self) -> Result<[SubMars; 4], String> {
        let Some(ref g) = self.grid else {
            return Err("Mars::split_into_4: missing grid".to_string());
        };

        match g {
            Grid::Regular(grid) => {
                let (a, b, b_offset) = grid.split_with_overlap();

                let (aa, ab, ab_offset) = a.split_with_overlap();
                let (ba, bb, bb_offset) = b.split_with_overlap();

                let ret = [
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Regular(aa)),
                        },
                        offset: Index([0; 3]),
                    },
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Regular(ab)),
                        },
                        offset: ab_offset,
                    },
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Regular(ba)),
                        },
                        offset: b_offset,
                    },
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Regular(bb)),
                        },
                        offset: b_offset + bb_offset,
                    },
                ];
                Ok(ret)
            }
            Grid::Mesh(grid) => {
                let (a, b) = grid.split_in_half();
                let (aa, ab) = a.split_in_half();
                let (ba, bb) = b.split_in_half();
                let fake = Index::fake(0);
                let ret = [
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Mesh(aa)),
                        },
                        offset: fake,
                    },
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Mesh(ab)),
                        },
                        offset: fake,
                    },
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Mesh(ba)),
                        },
                        offset: fake,
                    },
                    SubMars {
                        mars: Mars {
                            complex: self.complex.clone(),
                            grid: Some(Grid::Mesh(bb)),
                        },
                        offset: fake,
                    },
                ];
                Ok(ret)
            }
        }
    }

    /// Run Vineyards across the instance.
    pub fn run<F: Fn(usize, usize)>(&self, progress: F) -> Result<Vineyards, String> {
        let Some(ref c) = self.complex else {
            return Err("Vineyards::run: no complex")?;
        };

        let Some(ref g) = self.grid else {
            return Err("Vineyards::run: no grid")?;
        };

        let (reductions, all_swaps) = match g {
            Grid::Regular(r) => {
                let i0 = Index([0; 3]);
                let p = r.coordinate(i0);
                let s0 = reduce_from_scratch(&c, p, false);
                r.run_vineyards_in_grid(c, i0, s0, false, progress)
            }
            Grid::Mesh(m) => m.run_vineyards(&c, false, progress),
        };

        // Split up the big list of swaps into one list per dimension, since this is always the
        // format in which we use it.  It would be better to have the mesh functions return this
        // data already, since it knows the dimensions of each swap.
        fn filter_dim(v: &[(Index, Index, Swaps)], dim: usize) -> Vec<(Index, Index, Swaps)> {
            v.iter()
                .flat_map(|(i, j, s)| {
                    let v: Vec<_> = s.v.iter().filter(|s| s.dim == dim).cloned().collect();
                    (0 < v.len()).then(|| (*i, *j, Swaps { v }))
                })
                .collect()
        }

        let swaps = [
            filter_dim(&all_swaps, 0),
            filter_dim(&all_swaps, 1),
            filter_dim(&all_swaps, 2),
        ];

        Ok(Vineyards { reductions, swaps })
    }

    /// Run slim Vineyards across the instance.
    pub fn run_slim<F: Fn(usize, usize)>(
        &self,
        progress: F,
    ) -> Result<[Vec<(Index, Index, Vec<(Swap, f64, f64)>)>; 3], String> {
        let Some(ref c) = self.complex else {
            return Err("Vineyards::run: no complex")?;
        };

        let Some(ref g) = self.grid else {
            return Err("Vineyards::run: no grid")?;
        };

        let mut ret = [Vec::new(), Vec::new(), Vec::new()];

        match g {
            Grid::Regular(_) => {
                todo!();
            }
            Grid::Mesh(m) => {
                m.run_vineyards_slim(&c, false, progress, |from, to, from_red, to_red, swaps| {
                    let mut inner_swaps = [Vec::new(), Vec::new(), Vec::new()];
                    for s in swaps.v {
                        let from_p = from_red
                            .persistence(c, s.dim, s.i)
                            .map(|b| b.lifetime())
                            .expect(".persistence should always return Some");
                        let to_p = to_red
                            .persistence(c, s.dim, s.j)
                            .map(|b| b.lifetime())
                            .expect(".persistence should always return Some");

                        inner_swaps[s.dim as usize].push((s, from_p, to_p));
                    }
                    for dim in 0..3 {
                        ret[dim].push((from, to, std::mem::take(&mut inner_swaps[dim])));
                    }
                })
            }
        };

        Ok(ret)
    }
}

/// Output data from running the Vineyard algorithm for an entire domain.
#[derive(Debug, Serialize, Deserialize)]
pub struct Vineyards {
    /// A [Reduction] for every [Index] of the grid we ran on.
    pub reductions: HashMap<Index, Reduction>,

    /// Swaps for each dimension. This is a list of adjacent grid index pairs, together with
    /// another [Vec]  of [Swap] objects, for each pair of simplices that were swapped in a
    /// Faustian swap.  Empty pairs are not included.
    pub swaps: [Vec<(Index, Index, Swaps)>; 3],
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PruningParam {
    pub euclidean: bool,
    pub euclidean_distance: Option<f64>,
    pub coface: bool,
    pub face: bool,
    pub persistence: bool,
    pub persistence_threshold: Option<f64>,
}

impl Vineyards {
    /// Prune the swaps for a given dimension and return a new list of swaps.
    pub fn prune_dim<F: Fn(usize, usize)>(
        &self,
        dim: usize,
        params: &PruningParam,
        complex: &Complex,
        progress: F,
    ) -> Vec<(Index, Index, Swaps)> {
        let mut pruned = Vec::new();
        let n = self.swaps[dim].len();
        for (i, s) in self.swaps[dim].iter().enumerate() {
            progress(i, n);
            if s.2.v.len() == 0 {
                continue;
            }

            let mut dim_swaps = s.2.clone();

            if params.euclidean {
                if let Some(dist) = params.euclidean_distance {
                    dim_swaps.prune_euclidian(&complex, dist)
                } else {
                    warn!(
                        "params dim {}: euclidean was true but distance was None",
                        dim
                    );
                }
            }

            if params.face {
                dim_swaps.prune_common_face(&complex);
            }

            if params.coface {
                dim_swaps.prune_coboundary(&complex);
            }

            if params.persistence {
                if let Some(dist) = params.persistence_threshold {
                    let grid_index_a = s.0;
                    let reduction_at_a = self.reductions.get(&grid_index_a).unwrap();
                    let grid_index_b = s.1;
                    let reduction_at_b = self.reductions.get(&grid_index_b).unwrap();
                    dim_swaps.prune_persistence(&complex, reduction_at_a, reduction_at_b, dist)
                } else {
                    warn!(
                        "params dim {}: persistence was true but threshold was None",
                        dim
                    );
                }
            }

            if dim_swaps.v.is_empty() {
                continue;
            }

            pruned.push((s.0, s.1, dim_swaps));
        }

        pruned
    }

    /// Add the swaps from another [Vineyards] instance.  The indices of the other instance is
    /// assumed to already be in the same coordinate system as [Self].
    pub fn add_other(&mut self, mut other: Vineyards) {
        for (index, state) in other.reductions.into_iter() {
            self.reductions.entry(index).or_insert(state);
        }

        // Add swaps
        for dim in 0..3 {
            // Properly merge in swaps for existing grid index pairs so that we don't get
            // duplicates.
            let swaps = &mut self.swaps[dim];
            let ij_to_index = swaps
                .iter()
                .enumerate()
                .map(|(i, (a, b, _))| ((*a, *b), i))
                .collect::<HashMap<_, _>>();

            for (i, j, new_swaps) in std::mem::take(&mut other.swaps[dim]).into_iter() {
                if let Some(k) = ij_to_index.get(&(i, j)) {
                    swaps[*k].2.v.extend_from_slice(&new_swaps.v);
                } else {
                    swaps.push((i, j, new_swaps));
                }
            }
        }
    }
}

pub type SwapList = Vec<(Index, Index, Swaps)>;

/// Sub problems for a [Mars] instance.  This is just like a regular instance, except that we have
/// a offset for the grid which we need to map the swaps we compute here to the right "coordinate
/// system"  in the [Mars] instance from which the [SubMars] was created.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubMars {
    pub mars: Mars,
    /// Where this sub-problem is relative to the main problem it was derived from.
    pub offset: Index,
}

impl SubMars {
    /// Run Vineyards, and map the result swaps back to the original coorinate system of the [Mars]
    /// instance this [SubMars] instance came from.
    pub fn run<F: Fn(usize, usize)>(&self, progress: F) -> Result<Vineyards, String> {
        let inner = self.mars.run(progress)?;

        let reductions = inner
            .reductions
            .into_iter()
            .map(|(index, reduction)| (index + self.offset, reduction))
            .collect();

        let mut swaps = inner
            .swaps
            .into_iter()
            .map(|s| {
                s.into_iter()
                    .map(|(i, j, swaps)| (i + self.offset, j + self.offset, swaps))
                    .collect()
            })
            .collect::<Vec<_>>();

        let swaps2 = swaps.pop().unwrap();
        let swaps1 = swaps.pop().unwrap();
        let swaps0 = swaps.pop().unwrap();

        Ok(Vineyards {
            reductions,
            swaps: [swaps0, swaps1, swaps2],
        })
    }

    /// Run Vineyards, and map the result swaps back to the original coorinate system of the [Mars]
    /// instance this [SubMars] instance came from.
    pub fn run_slim<F: Fn(usize, usize)>(
        &self,
        progress: F,
    ) -> Result<[Vec<(Index, Index, Vec<(Swap, f64, f64)>)>; 3], String> {
        let mut swaps_all_dims = self.mars.run_slim(progress)?;

        for swaps in &mut swaps_all_dims {
            for s in swaps {
                s.0 += self.offset;
                s.1 += self.offset;
            }
        }

        Ok(swaps_all_dims)
    }
}

/// A single Faustian swap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Swap {
    /// Dimension in which the swap happened.
    pub dim: usize,
    /// Canonical index of the first simplex.
    pub i: CI,
    /// Canonical index of the second simplex.
    pub j: CI,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Swaps {
    pub v: Vec<Swap>,
}

impl Swaps {
    pub fn new(v: Vec<Swap>) -> Self {
        Self { v }
    }

    /// Remove all swaps that were done between simplices that are closer than
    /// `min_dist`.
    ///
    /// Useful for 0th MA.
    pub fn prune_euclidian(&mut self, complex: &Complex, min_dist: f64) {
        self.v.retain(|swap| {
            let c1 = complex.simplices_per_dim[swap.dim][swap.i as usize].center_point(complex);
            let c2 = complex.simplices_per_dim[swap.dim][swap.j as usize].center_point(complex);
            let dist = c1.dist2(&c2);
            min_dist < dist
        })
    }

    pub fn prune_common_face(&mut self, complex: &Complex) {
        let mut simp_to_vertices: HashMap<(usize, CI), HashSet<CI>> = HashMap::new();

        for dim in 1..3 {
            for (s, i) in complex.simplices_per_dim[dim].iter().zip(0..) {
                if dim == 1 {
                    let set = simp_to_vertices
                        .entry((dim, i))
                        .or_insert_with(|| HashSet::new());
                    for b in &s.boundary {
                        set.insert(*b);
                    }
                } else {
                    for b in &s.boundary {
                        let face_set = simp_to_vertices
                            .get(&(dim - 1, *b))
                            .expect("Should have inserted this simplex before")
                            .clone();
                        let set = simp_to_vertices
                            .entry((dim, i))
                            .or_insert_with(|| HashSet::new());
                        set.extend(face_set);
                    }
                }
            }
        }

        self.v.retain(|swap| {
            if swap.dim == 0 {
                return true;
            }
            let set_i = simp_to_vertices.get(&(swap.dim, swap.i)).unwrap();
            let set_j = simp_to_vertices.get(&(swap.dim, swap.j)).unwrap();
            let mut intersection = set_i.intersection(set_j);
            intersection.next() == None
        });
    }

    /// Remove all swaps that happen between simplices if there is a simplex
    /// with the two simplices in its boundary.
    pub fn prune_coboundary(&mut self, complex: &Complex) {
        // (dim, id) to [id].
        let mut coboundary: HashMap<(usize, CI), HashSet<CI>> = HashMap::new();

        for dim in 1..3 {
            for (s, parent_i) in complex.simplices_per_dim[dim].iter().zip(0..) {
                for face_i in &s.boundary {
                    let v = coboundary
                        .entry((dim - 1, *face_i))
                        .or_insert_with(HashSet::new);
                    v.insert(parent_i);
                }
            }
        }

        self.v.retain(|swap| {
            if swap.dim == 2 {
                return true;
            }

            let cob_i = coboundary.get(&(swap.dim, swap.i));
            let cob_j = coboundary.get(&(swap.dim, swap.j));

            if let (Some(cob_i), Some(cob_j)) = (cob_i, cob_j) {
                let mut intersection = cob_i.intersection(cob_j);
                intersection.next() == None
            } else {
                true
            }
        });
    }

    /// Remove all swaps that happened where the persistence of any of the
    /// simplices were less than `lifetime`.
    ///
    /// `lifetime` can for instance be `0.01`.
    ///
    /// Probably only useful for 1st MA.
    pub fn prune_persistence(
        &mut self,
        complex: &Complex,
        reduction_from: &Reduction,
        reduction_to: &Reduction,
        lifetime: f64,
    ) {
        self.v.retain(|swap| {
            let persistence_i = reduction_from.persistence(complex, swap.dim, swap.i);
            let persistence_j = reduction_to.persistence(complex, swap.dim, swap.j);
            match (persistence_i, persistence_j) {
                (Some(p), Some(q)) => {
                    if p.lifetime() < lifetime && q.lifetime() < lifetime {
                        false
                    } else {
                        true
                    }
                }
                _ => true,
            }
        });
    }

    pub fn pyclone(&self) -> Self {
        self.clone()
    }
}

pub mod prune {
    use std::collections::{HashMap, HashSet};

    use tracing::warn;

    use crate::{complex::Complex, grid::Index, sneaky_matrix::CI, PruningParam, Swap};

    type SlimSwap = Vec<(Swap, f64, f64)>;

    pub fn prune_dim<F: Fn(usize, usize)>(
        all_swaps: &Vec<(Index, Index, SlimSwap)>,
        dim: usize,
        params: &PruningParam,
        complex: &Complex,
        progress: F,
    ) -> Vec<(Index, Index, SlimSwap)> {
        let mut pruned = Vec::new();
        let n = all_swaps.len();
        for (i, s) in all_swaps.iter().enumerate() {
            progress(i, n);
            if s.2.len() == 0 {
                continue;
            }

            let mut dim_swaps = s.2.clone();

            if params.euclidean {
                if let Some(dist) = params.euclidean_distance {
                    prune_euclidian(&mut dim_swaps, &complex, dist)
                } else {
                    warn!(
                        "params dim {}: euclidean was true but distance was None",
                        dim
                    );
                }
            }

            if params.face {
                prune_common_face(&mut dim_swaps, &complex);
            }

            if params.coface {
                prune_coboundary(&mut dim_swaps, &complex);
            }

            if params.persistence {
                if let Some(dist) = params.persistence_threshold {
                    prune_persistence(&mut dim_swaps, dist)
                } else {
                    warn!(
                        "params dim {}: persistence was true but threshold was None",
                        dim
                    );
                }
            }

            if dim_swaps.is_empty() {
                continue;
            }

            pruned.push((s.0, s.1, dim_swaps));
        }

        pruned
    }

    /// Remove all swaps that were done between simplices that are closer than
    /// `min_dist^2`.
    ///
    /// Useful for 0th MA.
    pub fn prune_euclidian(swaps: &mut SlimSwap, complex: &Complex, min_dist2: f64) {
        swaps.retain(|(swap, _, _)| {
            let c1 = complex.simplices_per_dim[swap.dim][swap.i as usize].center_point(complex);
            let c2 = complex.simplices_per_dim[swap.dim][swap.j as usize].center_point(complex);
            let dist = c1.dist2(&c2);
            min_dist2 < dist
        })
    }

    pub fn prune_common_face(swaps: &mut SlimSwap, complex: &Complex) {
        let mut simp_to_vertices: HashMap<(usize, CI), HashSet<CI>> = HashMap::new();

        for dim in 1..3 {
            for (s, i) in complex.simplices_per_dim[dim].iter().zip(0..) {
                if dim == 1 {
                    let set = simp_to_vertices
                        .entry((dim, i))
                        .or_insert_with(|| HashSet::new());
                    for b in &s.boundary {
                        set.insert(*b);
                    }
                } else {
                    for b in &s.boundary {
                        let face_set = simp_to_vertices
                            .get(&(dim - 1, *b))
                            .expect("Should have inserted this simplex before")
                            .clone();
                        let set = simp_to_vertices
                            .entry((dim, i))
                            .or_insert_with(|| HashSet::new());
                        set.extend(face_set);
                    }
                }
            }
        }

        swaps.retain(|(swap, _, _)| {
            if swap.dim == 0 {
                return true;
            }
            let set_i = simp_to_vertices.get(&(swap.dim, swap.i)).unwrap();
            let set_j = simp_to_vertices.get(&(swap.dim, swap.j)).unwrap();
            let mut intersection = set_i.intersection(set_j);
            intersection.next() == None
        });
    }

    /// Remove all swaps that happen between simplices if there is a simplex
    /// with the two simplices in its boundary.
    pub fn prune_coboundary(swaps: &mut SlimSwap, complex: &Complex) {
        // (dim, id) to [id].
        let mut coboundary: HashMap<(usize, CI), HashSet<CI>> = HashMap::new();

        for dim in 1..3 {
            for (s, parent_i) in complex.simplices_per_dim[dim].iter().zip(0..) {
                for face_i in &s.boundary {
                    let v = coboundary
                        .entry((dim - 1, *face_i))
                        .or_insert_with(HashSet::new);
                    v.insert(parent_i);
                }
            }
        }

        swaps.retain(|(swap, _, _)| {
            if swap.dim == 2 {
                return true;
            }

            let cob_i = coboundary.get(&(swap.dim, swap.i));
            let cob_j = coboundary.get(&(swap.dim, swap.j));

            if let (Some(cob_i), Some(cob_j)) = (cob_i, cob_j) {
                let mut intersection = cob_i.intersection(cob_j);
                intersection.next() == None
            } else {
                true
            }
        });
    }

    /// Remove all swaps that happened where the persistence of any of the
    /// simplices were less than `lifetime`.
    ///
    /// `lifetime` can for instance be `0.01`.
    ///
    /// Probably only useful for 1st MA.
    pub fn prune_persistence(swaps: &mut SlimSwap, min_lifetime: f64) {
        swaps.retain(|&(_, lifetime_i, lifetime_j)| {
            if lifetime_i < min_lifetime && lifetime_j < min_lifetime {
                false
            } else {
                true
            }
        });
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[allow(non_snake_case)]
pub struct Stack {
    /// Boundary matrix. Size is (#vert, #edges)  (for 1st stack).
    pub D: SneakyMatrix,
    /// Reduced boundary matrix. Size is (#vert, #edges) (for 1st stack)
    pub R: SneakyMatrix,
    /// Inverse of the "column adds" matrix. Size is (#edges, #edges) (for 1st stack)
    pub U_t: SneakyMatrix,
    /// Ordering of the simplices. Cannonical to sorted order.
    pub ordering: Permutation,
}

impl Stack {
    pub fn mem_usage(&self) -> usize {
        self.D.mem_usage() + self.R.mem_usage() + self.U_t.mem_usage() + self.ordering.mem_usage()
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct BirthDeathPair {
    /// Dimension of the homology class.
    pub dim: isize,
    /// Birth time and canonical index of the simplex giving birth to the homology class.
    pub birth: Option<(f64, CI)>,
    /// Birth time and canonical index of the simplex giving birth to the homology class.
    pub death: Option<(f64, CI)>,
}

impl BirthDeathPair {
    pub fn lifetime(&self) -> f64 {
        let birth = self.birth.map(|t| t.0).unwrap_or(f64::NEG_INFINITY);
        let death = self.death.map(|t| t.0).unwrap_or(f64::INFINITY);
        death - birth
    }
}

/// Output data for a single point.  This is either computed from [`reduce_from_scratch`],
/// or by running the Vineyard algorithm from a point at which we already had a [`Reduction`].
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Reduction {
    /// Key point around which the reduction is done.
    pub key_point: Pos,
    /// Matrices for three dimensions.
    pub stacks: [Stack; 3],
}

impl Reduction {
    /// "Bake" all matrices for all dimensions. This means running
    /// [`SneakyMatrix::bake_in_permutations`] for all matrices.
    pub fn bake_all_matrices(&mut self) {
        for dim in 0..3 {
            self.stacks[dim].U_t.bake_in_permutations();
            self.stacks[dim].R.bake_in_permutations();
            self.stacks[dim].D.bake_in_permutations();
        }
    }

    /// Returns the Betti numbers for dimensions 0, 1, and 2.
    pub fn betti_numbers(&self) -> Vec<i8> {
        let mut bettis = vec![0; 3];
        for dim in 0..3 {
            let stack = &self.stacks[dim];
            for c in 0..stack.R.cols() {
                if stack.R.col_is_empty(c) {
                    bettis[dim] += 1;
                } else if 0 < dim {
                    bettis[dim - 1] -= 1;
                }
            }
        }
        bettis
    }

    /// Compute the entering value of the given simplex.
    ///
    /// Uses squared Euclidian distance.
    ///
    /// The `id` is a canonical index.
    pub fn simplex_entering_value(&self, complex: &Complex, dim: usize, id: CI) -> f64 {
        let simplex = &complex.simplices_per_dim[dim][id as usize];
        if dim == 0 {
            return simplex.coords.unwrap().dist2(&self.key_point);
        }
        simplex
            .boundary
            .iter()
            .map(|&b| self.simplex_entering_value(complex, dim - 1, b))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Find the killer of the given "simplex", if any.
    /// The `id` is a canonical index.
    fn find_killer(&self, dim: usize, id: CI) -> Option<CI> {
        if self.stacks.len() <= dim + 1 {
            return None;
        }
        let ordering = &self.stacks[dim].ordering;
        let sorted_i = ordering.map(id);
        let killer = self.stacks[dim + 1].R.col_with_low(sorted_i);
        if let Some(k) = killer {
            let can_k = self.stacks[dim + 1].ordering.inv(k);
            Some(can_k)
        } else {
            None
        }
    }

    /// Find the "simplex" that is killed by the given simplex, if any.
    ///
    /// `id` is a canonical index.
    fn find_victim(&self, dim: usize, id: CI) -> Option<CI> {
        if dim == 0 {
            // Pretend that the empty set doesn't exist, since we don't represent it explicitly.
            return None;
        }
        let sorted_id = self.stacks[dim].ordering.map(id);
        self.stacks[dim]
            .R
            .colmax(sorted_id)
            .map(|sorted_r| self.stacks[dim - 1].ordering.inv(sorted_r))
    }

    /// Compute the persistence of the given "simplex".
    /// `id` is the canonical index.
    ///
    /// Returns [None] if the "simplex" is not killed.
    pub fn persistence(&self, complex: &Complex, dim: usize, id: CI) -> Option<BirthDeathPair> {
        let killer = self.find_killer(dim, id);
        if let Some(killer) = killer {
            let dist = self.simplex_entering_value(complex, dim, id);
            let killer_dist = self.simplex_entering_value(complex, dim + 1, killer);
            Some(BirthDeathPair {
                dim: dim as isize,
                birth: Some((dist, id)),
                death: Some((killer_dist, killer)),
            })
        } else {
            // If we're the top dimension we will never be killed, but we might
            // have births. Check if column is zero.
            let ord_i = self.stacks[dim].ordering.map(id);
            if self.stacks[dim].R.col_is_empty(ord_i) {
                let dist = self.simplex_entering_value(complex, dim, id);
                Some(BirthDeathPair {
                    dim: dim as isize,
                    birth: Some((dist, id)),
                    death: None,
                })
            } else {
                panic!("we are in here after all!");
                None
            }
        }
    }

    pub fn barcode(&self, complex: &Complex, dim: isize) -> Vec<BirthDeathPair> {
        let mut ret = Vec::new();

        if dim == -1 {
            if 0 < complex.simplices_per_dim[0].len() {
                let first = self.stacks[0].ordering.inv(0);
                ret.push(BirthDeathPair {
                    dim: -1,
                    birth: None,
                    death: Some((self.simplex_entering_value(complex, 0, first), first)),
                });
            }
            return ret;
        }

        let ordering = &self.stacks[dim as usize].ordering;
        #[allow(non_snake_case)]
        let R = &self.stacks[dim as usize].R;

        for simplex in &complex.simplices_per_dim[dim as usize] {
            if !R.gives_birth(ordering.map(simplex.id)) {
                continue;
            }
            if let Some(persistence) = self.persistence(complex, dim as usize, simplex.id) {
                ret.push(persistence);
            }
        }
        ret
    }
}

#[allow(non_snake_case)]
impl Reduction {
    pub fn D(&self, dim: isize) -> &SneakyMatrix {
        assert!(0 <= dim);
        assert!(dim <= 2);
        &self.stacks[dim as usize].D
    }

    pub fn R(&self, dim: isize) -> &SneakyMatrix {
        assert!(0 <= dim);
        assert!(dim <= 2);
        &self.stacks[dim as usize].R
    }

    pub fn U_t(&self, dim: isize) -> &SneakyMatrix {
        assert!(0 <= dim);
        assert!(dim <= 2);
        &self.stacks[dim as usize].U_t
    }

    pub fn ordering(&self, dim: isize) -> &Permutation {
        assert!(0 <= dim);
        assert!(dim <= 2);
        &self.stacks[dim as usize].ordering
    }

    /// Checks that the ordering is consistent.
    pub fn assert_ordering(&self, complex: &Complex) {
        let mut vertex_distance = HashMap::new();

        let mut vertex_order = (0..complex.simplices_per_dim[0].len() as CI)
            .map(|i| {
                let s = &complex.simplices_per_dim[0][i as usize];
                let coords = s.coords.unwrap();
                let dist = coords.dist2(&self.key_point);
                vertex_distance.insert(i, dist);
                let sorted_order = self.stacks[0].ordering.map(i);
                (float_ord::FloatOrd(dist), sorted_order)
            })
            .collect::<Vec<_>>();
        vertex_order.sort();

        for t in vertex_order.windows(2) {
            let a = t[0];
            let b = t[1];
            assert!(a.1 <= b.1, "a[1] = {:?}, b[1] = {:?}", a.1, b.1);
        }

        let mut edge_dist = HashMap::new();
        let mut edge_order = (0..complex.simplices_per_dim[1].len() as CI)
            .map(|i| {
                let s = &complex.simplices_per_dim[1][i as usize];
                let ai = s.boundary[0];
                let bi = s.boundary[1];

                let adist = *vertex_distance.get(&ai).unwrap();
                let bdist = *vertex_distance.get(&bi).unwrap();

                let dist = adist.max(bdist);
                edge_dist.insert(i, dist);

                let sorted_order = self.stacks[1].ordering.map(i);
                (float_ord::FloatOrd(dist), sorted_order)
            })
            .collect::<Vec<_>>();
        edge_order.sort();

        for t in edge_order.windows(2) {
            let a = t[0];
            let b = t[1];
            assert!(a.1 <= b.1, "a.1 = {:?}, b.1 = {:?}", a, b);
        }

        let mut tri_order = (0..complex.simplices_per_dim[2].len() as CI)
            .map(|i| {
                let s = &complex.simplices_per_dim[2][i as usize];

                let ai = s.boundary[0];
                let bi = s.boundary[1];
                let ci = s.boundary[1];

                let adist = *edge_dist.get(&ai).unwrap();
                let bdist = *edge_dist.get(&bi).unwrap();
                let cdist = *edge_dist.get(&ci).unwrap();

                let dist = adist.max(bdist).max(cdist);

                let sorted_order = self.stacks[2].ordering.map(i);
                (float_ord::FloatOrd(dist), sorted_order)
            })
            .collect::<Vec<_>>();
        tri_order.sort();

        for t in tri_order.windows(2) {
            let a = t[0];
            let b = t[1];
            assert!(a.0 <= b.0, "a[0] = {:?}, b[0] = {:?}", a.0, b.0); // This will always pass
            assert!(a.1 <= b.1, "a[1] = {:?}, b[1] = {:?}", a.1, b.1);
        }
    }
}

/// The permutations returned are such that when you go forwards through the
/// permutation, you get the simplices in sorted order based on their distance
/// to the key point.
///
/// I.e, `v_perm[0]` is the canonical index of the closest vertex, and
/// `v_perm[1]` is the canonical index of the second closest index.
///
/// In other words, `v_perm.map` takes a "sorted" index and returns a "canonical" index.
fn compute_permutations(
    complex: &Complex,
    key_point: Pos,
) -> (Permutation, Permutation, Permutation) {
    let vertex_distances = complex.simplices_per_dim[0]
        .iter()
        .map(|v| float_ord::FloatOrd(v.coords.unwrap().dist2(&key_point)))
        .collect::<Vec<_>>();

    let edge_distances = complex.simplices_per_dim[1]
        .iter()
        .map(|e| {
            let dist_a = vertex_distances[e.boundary[0] as usize];
            let dist_b = vertex_distances[e.boundary[1] as usize];
            dist_a.max(dist_b)
        })
        .collect::<Vec<_>>();

    let triangle_distances = complex.simplices_per_dim[2]
        .iter()
        .map(|f| {
            let dist_a = edge_distances[f.boundary[0] as usize];
            let dist_b = edge_distances[f.boundary[1] as usize];
            let dist_c = edge_distances[f.boundary[2] as usize];
            dist_a.max(dist_b).max(dist_c)
        })
        .collect::<Vec<_>>();

    let v_perm = Permutation::from_ord(&vertex_distances);
    let e_perm = Permutation::from_ord(&edge_distances);
    let t_perm = Permutation::from_ord(&triangle_distances);

    (v_perm, e_perm, t_perm)
}

/// Run Vineyards from a previous [`Reduction`] to a new key point.
pub fn vineyards_step(
    complex: &Complex,
    reduction: &Reduction,
    key_point: Pos,
    require_hom_birth_to_be_first: bool,
) -> (Reduction, Swaps) {
    let (mut v_perm, mut e_perm, mut t_perm) = compute_permutations(complex, key_point);

    let mut stack0 = reduction.stacks[0].clone();
    let mut stack1 = reduction.stacks[1].clone();
    let mut stack2 = reduction.stacks[2].clone();

    let mut faustian_swap_simplices = Vec::new();

    if 0 < t_perm.len() {
        t_perm.reverse();
        let vine_ordering2 = Permutation::from_to(&stack2.ordering, &t_perm);
        let (swap_is2, simplices_that_got_swapped2) =
            compute_transpositions(vine_ordering2.clone().into_forwards());
        for (swap_i, &i) in swap_is2.iter().enumerate() {
            let res = perform_one_swap_top_dim(i, &mut stack2);
            stack2.D.swap_cols(i, i + 1);
            if let Some(true) = res {
                let (i, j) = simplices_that_got_swapped2[swap_i];
                let cann_i = t_perm.inv(i);
                let cann_j = t_perm.inv(j);

                faustian_swap_simplices.push(Swap {
                    dim: 2,
                    i: cann_i,
                    j: cann_j,
                });
            }
        }

        stack2.ordering = t_perm;
    }

    if 0 < e_perm.len() {
        e_perm.reverse();
        let vine_ordering1 = Permutation::from_to(&stack1.ordering, &e_perm);
        let (swap_is1, simplices_that_got_swapped1) =
            compute_transpositions(vine_ordering1.clone().into_forwards());

        let up_U_t = &mut stack2.U_t;
        let mut up_cwi = ColWithInv::new(&mut stack2.R);

        for (swap_i, &i) in swap_is1.iter().enumerate() {
            let res = perform_one_swap(i, &mut stack1, &mut up_cwi, up_U_t);
            stack1.D.swap_cols(i, i + 1);
            stack2.D.swap_rows(i, i + 1);
            if let Some(true) = res {
                let (i, j) = simplices_that_got_swapped1[swap_i];
                let cann_i = e_perm.inv(i);
                let cann_j = e_perm.inv(j);

                faustian_swap_simplices.push(Swap {
                    dim: 1,
                    i: cann_i,
                    j: cann_j,
                });
            }
        }

        stack1.ordering = e_perm;
    }

    // NOTE: we need the permutation from cannonical to sorted, so that we can
    // get the permutation that takes us from the `b` point to the `a` point, so
    // that, in turn, we can bubble sort from `a` to `b`.
    v_perm.reverse();

    // v_perm[0] = 12          this is the column at which v0 is at in the ordering.
    // stack0.ordering[0] = 14 this is the column at which v0 is at in the stack0 ordering.
    // if vine_ordering0[12] == 14, then
    // vine_ordering0[v_perm[0]] == stack0.ordering[0]  means that
    // v0 is column 12 at this ordering, and column 14 at the old ordering.
    // Which means that vine_ordering0 takes a new column index and produces an old column index.

    // This should map old to new indices.
    let vine_ordering0 = Permutation::from_to(&stack0.ordering, &v_perm);

    //  0 . 1 . 2 . 3 . 4 ...   <--- old indices
    // [0,  3,  1,  4,  2, ...] <------- new indices

    // First swap:
    //      1,  3
    // This was at index i=1, and corresponds to simplices at (3) and (1).

    // swap_is0 has to contain ordered indices from the OLD ordering.
    let (swap_is0, simplices_that_got_swapped0) =
        compute_transpositions(vine_ordering0.clone().into_forwards());

    let up_U_t = &mut stack1.U_t;
    let mut up_cwi = ColWithInv::new(&mut stack1.R);

    for (swap_i, &i) in swap_is0.iter().enumerate() {
        let res = perform_one_swap(i, &mut stack0, &mut up_cwi, up_U_t);
        stack0.D.swap_cols(i, i + 1);
        stack1.D.swap_rows(i, i + 1);

        if let Some(true) = res {
            // These are indices of simplices that we said were the 0,1,2... order
            // in the bubble sort (compute_transpositions).  This is the order
            // of the simplices at `a`.
            let (i, j) = simplices_that_got_swapped0[swap_i];
            let cann_i = v_perm.inv(i);
            let cann_j = v_perm.inv(j);

            faustian_swap_simplices.push(Swap {
                dim: 0,
                i: cann_i,
                j: cann_j,
            });
        }
    }

    stack0.ordering = v_perm;

    let state = Reduction {
        key_point,
        stacks: [stack0, stack1, stack2],
    };

    if require_hom_birth_to_be_first {
        // We now have a bunch of faustian swaps.  However, we only want the first "real" cycle for
        // each dimension, meaning it has >0 persistence.  A swap consists of two simplices (A B) where
        // A used to give death, and B used to give birth, that have changed in the new ordering (B A).
        // We look at the persistence of the cycle that is given birth to.
        //
        // Since we're only interested in the first cycle (per dim), we precompute this to figure out
        // which final simplices we're interested in.  For the used-to-give-birth simplex we check
        // the initial input matrix.  For the is-now-giving-birth simplex we check the newly reduced
        // matrix.  Then we go over the swaps and remove those that don't contain either.

        /// Finds the simplex that creates the first homology class in a dimension.
        ///
        /// Returns the canonical index of the simplex.
        fn find_interesting(reduction: &Reduction, complex: &Complex, dim: usize) -> Option<CI> {
            #[allow(non_snake_case)]
            let R = &reduction.stacks[dim].R;

            for ord_i in 0..R.cols() {
                if R.col_is_not_empty(ord_i) {
                    continue;
                }
                let can_i = reduction.stacks[dim].ordering.inv(ord_i);
                let Some(p) = reduction.persistence(complex, dim, can_i) else {
                    continue;
                };
                if 1e-6 < p.lifetime() {
                    return Some(can_i);
                }
            }

            None
        }

        // NOTE: we only tested this for dim=0, so let's crash here if we try to use it.
        unimplemented!();

        // TODO: range 0..3
        for dim in 1..2 {
            let old_interesting_edge = find_interesting(reduction, complex, dim);
            let new_interesting_edge = find_interesting(&state, complex, dim);

            if let Some((min, max)) = old_interesting_edge
                .zip(new_interesting_edge)
                .map(|(o, n)| (o.min(n), o.max(n)))
            {
                faustian_swap_simplices
                    .retain(|p| p.dim != 1 || p.i.min(p.j) == min || p.i.max(p.j) == max);
            }
        }
    }

    (
        state,
        Swaps {
            v: faustian_swap_simplices,
        },
    )
}

/// Compute the reduction at a key point from scratch.
#[allow(non_snake_case)]
pub fn reduce_from_scratch(complex: &Complex, key_point: Pos, noisy: bool) -> Reduction {
    info!("reduce from scratch");
    let (mut v_perm, mut e_perm, mut t_perm) = compute_permutations(complex, key_point);

    let mut boundary_0 = complex.boundary_matrix(0);
    // Order the cols and rows of the three boundary matrices by their distance
    // to the key point. This permutation is computed above, so we just need to
    // set the fields.
    boundary_0.col_perm = Some(v_perm.clone());
    let D0 = boundary_0.clone();

    let mut boundary_1 = complex.boundary_matrix(1);
    boundary_1.col_perm = Some(e_perm.clone());
    boundary_1.row_perm = Some(v_perm.clone());
    let D1 = boundary_1.clone();

    let mut boundary_2 = complex.boundary_matrix(2);
    boundary_2.col_perm = Some(t_perm.clone());
    boundary_2.row_perm = Some(e_perm.clone());
    let D2 = boundary_2.clone();

    if noisy {
        print!("Reduce dim0 ... ");
    }
    let adds0 = boundary_0.reduce();
    if noisy {
        print!("done\nReduce dim1 ... ");
    }
    let adds1 = boundary_1.reduce();
    if noisy {
        print!("done\nReduce dim2 ... ");
    }
    let adds2 = boundary_2.reduce();
    if noisy {
        println!("done");
    }

    let mut V0 = SneakyMatrix::eye(boundary_0.cols());
    for (target, other) in adds0 {
        V0.add_cols(target, other);
    }

    let mut V1 = SneakyMatrix::eye(boundary_1.cols());
    for (target, other) in adds1 {
        V1.add_cols(target, other);
    }

    let mut V2 = SneakyMatrix::eye(boundary_2.cols());
    for (target, other) in adds2 {
        V2.add_cols(target, other);
    }

    if noisy {
        print!("Invert V0 ... ");
    }
    let U_t0 = V0.inverse_gauss_jordan().transpose();
    if noisy {
        print!("done\nInvert V1 ... ");
    }
    let U_t1 = V1.inverse_gauss_jordan().transpose();
    if noisy {
        print!("done\nInvert V2 ... ");
    }
    let U_t2 = V2.inverse_gauss_jordan().transpose();
    if noisy {
        println!("done");
    }

    let R0 = boundary_0;
    let R1 = boundary_1;
    let R2 = boundary_2;

    v_perm.reverse();
    e_perm.reverse();
    t_perm.reverse();

    let ret = Reduction {
        key_point,
        stacks: [
            Stack {
                D: D0,
                R: R0,
                U_t: U_t0,
                ordering: v_perm,
            },
            Stack {
                D: D1,
                R: R1,
                U_t: U_t1,
                ordering: e_perm,
            },
            Stack {
                D: D2,
                R: R2,
                U_t: U_t2,
                ordering: t_perm,
            },
        ],
    };

    ret.assert_ordering(&complex);
    info!("reduce from scratch done");

    ret
}

/// Caching layer around [SneakyMatrix::col_with_low] so that we don't have to recompute this every
/// time.
///
/// The exact cache logic is a little gnarly and is tightly coupled with the logic in
/// [perform_one_swap].
struct ColWithInv<'a> {
    /// The [SneakyMatrix] that we are caching.
    inner: &'a mut SneakyMatrix,
    /// The cache.  Missing entries are marked with [CI::MAX].
    cache: Vec<CI>,
}

impl<'a> ColWithInv<'a> {
    fn new(inner: &'a mut SneakyMatrix) -> ColWithInv {
        let mut col_with_low = vec![CI::MAX; inner.rows() as usize];
        for c in 0..inner.cols() {
            if let Some(r) = inner.colmax(c) {
                col_with_low[r as usize] = c;
            }
        }

        Self {
            inner,
            cache: col_with_low,
        }
    }

    /// Verify that every entry in the cache is the same as the recomputed value from
    /// [SneakyMatrix::col_with_low].
    fn verify(&self) {
        for r in 0..self.inner.rows() {
            let res = self.inner.col_with_low(r);
            let ours = self.cache[r as usize];
            let ours_opt = if ours == CI::MAX { None } else { Some(ours) };
            assert_eq!(res, ours_opt, "Mismatch {r}");
        }
    }

    /// [SneakyMatrix::get]
    fn get(&self, r: CI, c: CI) -> bool {
        self.inner.get(r, c)
    }

    /// Swap rows in the matrix and in the cache.
    fn swap_rows(&mut self, r1: CI, r2: CI) {
        self.inner.swap_rows(r1, r2);
        self.cache.swap(r1 as usize, r2 as usize);
    }

    /// Swap rows without updating the cache.
    fn swap_rows_skip_cache(&mut self, r1: CI, r2: CI) {
        self.inner.swap_rows(r1, r2);
    }

    /// Swap rows with special logic in the death-birth case.
    fn swap_rows_db(&mut self, r1: CI, r2: CI) {
        let c = self.cache[r2 as usize];
        if c == CI::MAX || !self.inner.get(r1, c) {
            self.cache.swap(r1 as usize, r2 as usize);
        }
        self.inner.swap_rows(r1, r2);
    }

    /// Swap rows with special logic in the birth-birth case.
    fn swap_rows_bb(&mut self, r1: CI, r2: CI) {
        let c = self.cache[r2 as usize];
        if c == CI::MAX || !self.inner.get(r1, c) {
            self.cache.swap(r1 as usize, r2 as usize);
        }
        self.inner.swap_rows(r1, r2);
    }

    /// [SneakyMatrix::add_cols].
    fn add_cols(&mut self, c1: CI, c2: CI) {
        self.inner.add_cols(c1, c2);
    }

    fn col_with_low(&self, r: CI) -> Option<CI> {
        let c = self.cache[r as usize];
        let c_opt = if c == CI::MAX { None } else { Some(c) };
        c_opt
    }
}

#[allow(non_snake_case)]
fn perform_one_swap(
    i: CI,
    stack: &mut Stack,
    up_cwi: &mut ColWithInv,
    up_U_t: &mut SneakyMatrix,
) -> Option<bool> {
    #[allow(non_snake_case)]
    fn gives_death(R: &SneakyMatrix, c: CI) -> bool {
        R.col_is_not_empty(c)
    }

    #[allow(non_snake_case)]
    fn low(R: &SneakyMatrix, c: CI) -> Option<CI> {
        R.colmax(c)
    }

    #[allow(non_snake_case)]
    fn low_inv(R: &SneakyMatrix, r: CI) -> Option<CI> {
        R.col_with_low(r)
    }

    let gives_death_i = gives_death(&stack.R, i);
    let gives_birth_i = !gives_death_i;
    let gives_death_i_1 = gives_death(&stack.R, i + 1);
    let gives_birth_i_1 = !gives_death_i_1;

    // @perf: The branch ratio here is around
    //     bb=45.0  bd=22.0  db=23.0  dd=11.0

    // if gives_birth_i and gives_birth_i_1:
    if gives_birth_i && gives_birth_i_1 {
        // println!("bb");
        // U_t[i + 1, i] = 0
        stack.U_t.set(i + 1, i, false);
        // k = low_inv(i)
        let k = up_cwi.col_with_low(i);
        // l = low_inv(i + 1)
        let l = up_cwi.col_with_low(i + 1);
        // if k != None and l != None and R[i, l] == 1:
        if let (Some(k), Some(l)) = (k, l) {
            if up_cwi.get(i, l) {
                // if k < l:
                if k < l {
                    // R.swap_cols_and_rows(i, i + 1)  # PRP
                    stack.R.swap_cols(i, i + 1);
                    up_cwi.swap_rows(i, i + 1);
                    // R.add_cols(l, k)  # PRPV
                    up_cwi.add_cols(l, k);
                    // U_t.swap_cols_and_rows(i, i + 1)  # PUP
                    stack.U_t.swap_cols_and_rows(i, i + 1);
                    // U_t.add_cols(k, l)  # VPUP
                    up_U_t.add_cols(k, l);
                    // return (R, U_t, None)
                    return None;
                }
                // if l < k:
                if l < k {
                    // R.swap_cols_and_rows(i, i + 1)  # PRP
                    stack.R.swap_cols(i, i + 1);
                    up_cwi.swap_rows_skip_cache(i, i + 1);
                    // R.add_cols(k, l)  # PRPV
                    up_cwi.add_cols(k, l);
                    // U_t.swap_cols_and_rows(i, i + 1)  # PUP
                    stack.U_t.swap_cols_and_rows(i, i + 1);
                    // U_t.add_cols(l, k)  # VPUP
                    up_U_t.add_cols(l, k);
                    // return (R, U_t, False)
                    return Some(false);
                }
                panic!("This should never happen: l == k ({})", l);
                // raise Exception("k = l; This should never happen.")
                // else:
            }
        }

        // println!("k={:?} l={:?}", k, l);

        // else case
        // R.swap_cols_and_rows(i, i + 1)  # PRP
        stack.R.swap_cols(i, i + 1);
        up_cwi.swap_rows_bb(i, i + 1);
        // U_t.swap_cols_and_rows(i, i + 1)  # PUP
        stack.U_t.swap_cols_and_rows(i, i + 1);
        // return (R, U_t, None)
        return None;
    }
    // if gives_death_i and gives_death_i_1:
    if gives_death_i && gives_death_i_1 {
        // println!("dd");
        // if U_t[i + 1, i] == 1:
        if stack.U_t.get(i + 1, i) {
            // low_i = low(i)
            let low_i = low(&stack.R, i);
            // low_i_1 = low(i + 1)
            let low_i_1 = low(&stack.R, i + 1);
            // U_t.add_cols(i, i + 1)  # W U
            stack.U_t.add_cols(i, i + 1);
            // R.add_cols(i + 1, i)  # R W
            stack.R.add_cols(i + 1, i);
            // R.swap_cols_and_rows(i, i + 1)  # P R W P
            stack.R.swap_cols(i, i + 1);
            up_cwi.swap_rows(i, i + 1);
            // U_t.swap_cols_and_rows(i, i + 1)  # P W U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // if low_i < low_i_1:
            if low_i < low_i_1 {
                // return (R, U_t, None)
                return None;
            // else:
            } else {
                // R.add_cols(i + 1, i)  # (P R W P) W
                stack.R.add_cols(i + 1, i);
                // U_t.add_cols(i, i + 1)  # W (P W U P)
                stack.U_t.add_cols(i, i + 1);
                // return (R, U_t, False)
                return Some(false);
            }
        // else:
        } else {
            // R.swap_cols_and_rows(i, i + 1)  # P R P
            stack.R.swap_cols(i, i + 1);
            up_cwi.swap_rows(i, i + 1);
            // U_t.swap_cols_and_rows(i, i + 1)  # P U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // return (R, U_t, None)
            return None;
        }
    }
    // if gives_death_i and gives_birth_i_1:
    if gives_death_i && gives_birth_i_1 {
        // println!("db");
        // if U_t[i + 1, i] == 1:
        if stack.U_t.get(i + 1, i) {
            // U_t.add_cols(i, i + 1)  # W U
            stack.U_t.add_cols(i, i + 1);
            // R.add_cols(i + 1, i)  # R W
            stack.R.add_cols(i + 1, i);
            // R.swap_cols_and_rows(i, i + 1)  # P R W P
            stack.R.swap_cols(i, i + 1);
            up_cwi.swap_rows_db(i, i + 1);
            // R.add_cols(i + 1, i)  # (P R W P) W
            stack.R.add_cols(i + 1, i);
            // U_t.swap_cols_and_rows(i, i + 1)  # P W U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // U_t.add_cols(i, i + 1)  # W (P W U P)
            stack.U_t.add_cols(i, i + 1);
            // return (R, U_t, True)
            return Some(true);
        // else:
        } else {
            // R.swap_cols_and_rows(i, i + 1)  # P R P
            stack.R.swap_cols(i, i + 1);
            up_cwi.swap_rows(i, i + 1);
            // U_t.swap_cols_and_rows(i, i + 1)  # P U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // return (R, U_t, None)
            return None;
        }
    }
    // if gives_birth_i and gives_death_i_1:
    if gives_birth_i && gives_death_i_1 {
        // println!("bd");
        // U_t[i + 1, i] = 0
        stack.U_t.set(i + 1, i, false);
        // R.swap_cols_and_rows(i, i + 1)  # P R P
        stack.R.swap_cols(i, i + 1);
        up_cwi.swap_rows(i, i + 1);
        // U_t.swap_cols_and_rows(i, i + 1)  # P U P
        stack.U_t.swap_cols_and_rows(i, i + 1);
        // return (R, U_t, None)
        return None;
    }

    // raise Exception("bottom of the function; This should never happen.")
    panic!("This should never happen: no cases matched.");
}

/// [perform_one_swap], but where we don't have a dimension above us.
/// This simplifies the logic some.
#[allow(non_snake_case)]
fn perform_one_swap_top_dim(i: CI, stack: &mut Stack) -> Option<bool> {
    #[allow(non_snake_case)]
    fn gives_death(R: &SneakyMatrix, c: CI) -> bool {
        R.col_is_not_empty(c)
    }

    #[allow(non_snake_case)]
    fn low(R: &SneakyMatrix, c: CI) -> Option<CI> {
        R.colmax(c)
    }

    let gives_death_i = gives_death(&stack.R, i);
    let gives_birth_i = !gives_death_i;
    let gives_death_i_1 = gives_death(&stack.R, i + 1);
    let gives_birth_i_1 = !gives_death_i_1;

    // if gives_birth_i and gives_birth_i_1:
    if gives_birth_i && gives_birth_i_1 {
        // U_t[i + 1, i] = 0
        stack.U_t.set(i + 1, i, false);
        // k = low_inv(i)

        // else case
        // R.swap_cols_and_rows(i, i + 1)  # PRP
        stack.R.swap_cols(i, i + 1);
        // U_t.swap_cols_and_rows(i, i + 1)  # PUP
        stack.U_t.swap_cols_and_rows(i, i + 1);
        // return (R, U_t, None)
        return None;
    }
    // if gives_death_i and gives_death_i_1:
    if gives_death_i && gives_death_i_1 {
        // if U_t[i + 1, i] == 1:
        if stack.U_t.get(i + 1, i) {
            // low_i = low(i)
            let low_i = low(&stack.R, i);
            // low_i_1 = low(i + 1)
            let low_i_1 = low(&stack.R, i + 1);
            // U_t.add_cols(i, i + 1)  # W U
            stack.U_t.add_cols(i, i + 1);
            // R.add_cols(i + 1, i)  # R W
            stack.R.add_cols(i + 1, i);
            // R.swap_cols_and_rows(i, i + 1)  # P R W P
            stack.R.swap_cols(i, i + 1);
            // U_t.swap_cols_and_rows(i, i + 1)  # P W U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // if low_i < low_i_1:
            if low_i < low_i_1 {
                // return (R, U_t, None)
                return None;
            // else:
            } else {
                // R.add_cols(i + 1, i)  # (P R W P) W
                stack.R.add_cols(i + 1, i);
                // U_t.add_cols(i, i + 1)  # W (P W U P)
                stack.U_t.add_cols(i, i + 1);
                // return (R, U_t, False)
                return Some(false);
            }
        // else:
        } else {
            // R.swap_cols_and_rows(i, i + 1)  # P R P
            stack.R.swap_cols(i, i + 1);
            // U_t.swap_cols_and_rows(i, i + 1)  # P U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // return (R, U_t, None)
            return None;
        }
    }
    // if gives_death_i and gives_birth_i_1:
    if gives_death_i && gives_birth_i_1 {
        // if U_t[i + 1, i] == 1:
        if stack.U_t.get(i + 1, i) {
            // U_t.add_cols(i, i + 1)  # W U
            stack.U_t.add_cols(i, i + 1);
            // R.add_cols(i + 1, i)  # R W
            stack.R.add_cols(i + 1, i);
            // R.swap_cols_and_rows(i, i + 1)  # P R W P
            stack.R.swap_cols(i, i + 1);
            // R.add_cols(i + 1, i)  # (P R W P) W
            stack.R.add_cols(i + 1, i);
            // U_t.swap_cols_and_rows(i, i + 1)  # P W U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // U_t.add_cols(i, i + 1)  # W (P W U P)
            stack.U_t.add_cols(i, i + 1);
            // return (R, U_t, True)

            for k in 0..i {
                if stack.R.col_is_empty(k) {
                    return Some(false);
                }
            }

            return Some(true);
        // else:
        } else {
            // R.swap_cols_and_rows(i, i + 1)  # P R P
            stack.R.swap_cols(i, i + 1);
            // U_t.swap_cols_and_rows(i, i + 1)  # P U P
            stack.U_t.swap_cols_and_rows(i, i + 1);
            // return (R, U_t, None)
            return None;
        }
    }
    // if gives_birth_i and gives_death_i_1:
    if gives_birth_i && gives_death_i_1 {
        // U_t[i + 1, i] = 0
        stack.U_t.set(i + 1, i, false);
        // R.swap_cols_and_rows(i, i + 1)  # P R P
        stack.R.swap_cols(i, i + 1);
        // U_t.swap_cols_and_rows(i, i + 1)  # P U P
        stack.U_t.swap_cols_and_rows(i, i + 1);
        // return (R, U_t, None)
        return None;
    }

    // raise Exception("bottom of the function; This should never happen.")
    panic!("This should never happen: no cases matched.");
}

/// Compute the transpositions required to swap a permutation to become `0..n`.
///
/// Returns a `Vec<usize>` where each element `i` correspond to the transposition
/// `(i, i+1)` taking place.
/// Also return the "column value" of the simplices that were swapped. This is
/// used to figure out which simplices the swap consisted of.
pub fn compute_transpositions(mut b: Vec<CI>) -> (Vec<CI>, Vec<(CI, CI)>) {
    // NOTE: The `this` ordering is implicitly `0..n`.
    let n = b.len();
    let n0 = zip(&b, 0..n)
        .position(|(&aa, bb)| aa as usize != bb)
        .unwrap_or(n) as CI;
    let n1 = zip(&b, 0..n)
        .rposition(|(&aa, bb)| aa as usize != bb)
        .unwrap_or(n) as CI;
    let mut ret = Vec::with_capacity(n);
    let mut swapped_indices = Vec::with_capacity(n);
    for _ in n0..=n1 {
        let mut swap = false;
        for i in 0..(n - 1) {
            if b[i] > b[i + 1] {
                ret.push(i as CI);
                b.swap(i, i + 1);
                swapped_indices.push((b[i], b[i + 1]));
                swap = true;
            }
        }
        if !swap {
            break;
        }
    }
    (ret, swapped_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_transpositions() {
        let a = vec![0, 1, 4, 3, 2, 5];
        // [0, 1, 4, 3, 2, 5]  start
        // [0, 1, 3, 4, 2, 5]  swapped 2 and 3
        // [0, 1, 3, 2, 4, 5]  swapped 3 and 4
        // [0, 1, 2, 3, 4, 5]  swapped 2 and 3
        let res = compute_transpositions(a);
        assert_eq!(res.0, vec![2, 3, 2]);

        let a = vec![3, 4, 5, 2, 0, 1];
        // [3, 4, 5, 2, 0, 1]  start
        // [3, 4, 2, 5, 0, 1]  2
        // [3, 4, 2, 0, 5, 1]  3
        // [3, 4, 2, 0, 1, 5]  4
        // [3, 2, 4, 0, 1, 5]  1
        // [3, 2, 0, 4, 1, 5]  2
        // [3, 2, 0, 1, 4, 5]  3
        // [2, 3, 0, 1, 4, 5]  0
        // [2, 0, 3, 1, 4, 5]  1
        // [2, 0, 1, 3, 4, 5]  2
        // [0, 2, 1, 3, 4, 5]  0
        // [0, 1, 2, 3, 4, 5]  1
        let res = compute_transpositions(a);
        assert_eq!(res.0, vec![2, 3, 4, 1, 2, 3, 0, 1, 2, 0, 1]);

        let a = vec![0, 1, 2, 4, 5, 6, 7, 3, 8, 9, 10, 11];
        let res = compute_transpositions(a);
        assert_eq!(res.0, vec![6, 5, 4, 3]);
    }

    use crate::test::*;

    #[test]
    fn snapshot_grid_reduction_matrices() {
        let complex = test_complex_cube();

        fn test(complex: &Complex, pos: complex::Pos) {
            let mut reduction = reduce_from_scratch(complex, pos, false);
            reduction.bake_all_matrices();

            for dim in 0..3 {
                insta::assert_snapshot!(reduction.D(dim).__str__());
                insta::assert_snapshot!(reduction.R(dim).__str__());
                insta::assert_snapshot!(reduction.U_t(dim).__str__());
            }
        }

        test(&complex, complex::Pos([0.0, 0.0, 0.0]));
        test(&complex, complex::Pos([0.1, 0.2, 0.3]));
        test(&complex, complex::Pos([0.0, -0.5, 0.25]));
    }

    #[test]
    fn snapshot_medial_axes_for_grid() {
        let complex = test_complex_cube();
        let grid = test_grid_for_cube();

        let mars = Mars {
            complex: Some(complex),
            grid: Some(Grid::Regular(grid)),
        };

        let no_progress = |_, _| {};

        let vin = mars.run(&no_progress).expect("failed to run mars");

        for dim in 0..3 {
            let params = default_pruning_param(dim);
            let pruned = vin.prune_dim(dim, &params, mars.complex.as_ref().unwrap(), no_progress);
            let mut pairs = pruned
                .into_iter()
                .filter(|t| t.2.v.len() > 0)
                .map(|(i, j, _)| (i.min(j), i.max(j)))
                .collect::<Vec<_>>();
            pairs.sort();
            insta::assert_json_snapshot!(pairs);
        }
    }

    /// Endpoint positions of a swap-bearing grid edge, rounded, ordered, plus the number of
    /// vertices of its dual face.  Comparable across grid representations.
    fn edge_signature(pa: Pos, pb: Pos, face_vertices: usize) -> ([i64; 3], [i64; 3], usize) {
        let key = |p: Pos| {
            [
                (p.x() * 1e6).round() as i64,
                (p.y() * 1e6).round() as i64,
                (p.z() * 1e6).round() as i64,
            ]
        };
        let (a, b) = (key(pa), key(pb));
        if a <= b {
            (a, b, face_vertices)
        } else {
            (b, a, face_vertices)
        }
    }

    fn run_and_prune_mesh(mars: &Mars) -> [Vec<([i64; 3], [i64; 3], usize)>; 3] {
        let no_progress = |_, _| {};
        let vin = mars.run(&no_progress).expect("failed to run mars");
        let Some(Grid::Mesh(ref grid)) = mars.grid else {
            unreachable!()
        };
        let complex = mars.complex.as_ref().unwrap();
        let mut out = [Vec::new(), Vec::new(), Vec::new()];
        for dim in 0..3 {
            let pruned = vin.prune_dim(dim, &default_pruning_param(dim), complex, no_progress);
            let mut sigs: Vec<_> = pruned
                .iter()
                .filter(|t| t.2.v.len() > 0)
                .map(|(i, j, _)| {
                    edge_signature(
                        grid.coordinate(*i),
                        grid.coordinate(*j),
                        grid.dual_face_points(*i, *j).len(),
                    )
                })
                .collect();
            sigs.sort();
            sigs.dedup();
            out[dim] = sigs;
        }
        out
    }

    /// The mesh-grid path on a simple cubic block must find the same swap-bearing edges as the
    /// regular grid on the same points.  The grid is placed in generic position: with
    /// [test_grid_for_cube] the grid points lie on the cube's symmetry planes, ties between
    /// simplex values are then resolved differently depending on the traversal direction, and
    /// a handful of edges differ between the BFS of the regular grid and the DFS of the mesh.
    #[test]
    fn mesh_grid_matches_regular_grid_on_cube() {
        let complex = test_complex_cube();
        let regular = grid::VineyardsGrid::new(Pos([-0.5137, -0.4871, -0.5023]), 0.4123, [5, 5, 5]);
        let mars = Mars {
            complex: Some(complex.clone()),
            grid: Some(Grid::Regular(regular.clone())),
        };
        let no_progress = |_, _| {};
        let vin = mars.run(&no_progress).expect("failed to run mars");
        let mut expected = [Vec::new(), Vec::new(), Vec::new()];
        for dim in 0..3 {
            let pruned = vin.prune_dim(dim, &default_pruning_param(dim), &complex, no_progress);
            let mut sigs: Vec<_> = pruned
                .iter()
                .filter(|t| t.2.v.len() > 0)
                .map(|(i, j, _)| edge_signature(regular.coordinate(*i), regular.coordinate(*j), 4))
                .collect();
            sigs.sort();
            sigs.dedup();
            expected[dim] = sigs;
        }

        // shape [5; 3] == 4 cells of side `size`, centred two cells in from the corner.
        let centre = regular.corner + Pos([2.0 * regular.size; 3]);
        let block = lattice_block(Lattice::Sc, [4, 4, 4], regular.size, centre);
        let obj = block_to_obj(
            &block,
            &ObjOptions {
                lines: Lines::Shell1,
                ..Default::default()
            },
        );
        let grid = grid::VineyardsGridMesh::read_from_obj_string(&obj).unwrap();
        assert_eq!(grid.points.len(), 125);
        let mars = Mars {
            complex: Some(complex),
            grid: Some(Grid::Mesh(grid)),
        };
        let got = run_and_prune_mesh(&mars);
        for dim in 0..3 {
            let g: std::collections::BTreeSet<_> = got[dim].iter().cloned().collect();
            let e: std::collections::BTreeSet<_> = expected[dim].iter().cloned().collect();
            let only_mesh: Vec<_> = g.difference(&e).take(5).collect();
            let only_regular: Vec<_> = e.difference(&g).take(5).collect();
            assert!(
                g == e,
                "dimension {}: regular grid has {} swap-bearing edges, mesh grid {}; \
                 {} only in mesh (e.g. {:?}), {} only in regular (e.g. {:?})",
                dim,
                e.len(),
                g.len(),
                g.difference(&e).count(),
                only_mesh,
                e.difference(&g).count(),
                only_regular
            );
            assert!(!got[dim].is_empty() || dim == 2);
        }
    }

    #[test]
    fn snapshot_medial_axes_for_bcc_mesh_grid() {
        let complex = test_complex_cube();
        let block = lattice_block(Lattice::Bcc, [4, 4, 4], 0.4, Pos([0.3; 3]));
        let obj = block_to_obj(&block, &ObjOptions::default());
        let grid = grid::VineyardsGridMesh::read_from_obj_string(&obj).unwrap();
        assert_eq!(grid.points.len(), 125 + 64);
        let mars = Mars {
            complex: Some(complex),
            grid: Some(Grid::Mesh(grid)),
        };
        let got = run_and_prune_mesh(&mars);
        for dim in 0..3 {
            insta::assert_json_snapshot!(got[dim]);
        }
    }
}
