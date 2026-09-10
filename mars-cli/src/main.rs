#![allow(non_snake_case)]
use anyhow::{anyhow, bail, Context, Result};
use mars_core::{
    complex::{Complex, Pos},
    grid::{write_face, GridReport, Index, VineyardsGridMesh},
    stats::{MarsMem, ReductionMem},
    Grid, Mars, PruningParam, Swap,
};
use std::{
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Sub,
}

#[derive(Debug, Subcommand)]
enum Sub {
    /// Print the default parameters used for pruning.
    ///
    /// The output can be used as a starting point for your own pruning parameters, which can then
    /// be passed to either `mars-cli run` or `mars-cli prune`.
    PrintPrune,
    /// Run the algorithm and output a file containing the entire state.
    ///
    /// If pruning is enabled, default pruning parameters are used.  If the path to a pruning file
    /// is included, those parameters are used.  The default pruning paramters can be obtained by
    /// running `mars-cli print-prune`.
    ///
    /// Re-pruning a state file is TODO.
    Run(RunArgs),
    /// Output .obj files from the state file.
    ///
    /// The medial axes .obj will contain the three axes as separate objects.
    Obj(ObjArgs),

    /// Prune swaps from a state file.
    Prune(PruneArgs),

    Stats(StatsArgs),

    /// Read a grid (and its dual) and report what was found, without running anything.
    ///
    /// Use this to validate a Blender export before a long run: how many dual faces were linked
    /// to grid edges, how many were dropped as outer walls, and whether every edge has a face.
    GridCheck(GridCheckArgs),
}

#[derive(Debug, Args)]
struct RunArgs {
    #[arg(
        value_name = "complex.obj",
        help = "Path to the .obj file for the input complex."
    )]
    obj_path: PathBuf,

    #[arg(
        short,
        long,
        help = "Path to the .obj file for a grid mesh, optionally with its dual as a second object.",
        value_name = "mesh.obj"
    )]
    mesh_path: PathBuf,

    #[arg(
        short = 'd',
        long,
        help = "Path to an .obj file with the dual (Voronoi) faces of the grid, if they are not in the grid file.",
        value_name = "dual.obj"
    )]
    dual_path: Option<PathBuf>,

    #[arg(short, long, help = "Number of threads to run in parallel.")]
    threads: Option<usize>,

    #[arg(
        short,
        long,
        help = "Path to the output state file.",
        value_name = "STATE"
    )]
    output_path: Option<PathBuf>,

    #[arg(
        short,
        long,
        help = "Whether to pre-prune, with an optional prune file.",
        value_name = "PRUNE.json",
        num_args=0..=1
    )]
    prune: Option<Option<PathBuf>>,

    #[arg(short, long, help = "Don't include matrices in the output.")]
    slim: bool,
}

#[derive(Debug, Args)]
struct StatsArgs {
    #[arg(
        value_name = "state",
        help = "Path to the output state file from `mars-cli run`."
    )]
    state: PathBuf,
}

impl StatsArgs {
    fn run(&self) -> Result<()> {
        info!("Read state");
        let (mars, vin): (mars_core::Mars, mars_core::Vineyards) = {
            let f = std::fs::File::open(&self.state).context("open file")?;
            let mut reader = BufReader::new(f);
            rmp_serde::from_read(&mut reader).context("rmp read")?
        };

        let mm: MarsMem = (&mars).into();
        info!("{:?}", mm);

        for (_, r) in vin.reductions.iter().take(3) {
            let mem: ReductionMem = r.into();
            for dim in 0..3 {
                let idim = dim as isize;
                info!(dim);

                let D = &r.D(idim);
                info!(
                    bytes = mem.stacks[dim].D.core,
                    fill = r.D(idim).fill_ratio(),
                    size =? (D.rows(), D.cols()),
                    empty_cols = D.count_empty_columns(),
                    "D",
                );

                let R = &r.R(idim);
                info!(
                    bytes = mem.stacks[dim].R.core,
                    fill = r.R(idim).fill_ratio(),
                    size =? (R.rows(), R.cols()),
                    empty_cols = R.count_empty_columns(),
                    "R"
                );

                let U_t = &r.U_t(idim);
                info!(
                    bytes = mem.stacks[dim].U_t.core,
                    fill = r.U_t(idim).fill_ratio(),
                    size =? (U_t.rows(), U_t.cols()),
                    empty_cols = U_t.count_empty_columns(),
                    "U"
                );
                info!(ordering = mem.stacks[dim].ordering);
            }
        }

        // let vm: mars_core::stats::VineyardsMem = (&vin).into();
        // let rm = vm
        //     .reductions
        //     .into_iter()
        //     .map(|(i, r)| (size_of_val(&i), r))
        //     .fold((0, ReductionMem::default()), |(i, r), (ii, rr)| {
        //         (i + ii, r + rr)
        //     });
        // println!("{:#?}", rm);
        // for dim in 0..3 {
        //     info!(dim = dim, "swaps: {:>10}", vm.swaps[dim]);
        // }
        // dbg!(vm);

        Ok(())
    }
}

#[derive(Debug, Args)]
struct ObjArgs {
    #[arg(
        value_name = "state",
        help = "Path to the output state file from `mars-cli run`."
    )]
    state: PathBuf,

    #[arg(short, long, help = "Treat input file as a --slim output")]
    slim: bool,

    #[arg(
        short = 'c',
        long,
        help = "Output the complex as an .obj to this path.",
        value_name = "mesh.obj"
    )]
    complex: Option<PathBuf>,

    #[arg(
        short = 'g',
        long,
        help = "Output the grid to this path.",
        value_name = "grid.obj"
    )]
    grid: Option<PathBuf>,

    #[arg(
        short = 'a',
        long,
        help = "Output the medial axes to this path.",
        value_name = "ma.obj"
    )]
    medial_axes: Option<PathBuf>,
}

/// Lists of swaps for each dimension, per grid edge.
/// `(Index, Index, Vec<...>)` is two grid indices and the list of swaps between them.
/// `(Swap, f64, f64)` is the swap, plus the persistence lifetime of the swapped simplices
/// (0.0 if ).
type SlimFile = ([Vec<(Index, Index, Vec<(Swap, f64, f64)>)>; 3], Mars);

/// Read the grid mesh, and its dual from the same file or from `dual_path`, and check the report.
fn load_grid(mesh_path: &Path, dual_path: Option<&Path>) -> Result<(VineyardsGridMesh, GridReport)> {
    let obj_string = std::fs::read_to_string(mesh_path)
        .with_context(|| format!("failed to read grid path: {:?}", mesh_path))?;
    let (mut grid, mut report) = VineyardsGridMesh::read_from_obj_string_with_report(&obj_string)
        .map_err(|e| anyhow!(e))
        .context("failed to read grid")?;

    if let Some(dual_path) = dual_path {
        let dual_string = std::fs::read_to_string(dual_path)
            .with_context(|| format!("failed to read dual path: {:?}", dual_path))?;
        report = grid
            .merge_dual_from_obj_string(&dual_string)
            .map_err(|e| anyhow!(e))
            .context("failed to read dual")?;
    }

    for line in report.to_string().lines() {
        debug!("{}", line);
    }
    if report.has_dual() && report.faces_matched == 0 {
        bail!(
            "none of the {} dual faces could be linked to a grid edge; are the grid and its dual in the same coordinate system?",
            report.faces_total
        );
    }
    if report.has_dual() && report.faces_matched < report.faces_total / 2 {
        error!(
            "only {} of {} dual faces could be linked to a grid edge. Culled grids legitimately \
             have many outer walls, but this also happens when the dual was exported with other \
             axes than the grid, belongs to another lattice, or when objects were hidden during \
             export. Check with `mars-cli grid-check` before spending time on a run.",
            report.faces_matched, report.faces_total
        );
    }
    if report.edges_without_face > 0 {
        let msg = format!(
            "{} of {} grid edges have no dual face; swaps found along them will not appear in the output",
            report.edges_without_face, report.edges
        );
        if report.edges_without_face > report.edges / 2 {
            error!("{}. Is the dual the Voronoi tessellation of this lattice?", msg);
        } else {
            warn!("{}", msg);
        }
    }
    Ok((grid, report))
}

/// Write the medial axes as one `.obj` with objects `ma-dim-0`, `ma-dim-1`, `ma-dim-2`.
/// `edges[d]` lists the swap-bearing grid edges of dimension `d`; `face_of` is the dual polygon
/// of an edge (empty if it has none, in which case the edge is skipped and counted).
fn write_axes_obj(
    path: &Path,
    edges: [Vec<(Index, Index)>; 3],
    face_of: impl Fn(Index, Index) -> Vec<Pos>,
) -> Result<()> {
    info!("Write medial axes to {}", path.display());
    let mut f = BufWriter::new(std::fs::File::create(path).context("create passed file")?);
    let (mut vi, mut skipped) = (0, 0);
    for (dim, edges) in edges.iter().enumerate() {
        writeln!(f, "o ma-dim-{}", dim)?;
        for &(a, b) in edges {
            if !write_face(&mut f, &face_of(a, b), &mut vi)? {
                skipped += 1;
            }
        }
    }
    f.flush()?;
    if skipped > 0 {
        warn!(
            "{} swap-bearing grid edges had no dual face and were not written",
            skipped
        );
    }
    Ok(())
}

#[derive(Debug, Args)]
struct GridCheckArgs {
    #[arg(
        value_name = "grid.obj",
        help = "Path to the .obj file with the grid, optionally with its dual as a second object."
    )]
    mesh_path: PathBuf,

    #[arg(
        short = 'd',
        long,
        value_name = "dual.obj",
        help = "Path to a separate .obj file with the dual faces."
    )]
    dual_path: Option<PathBuf>,
}

impl GridCheckArgs {
    fn run(&self) -> Result<()> {
        let (_, report) = load_grid(&self.mesh_path, self.dual_path.as_deref())?;
        println!("{}", report);
        Ok(())
    }
}

impl ObjArgs {
    fn run_slim(&self) -> Result<()> {
        let bytes = std::fs::read(&self.state).context("read state file")?;
        let slim_file: SlimFile = rmp_serde::from_slice(&bytes).context("rmp read")?;

        let swaps = slim_file.0;
        let Some(Grid::Mesh(grid)) = slim_file.1.grid else {
            unimplemented!();
        };

        if let Some(ref p) = self.medial_axes {
            let edges = swaps.each_ref().map(|dim| {
                dim.iter()
                    .filter(|(_, _, s)| !s.is_empty())
                    .map(|(a, b, _)| (*a, *b))
                    .collect()
            });
            write_axes_obj(p, edges, |a, b| grid.dual_face_points(a, b))?;
        }

        if self.complex.is_some() || self.grid.is_some() {
            warn!("--complex and --grid are not available for --slim state files");
        }

        Ok(())
    }

    fn run(&self) -> Result<()> {
        info!("Reading input file {}", self.state.display());

        if self.slim {
            return self.run_slim();
        }

        let bytes = std::fs::read(&self.state).context("read state file")?;
        let (mars, vin): (mars_core::Mars, mars_core::Vineyards) =
            rmp_serde::from_slice(&bytes).context("rmp read")?;

        if let Some(ref p) = self.complex {
            let c = mars
                .complex
                .as_ref()
                .ok_or_else(|| anyhow!("missing complex in state"))?;

            let mut f = std::fs::File::create(p).context("create passed file")?;
            info!("Write complex to {}", p.display());
            c.write_as_obj(&mut f).context("write obj")?;
        }

        if let Some(ref p) = self.grid {
            let c = mars
                .grid
                .as_ref()
                .ok_or_else(|| anyhow!("missing grid in state"))?;

            let g = match c {
                mars_core::Grid::Regular(_) => {
                    error!("Trying to output a grid, but it is a regular grid.");
                    bail!("Trying to output a grid, but it is a regular grid.");
                }
                mars_core::Grid::Mesh(mesh) => mesh,
            };

            let mut f = std::fs::File::create(p).context("create passed file")?;
            info!("Write grid to {}", p.display());
            g.write_as_obj(&mut f).context("write obj")?;
        }

        if let Some(ref p) = self.medial_axes {
            let grid = mars
                .grid
                .as_ref()
                .ok_or_else(|| anyhow!("missing grid in state"))?;
            let edges = vin.swaps.each_ref().map(|dim| {
                dim.iter()
                    .filter(|s| !s.2.v.is_empty())
                    .map(|s| (s.0, s.1))
                    .collect()
            });
            write_axes_obj(p, edges, |a, b| match grid {
                Grid::Regular(g) => g.dual_quad_points(a, b).to_vec(),
                Grid::Mesh(g) => g.dual_face_points(a, b),
            })?;
        }

        Ok(())
    }
}

fn default_pruning_params() -> [PruningParam; 3] {
    let dim0 = PruningParam {
        euclidean: true,
        euclidean_distance: Some(0.01),
        coface: true,
        face: false,
        persistence: false,
        persistence_threshold: None,
    };
    let dim1 = PruningParam {
        euclidean: true,
        euclidean_distance: Some(0.01),
        coface: false,
        face: true,
        persistence: true,
        persistence_threshold: Some(0.01),
    };
    let dim2 = PruningParam {
        euclidean: true,
        euclidean_distance: Some(0.01),
        coface: false,
        face: true,
        persistence: false,
        persistence_threshold: None,
    };

    [dim0, dim1, dim2]
}

#[derive(Debug, Args)]
struct PruneArgs {
    #[arg(value_name = "state", help = "Path to the state file.")]
    state_path: PathBuf,

    #[arg(
        short,
        long,
        value_name = "pruned-state",
        help = "Path to the output pruned file."
    )]
    output: PathBuf,

    #[arg(
        short,
        long,
        help = "Whether to pre-prune, with an optional prune file",
        value_name = "PRUNE.json"
    )]
    params: Option<PathBuf>,

    #[arg(short, long, help = "Treat input file as a --slim output")]
    slim: bool,
}

impl PruneArgs {
    fn run_slim(&self) -> Result<()> {
        info!("Read parameters");
        let params = if let Some(ref p) = self.params {
            let f = std::fs::File::open(p).context("open file")?;
            let mut reader = BufReader::new(f);
            serde_json::from_reader(&mut reader).context("rmp read")?
        } else {
            default_pruning_params()
        };

        info!("Read state");
        let bytes = std::fs::read(&self.state_path).context("read state file")?;
        let (all_swaps, mars): SlimFile = rmp_serde::from_slice(&bytes).context("rmp read")?;

        info!("Prune");
        let mut all_pruned = [Vec::new(), Vec::new(), Vec::new()];
        for dim in 0..3 {
            let num_swaps = all_swaps[dim].iter().map(|s| s.2.len()).sum::<usize>();
            info!(dim = dim, "read {} swaps", num_swaps);
            let pruned = mars_core::prune::prune_dim(
                &all_swaps[dim],
                dim,
                &params[dim],
                mars.complex.as_ref().expect("Missing complex"),
                |i, n| {
                    if i % 127 == 0 {
                        info!(
                            dim = dim,
                            "pruning {}%",
                            ((i as f64 / n as f64) * 100.0).round()
                        );
                    }
                },
            );

            let num_left = pruned.iter().map(|s| s.2.len()).sum::<usize>();
            let num_pruned = num_swaps - num_left;
            info!(
                dim = dim,
                "pruned {} swaps ({}%). New count is {}",
                num_pruned,
                ((num_pruned as f64 / num_swaps as f64) * 100.0).floor(),
                num_left
            );
            all_pruned[dim] = pruned;
        }

        info!("Write output");
        let output = (all_pruned, mars);
        let output_bytes = rmp_serde::to_vec(&output)?;
        let mut f = std::fs::File::create(&self.output).context("create output file")?;
        f.write_all(&output_bytes).context("write to output file")?;

        Ok(())
    }

    fn run(&self) -> Result<()> {
        if self.slim {
            return self.run_slim();
        }
        info!("Read parameters");
        let params = if let Some(ref p) = self.params {
            let f = std::fs::File::open(p).context("open file")?;
            let mut reader = BufReader::new(f);
            serde_json::from_reader(&mut reader).context("rmp read")?
        } else {
            default_pruning_params()
        };

        info!("Read state");
        let (mars, mut vin): (mars_core::Mars, mars_core::Vineyards) = {
            let f = std::fs::File::open(&self.state_path).context("open file")?;
            let mut reader = BufReader::new(f);
            rmp_serde::from_read(&mut reader).context("rmp read")?
        };

        let complex = mars
            .complex
            .as_ref()
            .ok_or_else(|| anyhow!("Missing complex in state"))?;

        info!("Prune");
        for dim in 0..3 {
            let num_swaps = vin.swaps[dim].iter().map(|s| s.2.v.len()).sum::<usize>();
            info!(dim = dim, "read {} swaps", num_swaps);
            let pruned = vin.prune_dim(dim, &params[dim], complex, |i, n| {
                if i % 127 == 0 {
                    info!(
                        dim = dim,
                        "pruning {}%",
                        ((i as f64 / n as f64) * 100.0).round()
                    );
                }
            });

            let num_left = pruned.iter().map(|s| s.2.v.len()).sum::<usize>();
            let num_pruned = num_swaps - num_left;
            info!(
                dim = dim,
                "pruned {} swaps ({}%). New count is {}",
                num_pruned,
                ((num_pruned as f64 / num_swaps as f64) * 100.0).floor(),
                num_left
            );
            vin.swaps[dim] = pruned;
        }

        info!("Write output");
        let output = (mars, vin);
        let output_bytes = rmp_serde::to_vec(&output)?;
        let mut f = std::fs::File::create(&self.output).context("create output file")?;
        f.write_all(&output_bytes).context("write to output file")?;

        Ok(())
    }
}

fn print_prune_config() -> Result<()> {
    let cfgs = default_pruning_params();
    let string = serde_json::to_string_pretty(&cfgs)?;
    println!("{}", string);
    Ok(())
}

impl RunArgs {
    fn run_slim(&self) -> Result<()> {
        if self.prune.is_some() {
            bail!("Cannot prune and --slim at the same time");
        }
        let complex = Complex::read_from_obj_path(&self.obj_path)
            .map_err(|e| anyhow!(e))
            .context("failed to read complex")?;

        let (mesh_grid, _report) = load_grid(&self.mesh_path, self.dual_path.as_deref())?;

        let mars = mars_core::Mars {
            complex: Some(complex),
            grid: Some(mars_core::Grid::Mesh(mesh_grid)),
        };

        use rayon::prelude::*;
        let mut parts: Vec<_> = {
            let parts = mars.split_into_4().map_err(|e| anyhow!(e))?;
            parts
                .par_iter()
                .enumerate()
                .map(|(k, sub)| {
                    sub.run_slim(|i, n| {
                        if i == 0 {
                            return;
                        }
                        let p = (i as f64 / n as f64 * 100.0).round();
                        let pprev = ((1.0 + i as f64) / n as f64 * 100.0).round();
                        let on_step = p != pprev;
                        if on_step || i == n {
                            info!(?k, "{:3}%", p);
                        }
                    })
                })
                .collect()
        };

        let mut joined = [Vec::new(), Vec::new(), Vec::new()];
        for part in &mut parts {
            let Ok(part) = part else {
                error!("Part failed: {:?}", part);
                continue;
            };
            for dim in 0..3 {
                joined[dim].extend(std::mem::take(&mut part[dim]));
            }
        }

        let output: SlimFile = (joined, mars);

        let output_bytes = rmp_serde::to_vec(&output)?;

        if let Some(ref path) = self.output_path {
            let mut f = std::fs::File::create(path).context("create output file")?;
            f.write_all(&output_bytes).context("write to output file")?;
            info!("Wrote output to {}", path.display());
        } else {
            std::io::stdout()
                .write_all(&output_bytes)
                .context("write to stdout")?;
            info!("Wrote output to stdout");
        }

        Ok(())
    }
}

fn run(args: &RunArgs) -> Result<()> {
    if args.slim {
        return args.run_slim();
    }

    let complex = Complex::read_from_obj_path(&args.obj_path)
        .map_err(|e| anyhow!(e))
        .context("failed to read complex")?;

    let (mesh_grid, _report) = load_grid(&args.mesh_path, args.dual_path.as_deref())?;

    let mars = mars_core::Mars {
        complex: Some(complex),
        grid: Some(mars_core::Grid::Mesh(mesh_grid)),
    };

    use rayon::prelude::*;
    let vin: Vec<_> = {
        let parts = mars.split_into_4().map_err(|e| anyhow!(e))?;
        parts
            .par_iter()
            .enumerate()
            .map(|(k, sub)| {
                sub.run(|i, n| {
                    if i == 0 {
                        return;
                    }
                    let p = (i as f64 / n as f64 * 100.0).round();
                    let pprev = ((1.0 + i as f64) / n as f64 * 100.0).round();
                    let on_step = p != pprev;
                    if on_step || i == n {
                        info!(?k, "{:3}%", p);
                    }
                })
            })
            .collect()
    };

    let mut vin = vin
        .into_iter()
        .reduce(|a, e| {
            let mut a = a?;
            a.add_other(e?);
            Ok(a)
        })
        .expect("should be four vineyards")
        .map_err(|e| anyhow!(e))?;

    info!("Bake matrices");
    for r in vin.reductions.values_mut() {
        r.bake_all_matrices();
    }

    if let Some(ref prune) = args.prune {
        info!("Prune output");
        let prune_params = if let Some(path) = prune {
            info!("Use pruning parameters from {}", path.display());
            let file_contents = std::fs::read_to_string(path).context("read prune file")?;
            let params: [PruningParam; 3] =
                serde_json::from_slice(file_contents.as_bytes()).context("read json")?;
            params
        } else {
            info!("Use default pruning parameters");
            default_pruning_params()
        };

        let c = mars.complex.as_ref().unwrap();

        let mut swaps = [Vec::new(), Vec::new(), Vec::new()];
        for dim in 0..3 {
            let num_swaps = vin.swaps[dim].iter().map(|s| s.2.v.len()).sum::<usize>();
            let pruned = vin.prune_dim(dim, &prune_params[dim], &c, |i, n| {
                if i % 511 == 0 {
                    let percent = (i as f64 / n as f64) * 100.0;
                    info!("prune dim {dim}: {percent:3.0}%");
                }
            });
            let num_left = pruned.iter().map(|s| s.2.v.len()).sum::<usize>();
            let num_pruned = num_swaps - num_left;
            info!(
                dim = dim,
                "pruned {} swaps ({}%). New count is {}",
                num_pruned,
                ((num_pruned as f64 / num_swaps as f64) * 100.0).floor(),
                num_left
            );
            swaps[dim] = pruned;
        }

        vin.swaps = swaps;
    }

    info!("Write output");
    let output = (mars, vin);
    let output_bytes = rmp_serde::to_vec(&output)?;

    if let Some(ref path) = args.output_path {
        let mut f = std::fs::File::create(path).context("create output file")?;
        f.write_all(&output_bytes).context("write to output file")?;
        info!("Wrote output to {}", path.display());
    } else {
        std::io::stdout()
            .write_all(&output_bytes)
            .context("write to stdout")?;
        info!("Wrote output to stdout");
    }

    Ok(())
}

fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .with_writer(std::io::stderr)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("set global default subscriber failed");

    let cli = Cli::parse();
    match cli.command {
        Sub::PrintPrune => print_prune_config(),
        Sub::Run(r) => run(&r),
        Sub::Obj(o) => o.run(),
        Sub::Prune(p) => p.run(),
        Sub::Stats(s) => s.run(),
        Sub::GridCheck(g) => g.run(),
    }
}
