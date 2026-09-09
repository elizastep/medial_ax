# Medial Axes

## Live Demo: 
We are hosting a working demo version at https://medial-ax.github.io/medial-ax/! Try it out! We have some example complexes you can play around with already, or you can upload your own.

## Local Setup
Note: We have mainly tested on Mac, but it should work on Linux and Windows as well, although you will have to use OS specific package managers and package install directions. 

To install the things needed to run the web app you first need to install
`node`. With the package manager `brew` on Mac, for example, this is

```sh
brew install node
```

or you can visit https://nodejs.org/en/download.

### Install Rust 🦀
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
or visit https://www.rust-lang.org/learn/get-started.

### Install Wasm-pack

Wasm-pack is the tool we use to generate webassembly from Rust code.
To install wasm-pack, run

```sh
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### Run the frontend
(note! if you only want to use the cli, you don't need to build the frontend)
Clone the repo, and then go to the `web/` directory and run

```sh
npm i # install the things. Only need the first time.
npm run dev # start the server
```

It'll tell you to which URL to go to to open the page; probably it's
[http://localhost:5173](http://localhost:5173). While the server is running you
get live edit of all the files.

#### Build the wasm bindings

This is required every time you change the Rust code, and want that change to
be in the frontend. Go to the `mars-wasm/` directory and run

```shell
wasm-pack build --target web
```

This will output a bunch of stuff to `mars-wasm/pkg`, but you don't have to worry about that.

### Build the CLI

For convenience or large jobs, we also have a cli in `mars-cli/`.
To install it as a binary you can use, go to the directory and run

```sh
cargo install --path .
```

Now the binary `mars-cli` should be possible to use in the terminal. It's
builtin help is the most up-to-date info on how to use it. At the time of
writing, it looks like this:

```sh
mars-cli
Usage: mars-cli <COMMAND>

Commands:
  print-prune  Print the default parameters used for pruning
  run          Run the algorithm and output a file containing the entire state
  obj          Output .obj files from the state file
  prune        Prune swaps from a state file
  stats
  grid-check   Read a grid (and its dual) and report what was found, without running anything
  help         Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print versionUsage: mars-cli <COMMAND>
```
#### try an example
Navigate to ```mars-cli``` and run the example bash script by executing ```sh ../examples/shell.sh```. This exports the 0th and 1st medial axes of a squished cylinder. You can compare them against ```examples/cylinder_ma_original.obj``` to check that you get the same output that we did. Notice that the pruning parameters in ```examples/prune_cylinder.txt``` are not the same for every dimension-- you just have to play around with the pruning parameters for each use case.

#### run your own files
To run the cli, you will need a triangulated mesh as an .obj, a grid as an .obj (vertices and edges, optionally with its Voronoi dual as a second object; see [Grid](#grid) below), a pruning file .txt as outlined below if you don't want to use the automatic pruning parameters, and, optionally, a bash file so you don't have to keep rewriting commands in the terminal.

Computing is done as follows. The first line spits out the complex, the second prunes it with custom prune settings you have set in prune_settings.txt, and the third makes an .obj file out of the pruned medial axes. All three axes will be in one .obj file as separate objects; each face is the dual face of one grid edge (a square for a cubic grid, a hexagon or a square for a BCC grid, a rhombus for FCC). It is highly recommended to use the -s option (slim) which doesn't use as much memory or storage.

```sh
mars-cli run complex.obj -m grid.obj -s -o complex_out.txt
&& mars-cli prune -s complex_out.txt -o complex_out_pruned.txt -p prune_settings.txt
&& mars-cli obj -s complex_out_pruned.txt -a complex_ma.obj
```
Note that you can just run the last two lines if you want to reprune with different parameters and didn't change the complex or grid.

Before a long run, `mars-cli grid-check grid.obj` reads the grid the same way `run` does and prints what it found: points, edges, the degree histogram and, if a dual is present, how many dual faces were linked to grid edges, how many were dropped as outer walls, and whether any edge is missing a face. If the dual lives in a separate file, pass it with `-d dual.obj` to both `grid-check` and `run`.

An example pruning file ```prune_settings.txt``` could look like this: 
```
[
  {
    "euclidean": true,
    "euclidean_distance": 0.01,
    "coface": true,
    "face": false,
    "persistence": false,
    "persistence_threshold": null
  },
  {
    "euclidean": true,
    "euclidean_distance": 0.01,
    "coface": false,
    "face": true,
    "persistence": true,
    "persistence_threshold": 0.01
  },
  {
    "euclidean": true,
    "euclidean_distance": 0.01,
    "coface": false,
    "face": true,
    "persistence": false,
    "persistence_threshold": null
  }
]
```

#### run locally to upload to web interface
If you want to load your output to the web interface, you can run the following shorter command:

```sh
mars-cli run  complex.obj  -m grid.obj  -o output
```

The output file `output` can then be uploaded in the web interface. See
`mars-cli run --help` for more options. Note that without the -s option, it will be slower and take more storage and memory space.

# Usage

## Input

### Complex
The user inputs an .obj file containing a three-dimensional simplicial complex.

### Grid
The grid is a graph of vertices and edges in space. The algorithm walks its edges; every edge along which a Faustian swap is found contributes its *dual face* to the output. There are two ways to supply one.

#### Cubic grid (vertices and edges only)
You can import a grid you have made yourself, for example in Blender, and exported as an .obj. Our favorite way to make a grid in Blender is to start with a cube, use three array modifiers to fit it to your object in 3 dimensions, apply the modifiers, deduplicate vertices, and delete all faces (leaving edges and vertices). If you wish, you can import ```blender_scripts/select_and_delete.py``` as a blender script to delete the grid vertices outside of your object to not waste computation time. Warning: we haven't tested the select_and_delete script very much, and blender can be finnicky. It works most of the time ⚠️, and it requires that the input object be closed, as it uses raycasting. In particular, it won't work on the squished cylinder example. A good heuristic for grid density is to have at least two grid cubes per input complex face.

Without a dual, every edge must be parallel to a coordinate axis: the dual face of an edge is then the axis-aligned square that bisects it, sized by the lattice spacing.

#### Any cubic lattice with its Voronoi dual (SC, BCC, FCC)
For non-cubic grids the dual faces are no longer squares, so they are supplied together with the grid. `blender_scripts/cubic_lattices.py` is a Blender add-on that builds simple cubic, body-centred cubic (BCC) and face-centred cubic (FCC) lattices together with their Voronoi tessellation (cubes, truncated octahedra and rhombic dodecahedra respectively), and exports both in one file. Install it via *Edit > Preferences > Add-ons > Install*, or open it in Blender's text editor and run it. Then:

1. Select the input complex and choose *Add > Mesh > Cubic Lattice*, or use the *Lattice* tab in the sidebar (N).
2. In the redo panel (F9) tick *Fit to object*, pick the lattice type and the spacing (either the side length `a` of the conventional cell or a *density*: grid edges per average triangle edge). *Cull to inside* drops lattice points outside a closed object, together with their edges and Voronoi walls. Leave *Voronoi faces* on *All faces* and *Bonds = Voronoi walls* on: the lattice edges are then exactly the pairs of points that share a wall (for BCC that includes the six axis neighbours, whose walls are the squares of the truncated octahedron). An edge without a wall is walked by the algorithm but its swaps have no face to be drawn on; `grid-check` reports such edges.
3. With the lattice selected, press *Export for mars (.obj)*. This writes one .obj with two objects, `<kind>_lattice` (vertices + `l` edges) and `<kind>_voronoi` (the Voronoi walls as n-gon faces), without normals, UVs or triangulation. It exports with **X forward, Z up**, the way we export complexes; if you export your complexes with other axes, set the same ones in the export dialog's sidebar, otherwise the grid is rotated relative to the complex. The operator refuses to export when no Voronoi object (an object with faces) is among the selected lattice, its children and its `<kind>_voronoi` partner.

You can edit both objects freely in Blender before exporting; the only contract is that the file contains an object with `l` lines (the grid) and an object with `f` lines and no `l` lines (the dual), in the same coordinate system. The dual may also come in a separate file (`-d dual.obj`). `mars-cli` links every dual face to the grid edge it bisects by geometry: the two grid points nearest to the face's centroid are its edge, and all its vertices must be equidistant from them. Faces that fail this test are the outer walls of the boundary cells and are dropped; faces that appear twice are merged; a triangulated dual is rejected. The grid's edge set is the union of the `l` lines and the matched faces, so the `l` lines are optional when a dual is present. Check the result with `mars-cli grid-check`.

# What's happening on the inside

## Creating dual grids

In the web interface we create a grid aligned to the x,y,z axes by taking the bounding box of the imported .obj and subdividing it according to the selected grid density. The grid can afterwards be moved around and adjusted manually by the user using the Grid Controls context. We refer to the created grid as the Vineyards Grid and its dual grid as the Medial Axis Grid. The grid we visualize in the display window is the Vineyards Grid.

The Medial Axis Grid is the Voronoi tessellation of the grid vertices: the dual face of a grid edge is the Voronoi wall that separates its two endpoints, which lies in the edge's perpendicular bisector plane. For a cubic grid that wall is a square and we compute it directly. For other lattices the walls are supplied with the grid (see [Grid](#grid)) and linked to the edges by the nearest-two-sites rule described there; the state file then carries the wall polygons, so `mars-cli obj` writes them out as n-gons. The web interface still only knows cubic grids.

## Splitting grids for parallelization

Computing the medial axes is very parallelizable, since processing each grid segment is independent of the other segments.
However, since processing a segment requires that one endpoint has the reduced matrix data computed, it is not trivially parallelizable.
We take the input grid and split the vertices into four sub-grids by dividing along the two coordinate axes with the largest span.
There's also an overlap of size one when splitting so that both subgrids include the vertices on the boundary.
This is required in order not to lose the segments that would otherwise fall in between two sub-grids.
Then, the medial axes for each sub grid is computed separately, and then the results are joined up after.

## Sneaky Matrices and other optimizations

We have a tailored matrix struct for our own need.

- We only do Z/Z2 math, so we only have Boolean coefficients
- Only the operations that we needed were implemented
- The maximum size is limited to 2^15

We use a sparse column format, in which each column contains the indices of the rows that are set in the column.
That is, a 4x4 identity matrix basically looks like

```rs
[[0], [1], [2], [3]]
```

The Vineyards algorithm swaps a lot of rows and columns of matrices.
To make this efficient, we've wrapped two permutations, one for row and one for coulmn, around each matrix.
The permutations map from "logical" indices (what we think the matrix contains) into "physical" indices (what's actually stored).
This allows us to swap columns and rows by only swapping two numbers in the permutation instead of in the actual matrix storage.

See `sneaky_matrix.rs` for more details.

## use heuristics

- The input complex cannot exceed 32,000 simplices of any dimension due to being stored as 16-bit signed numbers
- 30,000 total edges (counting both grid and input) usually gives a nice result, as long as the object is not too complicated
- Size the grid such that at least two grid cells fit in an average triangle
- Running time scales with the number of grid edges. At the same nearest-neighbour distance a BCC grid has about 1.3 times the points and 14 instead of 6 edges per point, so roughly 3 times the edges of a cubic grid; FCC has 12 edges per point and about 1.4 times the points, so roughly 3 times as well. At the same conventional cell side `a`, BCC has about 4.7 times and FCC about 8 times the edges of the cubic grid.
