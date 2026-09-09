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
  help         Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print versionUsage: mars-cli <COMMAND>
```
#### try an example
Navigate to ```mars-cli``` and run the example bash script by executing ```sh ../examples/shell.sh```. This exports the 0th and 1st medial axes of a squished cylinder. You can compare them against ```examples/cylinder_ma_original.obj``` to check that you get the same output that we did. Notice that the pruning parameters in ```examples/prune_cylinder.txt``` are not the same for every dimension-- you just have to play around with the pruning parameters for each use case.

#### run your own files
To run the cli, you will need a triangulated mesh as an .obj, a grid made up of vertices and edges as an .obj, a pruning file .txt as outlined below if you don't want to use the automatic pruning parameters, and, optionally, a bash file so you don't have to keep rewriting commands in the terminal.

Computing is done as follows. The first line spits out the complex, the second prunes it with custom prune settings you have set in prune_settings.txt, and the third makes an .obj file out of the pruned medial axes. All three axes will be in one .obj file as separate objects. It is highly recommended to use the -s option (slim) which doesn't use as much memory or storage.

```sh
mars-cli run complex.obj -m grid.obj -s -o complex_out.txt
&& mars-cli prune -s complex_out.txt -o complex_out_pruned.txt -p prune_settings.txt
&& mars-cli obj -s complex_out_pruned.txt -a complex_ma.obj
```
Note that you can just run the last two lines if you want to reprune with different parameters and didn't change the complex or grid.

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
You can import a grid you have made yourself, for example in Blender, and exported as an .obj. Our favorite way to make a grid in Blender is to start with a cube, use three array modifiers to fit it to your object in 3 dimensions, apply the modifiers, deduplicate vertices, and delete all faces (leaving edges and vertices). If you wish, you can import ```blender_scripts/select_and_delete.py``` as a blender script to delete the grid vertices outside of your object to not waste computation time. Warning: we haven't tested the select_and_delete script very much, and blender can be finnicky. It works most of the time ⚠️, and it requires that the input object be closed, as it uses raycasting. In particular, it won't work on the squished cylinder example. A good heuristic for grid density is to have at least two grid cubes per input complex face.

### Dual mesh (optional, `mars-cli` only)
Without a dual mesh, the medial axis face for a grid edge is an axis-aligned quad, sized from the
grid spacing that we infer from the .obj.  That is only the right shape when the grid is a cubic
one, but not in general.

For other grids, pass the dual of the grid as a second .obj with `-d`:

```sh
mars-cli run complex.obj -m lattice.obj -d dual.obj -s -o output
```

The medial axes written by `mars-cli obj` are then made of the faces from `dual.obj`.  We assume
that each edge in `lattice.obj` intersects at most one face in `dual.obj` and that those edges and
faces correspond to each other.  This is only true for certain dual meshes.

# What's happening on the inside

## Creating dual grids

We create a grid aligned to the x,y,z axes by taking the bounding box of the imported .obj and subdividing it according to the selected grid density. The grid can afterwards be moved around and adjusted manually by the user using the Grid Controls context. We refer to the created grid as the Vineyards Grid and its dual grid as the Medial Axis Grid. The grid we visualize in the display window is the Vineyards Grid.

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
