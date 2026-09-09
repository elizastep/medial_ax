# Mars Core


## Testing

We have snapshot tests using [insta][insta]. These are checked with `cargo test`
(use `--release`, the vineyards runs are slow in debug). Snapshot files live in
`src/snapshots/` and are named after the crate (`mars_core__...`); if the crate is
renamed they all have to be renamed too, or insta treats every snapshot as new.

To overwrite old tests, run

```sh
INSTA_UPDATE=always cargo test --release 
```

The grid tests in `src/grid.rs` build SC/BCC/FCC lattices with their Voronoi
tessellation in `src/test.rs` (the same construction as
`blender_scripts/cubic_lattices.py`) and write them as `.obj` text, so the
`.obj` reader and the dual-face matching are tested end to end without Blender.

[insta]: https://insta.rs/docs/quickstart/
