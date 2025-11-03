# AERIS-GB
Frozen snapshot of AERIS code used for the 2025 submission for ACM Gordon Bell Prize for Climate Modeling

This code is difficult to run and not intended to be used by anyone. The code has been cleaned and a lot of unrelated experiments have been removed. The code is not tested after cleanup. This repository should only be used as a reference.

The presented configs:
- Benchmark: minimal startup overhead and most performant settings
- Train: Full training setup used for most science results. More conservative performance optimizations for long-term stability
- Train-6h-Hurricane: Full training setup used for the 6h hurricane tracks.

Most of the interesting parts are in:
- `src/aeris/parallelism/parallel_engine.py`
- `src/aeris/models/parallel_swin.py`
## WP-only inference
`scripts/wp_inference.sh` contains an entry point for running simpler window-partitioning only inference with no pipeline, sequence, or data parallelism. The code can be found at `src/aeris_wp_inference`
