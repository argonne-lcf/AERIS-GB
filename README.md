# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions

![](https://github.com/argonne-lcf/AERIS-GB/blob/main/media/aeris-github-cover.light.png#gh-light-mode-only)
![](https://github.com/argonne-lcf/AERIS-GB/blob/main/media/aeris-github-cover.dark.png#gh-dark-mode-only)

Frozen snapshot of the AERIS code used for the 2025 ACM Gordon Bell Prize for Climate Modeling submission. This repository is provided as a reference. The code has been cleaned and unrelated experiments have been removed but has not been tested after cleanup.

## Configurations

- **Benchmark**: Minimal startup overhead with most performant settings
- **Train**: Full training setup used for most science results with conservative performance optimizations
- **Hurricane**: Setup used for the 6-hour hurricane tracks

## Key Components

- [`parallel_engine.py`](src/aeris/parallelism/parallel_engine.py) — Pipeline and sequence parallel training engine
- [`parallel_swin.py`](src/aeris/models/parallel_swin.py) — Swin Transformer with sequence parallelism

## WP-Only Inference

`scripts/wp_inference.sh` provides an entry point for running window-partitioning-only inference without pipeline, sequence, or data parallelism. The implementation can be found in `src/aeris_wp_inference`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite [our paper](https://dl.acm.org/doi/full/10.1145/3712285.3772094):

```bibtex
@inproceedings{hatanpaa2025aeris,
  title={Aeris: Argonne earth systems model for reliable and skillful predictions},
  author={Hatanp{\"a}{\"a}, V{\"a}in{\"o} and Ku, Eugene and Stock, Jason and Emani, Murali and Foreman, Sam and Jung, Chunyong and Madireddy, Sandeep and Nguyen, Tung and Sastry, Varuni and Sinurat, Ray AO and others},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={72--85},
  year={2025}
}
```
