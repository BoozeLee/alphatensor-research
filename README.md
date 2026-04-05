# AlphaTensor Research

Personal research workspace exploring DeepMind's AlphaTensor — an AI system that discovered
novel matrix multiplication algorithms beating human-designed ones for the first time since
Strassen (1969).

## What's here

- `algorithms/` — DeepMind's discovered factorizations (real & F2 fields)
- `explore.py` — benchmark and verify factorizations vs numpy/JAX
- `benchmarking/` — original DeepMind benchmarking code
- `recombination/` — recombination experiments from the paper

## Setup

```bash
uv sync
uv run python explore.py
```

Requires Python 3.12. GPU benchmarking uses JAX with CUDA 12.

## Key findings (from the paper)

| Matrix size | Naive ops | AlphaTensor (R) | AlphaTensor (F2) |
|-------------|-----------|-----------------|------------------|
| 2×2×2       | 8         | 7               | 7                |
| 3×3×3       | 27        | 23              | 23               |
| 4×4×4       | 64        | 49              | **47**           |
| 5×5×5       | 125       | 98              | **96**           |
| 10×10×10    | 1000      | 682             | —                |

## Reference

- Paper: [Discovering faster matrix multiplication algorithms with reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4) — Nature, 2022
- Original repo: [google-deepmind/alphatensor](https://github.com/google-deepmind/alphatensor)

## License

MIT — see [LICENSE](LICENSE). Note: the factorization data in `algorithms/` is from DeepMind
and subject to their [Apache 2.0 license](https://github.com/google-deepmind/alphatensor/blob/main/LICENSE).
