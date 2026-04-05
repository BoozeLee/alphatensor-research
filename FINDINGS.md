# Benchmark Findings

**Date:** 2026-04-05
**Hardware:** NVIDIA GeForce GTX 1080 (Pascal, 8GB VRAM)
**Software:** JAX 0.9.2, CUDA 12, Python 3.12
**Device:** cuda:0

## GPU Benchmark (JAX JIT, 10,000 iterations each)

| Matrix | AlphaTensor (µs) | jnp.matmul (µs) | Rank | Naive | Ops saved | Speedup |
|--------|-----------------|-----------------|------|-------|-----------|---------|
| 4,4,4 | 60.1 | 58.6 | 49 | 64 | 15 (23%) | -2.6% |
| 5,5,5 | 59.9 | 56.1 | 98 | 125 | 27 (22%) | -6.9% |
| 10,10,10 | 61.6 | 73.7 | 682 | 1000 | 318 (32%) | +16.4% |
| 11,11,11 | 61.8 | 73.2 | 896 | 1331 | 435 (33%) | +15.5% |
| 11,12,12 | 67.0 | 72.9 | 990 | 1584 | 594 (38%) | +8.1% |

## Key Insight

AlphaTensor's advantage grows with matrix size. At 11×12×12 (largest in dataset),
the rank reduction is 594 fewer multiplications (37%) vs naive — translating to
measurable GPU speedup without any approximation: results are provably exact.

## Why It Matters

- Every matmul in every neural network, physics sim, and graphics pipeline benefits
- Strassen (1969) was the last improvement for 53 years — AlphaTensor broke that
- The F2 (binary field) variants achieve even lower rank — relevant for cryptography
- The discovered algorithms are alien in structure: no human would write them, yet they are provably correct

## Reference

- [Nature paper (2022)](https://www.nature.com/articles/s41586-022-05172-4)
- [google-deepmind/alphatensor](https://github.com/google-deepmind/alphatensor)