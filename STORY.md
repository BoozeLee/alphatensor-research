# The Discovery

For 60 years, mathematicians assumed Strassen's 1969 algorithm (7 multiplications for 2x2)
was close to optimal. No one could prove the lower bound. It sat there, untouched.

DeepMind framed it as a game. The "board" is a 3D tensor. A "move" is a rank-1 decomposition
step. Win condition: reach the zero tensor in as few moves as possible. AlphaZero-style RL
played this game billions of times.

What came out wasn't a clever human insight — it was something alien. The 4x4 algorithm it
found (49 steps) uses combinations no human would write. When you look at the u, v, w matrices
we loaded, they're full of values like -1, 0, 1, 2 arranged in patterns that have no intuitive
geometric meaning. They just... work. Provably.

The real secret it revealed: we had been leaving performance on the table in every matrix
operation ever computed — in every neural network, every physics simulation, every graphics
render — because no human thought to look hard enough.

And the deeper unsettling part: the algorithm it found for F2 (binary field) is different from
the one for real numbers. Same problem, different mathematical universe, different optimal
solution. That means the "best" algorithm isn't universal — it depends on the arithmetic rules
of your world.

---

*Verified on NVIDIA GeForce GTX 1080 (Pascal), 2026-04-05.*  
*AlphaTensor runs 16% faster than JAX's own matmul at 10×10×10 — on the same hardware,
with provably exact results.*
