import numpy as np

factorizations_r  = dict(np.load('algorithms/factorizations_r.npz',  allow_pickle=True))
factorizations_f2 = dict(np.load('algorithms/factorizations_f2.npz', allow_pickle=True))

# Show all available factorizations and their rank (= number of multiplications)
print("=== Real (R) factorizations ===")
for key in sorted(factorizations_r):
    u, v, w = factorizations_r[key]
    a, b, c = map(int, key.split(','))
    standard_rank = a * b * c  # naive algorithm cost
    print(f"  {key}: rank={u.shape[-1]}  (naive={standard_rank}, saved={standard_rank - u.shape[-1]})")

print("\n=== F2 (binary field) factorizations ===")
for key in sorted(factorizations_f2):
    u, v, w = factorizations_f2[key]
    a, b, c = map(int, key.split(','))
    standard_rank = a * b * c
    print(f"  {key}: rank={u.shape[-1]}  (naive={standard_rank}, saved={standard_rank - u.shape[-1]})")

def get_matmul_tensor(a, b, c):
    """The matrix multiplication tensor T_{a,b,c}."""
    t = np.zeros((a*b, b*c, c*a), dtype=np.int32)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                t[i*b+j, j*c+k, k*a+i] = 1
    return t

def verify(key, factorizations, mod=None):
    u, v, w = factorizations[key]
    a, b, c = map(int, key.split(','))
    tensor = get_matmul_tensor(a, b, c)
    recon  = np.einsum('ir,jr,kr->ijk', u, v, w)
    if mod:
        recon = np.mod(recon, mod)
    ok = np.array_equal(tensor, recon)
    print(f"  {key} ({'R' if mod is None else 'F2'}): {'✓ correct' if ok else '✗ WRONG'}")

print("\n=== Verification (sample) ===")
for key in ['2,2,2', '3,3,3', '4,4,4', '4,5,5']:
    if key in factorizations_r:
        verify(key, factorizations_r)
    if key in factorizations_f2:
        verify(key, factorizations_f2, mod=2)

# Demonstrate using AlphaTensor's 4x4 algorithm to actually multiply two matrices
def alphatensor_matmul(A, B, key='4,4,4'):
    """Multiply matrices using AlphaTensor's factorization. A is (a,b), B is (b,c)."""
    u, v, w = factorizations_r[key]
    # u: (a*b, rank), v: (b*c, rank), w: (c*a, rank)
    scalars = (A.flatten() @ u) * (B.flatten() @ v)  # (rank,)
    a, b, c = map(int, key.split(','))
    C = (w @ scalars).reshape(c, a).T  # (a, c)
    return C

print("\n=== matmul via AlphaTensor vs numpy ===")
for key in ['2,2,2', '3,3,3', '4,4,4']:
    a, b, c = map(int, key.split(','))
    A = np.random.randint(0, 5, (a, b)).astype(float)
    B = np.random.randint(0, 5, (b, c)).astype(float)
    C_alpha = alphatensor_matmul(A, B, key)
    C_numpy = A @ B
    err = np.max(np.abs(C_alpha - C_numpy))
    rank = factorizations_r[key][0].shape[-1]
    print(f"  {key}: max_error={err:.1e}  rank={rank} (naive={a*b*c})")
