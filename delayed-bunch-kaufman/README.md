# Delayed-pivot Bunch--Kaufman

C++17 implementation of a delayed-pivot Bunch--Kaufman factorization for
column-major, real symmetric matrices. Only the lower triangle is read and
overwritten. Schur-complement updates use the symmetric CBLAS kernels. The
in-place result satisfies

```text
P^T * A * P = L * D * L^T.
```

## Interface

```cpp
void delayed_sytrf(int n, double* A, int lda,
                   int* perm, int* piv_size, int& info);

void delayed_sytrf_blocked(int n, double* A, int lda,
                           int* perm, int* piv_size, int& info);

void lapack_sytrf(int n, double* A, int lda,
                  int* perm, int* piv_size, int& info);
```

All exposed indices are zero-based. `A(i,j)` is stored at `A[i+j*lda]`.
Only entries with `i>=j` must be initialized. The upper triangle is ignored
and is not synchronized with the lower factor storage.

- `perm[pos]` is the original node now stored at position `pos`.
- `piv_size[pos] == 1` marks a 1x1 pivot.
- `piv_size[pos] == 2` marks the beginning of a 2x2 pivot.
- `piv_size[pos] == 0` marks its second position or a delayed position.
- `[0,info)` is eliminated and `[info,n)` is delayed.
- `info == n` means no node was delayed.
- `info == -1` means dimensions or pointers were invalid.

`delayed_sytrf_blocked` has exactly the same storage and output semantics as
`delayed_sytrf`. It accumulates mixed 1x1/2x2 pivots in panels of up to 64
columns. A heap-backed workspace stores `W=L_panel*D_panel`; pivot columns and
candidate rows are materialized lazily with `dgemv`, and `dsyr2k` applies the
whole panel update to the lower trailing matrix.

The third function is a C++ wrapper around LAPACK `DSYTRF` with `UPLO='L'`.
For this wrapper, `info==n` means LAPACK completed without a singular `D`
block; `info<n` is the first singular position converted to zero-based.
LAPACK continues after that position, so its `info` does not describe a
delayed suffix. After factorization, LAPACK `DSYCONV` separates the off-diagonal
entries of 2x2 `D` blocks and applies the pivot interchanges to the multiplier
rows. The wrapper then exposes the same explicit `L`, `D`, `perm`, and
`piv_size` presentation as `delayed_sytrf`.

The algorithm maintains three contiguous ranges:

```text
[0,k)             eliminated
[k,active_end)    pivot candidates
[active_end,n)    delayed nodes
```

When neither a stable 1x1 nor 2x2 pivot can be selected, node `k` is exchanged
with `active_end-1`, then `active_end` is decremented. A new delayed node is
therefore inserted immediately before the existing delayed suffix. It is never
blindly exchanged with `n-1`, which could replace an already delayed node.

## Implementation optimizations

The factorization stores and updates only the lower triangle:

- a 1x1 Schur update uses `cblas_dsyr`, replacing a full `dger` update;
- a 2x2 update uses two `cblas_dsyr` calls and one `cblas_dsyr2` call;
- 2x2 multipliers are computed in place, without allocating temporary vectors
  for every pivot;
- symmetric interchanges move only the lower-stored Schur complement and the
  already computed `L` rows instead of swapping a complete row and column;
- row-maximum searches read the symmetric value from the lower triangle.

These changes approximately halve the Schur-update storage traffic and remove
repeated heap allocations. `delayed_sytrf` remains the simple unblocked
reference implementation.

The separate blocked implementation adds:

- a 64-column panel that never splits a 2x2 pivot block;
- a workspace representation of the not-yet-written Schur complement;
- lazy `dgemv` materialization for each pivot column and candidate row;
- synchronized interchange of `A`, `perm`, and workspace rows when a node is
  pivoted or delayed;
- one lower-triangular BLAS-3 `dsyr2k` update at the end of each panel;
- Schur updates over delayed nodes as well as active candidates, so the delayed
  suffix remains a valid contribution block.

## Build

macOS uses the system Accelerate framework:

```sh
c++ -std=c++17 delayed_sytrf.cpp lapack_dsytrf.cpp symmetric_example.cpp \
  -framework Accelerate -o delayed_sytrf_example
./delayed_sytrf_example

c++ -std=c++17 delayed_sytrf.cpp lapack_dsytrf.cpp comparison_example.cpp \
  -framework Accelerate -o delayed_sytrf_comparison
./delayed_sytrf_comparison

c++ -O3 -DNDEBUG -std=c++17 delayed_sytrf.cpp delayed_sytrf_blocked.cpp \
  lapack_dsytrf.cpp blocked_comparison_example.cpp \
  -framework Accelerate -o delayed_sytrf_blocked_comparison
./delayed_sytrf_blocked_comparison 2048
```

Linux requires a CBLAS implementation, for example OpenBLAS:

```sh
sudo apt install cmake g++ libopenblas-dev
c++ -std=c++17 delayed_sytrf.cpp lapack_dsytrf.cpp symmetric_example.cpp \
  -llapack -lopenblas -o delayed_sytrf_example

c++ -std=c++17 delayed_sytrf.cpp lapack_dsytrf.cpp comparison_example.cpp \
  -llapack -lopenblas -o delayed_sytrf_comparison

c++ -O3 -DNDEBUG -std=c++17 delayed_sytrf.cpp delayed_sytrf_blocked.cpp \
  lapack_dsytrf.cpp blocked_comparison_example.cpp \
  -llapack -lopenblas -o delayed_sytrf_blocked_comparison
```

Or use CMake:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

`symmetric_example.cpp` validates consecutive delayed nodes, 1x1 and 2x2
pivots, a 2x2 Schur update, 100 deterministic random symmetric residual checks,
and a heap-backed 1200x1200 smoke test. All matrices and pivot metadata in the
example use `std::vector`; BLAS `dgemm` computes the residual products.

`comparison_example.cpp` applies the delayed algorithm and LAPACK `DSYTRF` to
the same heap-backed 4096x4096 rank-8 matrix. It reports factorization time,
delayed/singular pivot information, BLAS verification time, and maximum
absolute reconstruction error. The delayed algorithm stops after eight
pivots, whereas LAPACK reports the first singular block and continues.

`blocked_comparison_example.cpp` first runs 120 deterministic random residual
tests, a mixed 1x1/2x2 test crossing multiple panels, and a delayed-suffix test
whose zero nodes occur inside a panel. It then times the unblocked delayed
algorithm, the blocked delayed algorithm, and LAPACK `DSYTRF` on the same dense
mixed-pivot matrix. Its optional first command-line argument sets the benchmark
order; the default is 2048.

## Large matrices

Do not place a large dense matrix in a local `std::array` or C array. A single
1000x1000 `double` matrix occupies about 8 MB, which is already near the
default thread-stack limit on many systems. Allocate matrix and metadata on the
heap instead:

```cpp
const std::size_t elements =
    static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);
std::vector<double> A(elements, 0.0);
std::vector<int> perm(static_cast<std::size_t>(n));
std::vector<int> piv_size(static_cast<std::size_t>(n));

delayed_sytrf(n, A.data(), lda, perm.data(), piv_size.data(), info);
```

Converting to `std::size_t` before multiplying also avoids signed `int`
overflow when calculating the allocation size.
