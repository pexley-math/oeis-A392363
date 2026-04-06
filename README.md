# OEIS A392363 -- Smallest Polyiamond Containing All Free n-Iamonds

Solver code, data, and figures for [OEIS A392363](https://oeis.org/A392363).

## The Problem

a(n) = the minimum number of triangular cells in a connected polyiamond such that every free n-iamond can be placed entirely within it under a rigid motion of the triangular lattice (a D_6 orientation plus a parity-preserving lattice translation). This is the triangular-grid analog of [A327094](https://oeis.org/A327094) (square grid, Dawson's Minimum Common Superform problem, 1942). The input to the problem -- the number of free n-iamonds -- is [A000577](https://oeis.org/A000577).

## Results

**New proved terms (this work):**

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **a(n)** | 1 | 2 | 3 | 5 | 7 | 10 | 13 | 16 | 19 | 24 | 27 | 32 | 37 | 42 |
| **Pieces** | 1 | 1 | 1 | 3 | 4 | 12 | 24 | 66 | 160 | 448 | 1186 | 3334 | 9235 | 26166 |

Each value proved by matching SAT/UNSAT certificates (SAT at a(n), UNSAT at a(n)-1) on an unconstrained solver. Every container verified by two independent geometric verifiers with disjoint code paths, each checking both containment and local optimality.

## Conjecture

lim a(n)/n^2 = 1/5.

The empirical upper bound a(n) <= ceil((n^2 + 2n)/5) holds on all 14 computed terms, tight at n <= 8 and n = 10. The asymptotic ratio a(n)/n^2 decreases monotonically from 0.240 (n=10) to 0.214 (n=14).

## Method

Unconstrained Boolean satisfiability (SAT) solver (Glucose 4.2 via PySAT) with counterexample-guided abstraction refinement (CEGAR) for connectivity. Top-down search with binary search + linear descent proves optimality: SAT at k cells confirms a solution exists, UNSAT at k-1 proves no smaller solution is possible.

- **Tight grid heuristic:** R(n) = max(4, floor((n+2)/3)) rows by n columns. Gives >100x speedup over naive n x n grid.
- **Pre-solve:** Mandatory and impossible cells identified before SAT search, added as free unit clauses.
- **No shape-constraint heuristics:** Row-contiguity constraints were found to exclude true optima at n=11 and n=14 (optimal containers have non-contiguous bottom rows). All reported values use the unconstrained solver.

## Running the Solver

**Requirements:** Python 3.8+, python-sat

```bash
pip install python-sat

# Run all terms (n=1..14)
python code/solve_polyiamond_container.py --n 1-14

# Run specific term
python code/solve_polyiamond_container.py --n 12
```

**Verify all proofs** (two independent verifiers, no shared code paths):

```bash
python code/verify_independent.py 14    # ~63s, D_6 reimplementation
python code/verify_geometric.py 14      # ~28s, geometric containment
```

## Files

| File | Description |
|------|-------------|
| `code/solve_polyiamond_container.py` | Unconstrained SAT+CEGAR solver |
| `code/verify_independent.py` | Independent verifier #1 (containment + local optimality) |
| `code/verify_geometric.py` | Independent verifier #2 (containment + local optimality) |
| `code/generate-figures.py` | Publication figure generator |
| `research/solver-results.json` | Machine-readable results with solutions |
| `research/solver-run-log.txt` | Reviewer-grade proof of solver run |
| `submission/polyiamond-container-figures.pdf` | Publication figures (all 14 containers) |

## Prior Art and Acknowledgments

This is a new sequence -- no prior OEIS entry exists for this problem. Prior art search: OEIS (8 queries), arXiv, Google Scholar, GitHub, and general web sources. No match found. Closest known sequence [A033638](https://oeis.org/A033638) (quarter-squares + 1) diverges at a(8).

The problem generalises T. R. Dawson's 1942 Minimum Common Superform question for pentominoes (*Fairy Chess Review* Vol. 5 No. 4) to the triangular grid.

This work was inspired by the [OEIS](https://oeis.org/) and the community of contributors who maintain it.

## Hardware

AMD Ryzen 5 5600 (6-core / 12-thread), 16 GB RAM.

## License

[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) -- Peter Exley, 2026.

This work is freely available. If you find it useful, a citation or acknowledgment is appreciated but not required.

## Links

- **A327094** (square-grid analog): https://oeis.org/A327094
- **A000577** (free polyiamond count): https://oeis.org/A000577
- **A394840** (polyiamond hole, companion sequence): https://oeis.org/A394840
- T. R. Dawson, *Fairy Chess Review* Vol. 5 No. 4, 1942 -- original MCS problem. Archive: [The Problemist](https://www.theproblemist.org/mags.pl?type=fcr&page=volumes) (Vol. 5 covers 1942-1945).
- Puzzle Zapper, [Polyomino Common Superforms](https://puzzlezapper.com/aom/mathrec/polycover.html) -- secondary source on Dawson's 1942 problem.
