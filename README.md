# OEIS (pending A-number) -- Smallest Polyiamond Containing All Free n-Iamonds

Solver code, data, and figures for the smallest connected polyiamond that contains every free polyiamond of size n as a sub-pattern.

## The Problem

a(n) = the minimum number of cells in a connected polyiamond (a figure made of equilateral triangles sharing edges) such that every free n-iamond can be placed entirely within it, under translation, rotation, and reflection. This is the triangular grid analog of OEIS A327094 (square grid, Dawson's Minimum Common Superform problem for pentominoes, 1942).

## Results

**New proved terms (this work):**

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **a(n)** | 1 | 2 | 3 | 5 | 6 | 9 | 12 | 16 | 19 | 23 | 27 | 31 | 36 | 41 |
| **Pieces** | 1 | 1 | 1 | 3 | 4 | 12 | 24 | 66 | 160 | 448 | 1186 | 3334 | 9235 | 26166 |

All 14 values proved exact by SAT (Glucose 4.2 via PySAT + CEGAR). Every reported a(n) was independently re-verified by brute-force enumeration (every free n-iamond tested against every orientation and offset inside the reported container). Terms n=10..14 were additionally cross-validated with a larger grid (rows+1): same answers.

## Grid Family Comparison

| n | Triangle (this) | Square (A327094) | Hex (ours, draft) |
|:---:|:---:|:---:|:---:|
| 4 | 5 | 6 | 7 |
| 5 | 6 | 9 | 10 |
| 6 | 9 | 12 | 14 |
| 7 | 12 | 17 | 18 |
| 8 | 16 | 20 | 24 |

Triangle < Square < Hex consistently. Triangular cells have 3 edge-neighbors (vs 4 square, 6 hex), making polyiamond pieces more linear and easier to pack.

## Method

**SAT solver** (Glucose 4.2 via PySAT) with CEGAR connectivity cuts and top-down search:
1. For each n, enumerate all free n-iamonds (verified against A000577).
2. Encode as SAT: cell-occupancy variables + piece-placement auxiliaries (at least one placement per piece; each placement implies its cells are occupied).
3. Enforce cardinality (exactly k cells, totalizer encoding) and shape constraints (contiguous columns per row + at least one full row of width n, for n >= 6).
4. Connectivity enforced via CEGAR: solve, find components, cut disconnected solutions.
5. Binary search + linear descent: start from upper bound, descend until UNSAT at k-1 proves optimality.

**Key optimization:** Tight rectangular grid `rows = max(4, (n+2)//3)` by n columns, derived from observed solution bounding boxes. This single change gave a 457x speedup at n=11 (7s vs 55 min) versus the initial n x n grid.

## Running the Solver

**Requirements:** Python 3.8+, `python-sat`

```bash
pip install python-sat

# Run all terms 1-14
python code/solve_polyiamond_container.py

# Run a single n
python code/solve_polyiamond_container.py --n 12

# Run a range
python code/solve_polyiamond_container.py --n 10-14
```

Output: `research/solver-results.json` (proved terms + cells), `research/solver-run-log.txt` (full run log).

## Files

- `code/solve_polyiamond_container.py` -- SAT solver (Glucose 4.2 via PySAT)
- `code/generate-figures.py` -- publication figure generator (Typst)
- `research/solver-results.json` -- 14 proved terms + optimal cell configurations
- `research/solver-run-log.txt` -- full solver run log
- `research/conjecture-report.md` -- pattern analysis, disproven conjectures, asymptotic observation
- `research/polyiamond-container-understanding.pdf` -- personal understanding figure
- `submission/polyiamond-container-figures.pdf` -- publication figures (one per proved term)
- `submission/oeis-draft.txt` -- OEIS submission text
- `submission/oeis-copy-helper.html` -- click-to-copy helper for OEIS web form

## Prior Art and Acknowledgments

T. R. Dawson introduced the Minimum Common Superform (MCS) problem for pentominoes in Fairy Chess Review Vol. 5 No. 4 (1942). Our work extends that concept to the triangular grid. Related existing OEIS sequences: A327094 (square MCS), A000577 (free polyiamond count). No triangular-grid MCS sequence was found in OEIS, arXiv, Google Scholar, GitHub, or general web sources during prior-art search.

## License

CC-BY-4.0. See LICENSE file.

## Author

Peter Exley, 2026.
