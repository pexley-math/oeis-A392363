#!/usr/bin/env python3
"""Geometric verifier for polyiamond container solutions.

This verifier restricts placement checks to RIGID MOTIONS ONLY (D_6 point
group combined with parity-preserving integer translations). It reproduces
paper's free-polyiamond enumeration exactly, but in fits_in it allows only
(dr, dc) translations with (dr + dc) even -- the standard p6m translation
subgroup.

Rationale: in the standard crystallographic convention for triangular
tilings (p6m symmetry group), translations preserve cell orientation. The
paper's original fits_in allows arbitrary integer (dr, dc), including
odd-parity translations that flip cell orientations on a per-cell basis;
such operations are NOT rigid motions. They may produce false-positive
matches where the placed cell-set is NOT congruent to the original piece.

See references in the research brief: Lunnon 1972 (cube coordinates),
Polyform Puzzler ((x, y, z) with z = orientation bit), BorisTheBrave
(a, b, c with sum encoding orientation) -- all these standard conventions
encode orientation separately from position, so that integer translations
are automatically orientation-preserving.

Usage:
    python code/verify_geometric.py          # check n=1..14
    python code/verify_geometric.py 8        # check n=1..8
"""

import json
import sys
import time

A000577 = {1: 1, 2: 1, 3: 1, 4: 3, 5: 4, 6: 12, 7: 24, 8: 66,
           9: 160, 10: 448, 11: 1186, 12: 3334, 13: 9235, 14: 26166}


# ---------- Paper's D_6 (for free-polyiamond enumeration) ----------
# We reuse paper's R_60 and F formulas here because the PURPOSE of this
# verifier is to check paper's CONTAINMENT logic, not paper's D_6. The
# D_6 is used identically -- what changes is the translation set.

def R60(cell):
    r, c = cell
    t = (r + c) % 2
    return ((r + c + 1) // 2, (c - 3 * r - t - 2) // 2)


def F(cell):
    r, c = cell
    return (-r - 1, c)


def transform(shape, k):
    out = set(shape)
    if k >= 6:
        out = {F(x) for x in out}
        k -= 6
    for _ in range(k):
        out = {R60(x) for x in out}
    return frozenset(out)


def normalise(shape):
    if not shape:
        return tuple()
    rmin = min(r for r, _ in shape)
    cmin = min(c for _, c in shape)
    shift_c = rmin + 2 * ((cmin - rmin) // 2)
    return tuple(sorted((r - rmin, c - shift_c) for r, c in shape))


def canonical(shape):
    best = None
    for k in range(12):
        t = normalise(transform(shape, k))
        if best is None or t < best:
            best = t
    return best


# ---------- Edge adjacency (paper's, unchanged) ----------

def neighbours(cell):
    r, c = cell
    d = (r + c) % 2
    if d == 0:
        return [(r, c - 1), (r, c + 1), (r - 1, c)]
    return [(r, c - 1), (r, c + 1), (r + 1, c)]


# ---------- Free polyiamond enumeration ----------

def enumerate_free(n, _cache={}):
    if n in _cache:
        return _cache[n]
    if n == 1:
        result = {canonical(frozenset({(0, 0)}))}
    else:
        prev = enumerate_free(n - 1)
        result = set()
        for cf in prev:
            cells = set(cf)
            for cell in list(cells):
                for nb in neighbours(cell):
                    if nb not in cells:
                        result.add(canonical(frozenset(cells | {nb})))
    _cache[n] = result
    return result


# ---------- GEOMETRIC containment check ----------
# Key difference from paper: only EVEN-parity translations allowed.

def fits_in_geometric(piece_canonical, container_set):
    """Return True iff piece fits in container via some rigid motion.

    Rigid motion = D_6 point-group element + parity-preserving translation
    ((dr + dc) even). This is the standard crystallographic definition.
    """
    piece = set(piece_canonical)
    Rmin = min(r for r, _ in container_set)
    Rmax = max(r for r, _ in container_set)
    Cmin = min(c for _, c in container_set)
    Cmax = max(c for _, c in container_set)
    for k in range(12):
        orient = transform(piece, k)
        rmin_o = min(r for r, _ in orient)
        rmax_o = max(r for r, _ in orient)
        cmin_o = min(c for _, c in orient)
        cmax_o = max(c for _, c in orient)
        for dr in range(Rmin - rmin_o, Rmax - rmax_o + 1):
            for dc in range(Cmin - cmin_o, Cmax - cmax_o + 1):
                if (dr + dc) % 2 != 0:
                    continue  # skip parity-flipping "translations"
                translated = frozenset((r + dr, c + dc) for r, c in orient)
                if translated <= container_set:
                    return True
    return False


# ---------- Local optimality (cell-removal) check ----------

def is_connected(cells):
    """BFS connectivity test on triangular edge-adjacency graph."""
    if not cells:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in neighbours(cur):
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def check_local_optimality(container_set, pieces):
    """Return list of cells whose removal still yields a valid universal
    container (connected and contains all pieces). Empty list = locally
    optimal: no single-cell removal gives a smaller valid container.
    """
    removable = []
    container_set = set(container_set)
    for cell in sorted(container_set):
        test_set = container_set - {cell}
        if len(test_set) < 1:
            continue
        if not is_connected(test_set):
            continue
        test_frozen = frozenset(test_set)
        all_fit = True
        for piece in pieces:
            if not fits_in_geometric(piece, test_frozen):
                all_fit = False
                break
        if all_fit:
            removable.append(cell)
    return removable


# ---------- Self-checks ----------

def verify_orbit_counts(max_n=8):
    for n in range(1, max_n + 1):
        count = len(enumerate_free(n))
        expected = A000577[n]
        assert count == expected, f"n={n}: got {count}, expected A000577={expected}"
    print(f"A000577 cross-check passed for n=1..{max_n}")


# ---------- Main ----------

def main(max_n=14):
    verify_orbit_counts(max_n=6)

    with open('research-outputs/paper-project/oeis-new-polyiamond-container/research/solver-results.json') as f:
        data = json.load(f)

    print()
    print("GEOMETRIC verifier: containment + local optimality")
    print("=" * 82)
    print(f"{'n':>3} {'|C|':>5} {'|pieces|':>9} {'contain':<10} "
          f"{'removable':>10} {'opt?':<5} {'verify(s)':>10} {'opt(s)':>8}")
    print("-" * 82)

    all_ok = True
    all_opt = True
    fail_summary = []
    opt_fail_summary = []
    for n_str in sorted(data['solutions'].keys(), key=int):
        n = int(n_str)
        if n > max_n:
            break
        container = frozenset(tuple(c) for c in data['solutions'][n_str]['cells'])
        expected = A000577[n]

        pieces = enumerate_free(n)

        # Containment check
        t0 = time.time()
        missing_count = 0
        missing_pieces = []
        for piece in pieces:
            if not fits_in_geometric(piece, container):
                missing_count += 1
                if len(missing_pieces) < 3:
                    missing_pieces.append(piece)
        t_verify = time.time() - t0
        contain_status = "OK" if missing_count == 0 else f"MISS({missing_count})"

        # Local optimality (only when containment passes)
        if missing_count == 0:
            t0 = time.time()
            removable = check_local_optimality(container, pieces)
            t_opt = time.time() - t0
            opt_status = "OK" if not removable else "FAIL"
            removable_str = f"{len(removable)}" if removable else "none"
            if removable:
                all_opt = False
                opt_fail_summary.append((n, len(container), removable[:3]))
        else:
            removable_str = "--"
            opt_status = "--"
            t_opt = 0.0
            all_opt = False

        print(f"{n:>3} {len(container):>5} {len(pieces):>9} {contain_status:<10} "
              f"{removable_str:>10} {opt_status:<5} {t_verify:>10.2f} {t_opt:>8.2f}")
        if missing_count > 0:
            all_ok = False
            fail_summary.append((n, missing_count, len(pieces), missing_pieces))

    print("=" * 82)
    if all_ok and all_opt:
        print("All containers: GEOMETRIC containment OK AND locally optimal")
    else:
        if not all_ok:
            print(f"CONTAINMENT FAILURES at n = {[f[0] for f in fail_summary]}")
            for n, miss, total, examples in fail_summary:
                print(f"  n={n}: {miss}/{total} free {n}-iamonds not contained")
                for p in examples[:2]:
                    print(f"    example: {p}")
        if not all_opt:
            print(f"LOCAL-OPTIMALITY FAILURES at n = {[f[0] for f in opt_fail_summary]}")
            for n, cont_size, removable in opt_fail_summary:
                print(f"  n={n}: {len(removable)}+ cell(s) can be removed from "
                      f"{cont_size}-cell container; sample cells: {removable}")
    return 0 if (all_ok and all_opt) else 1


if __name__ == '__main__':
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    sys.exit(main(max_n=max_n))
