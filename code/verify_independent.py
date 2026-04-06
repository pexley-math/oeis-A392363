#!/usr/bin/env python3
"""Independent verifier for polyiamond container solutions.

This verifier is a FROM-SCRATCH reimplementation that shares no code with
solve_polyiamond_container.py:

1. D_6 generators R_60 and F are re-derived from the paper's Section 2.2
   formulas as fresh Python functions. A typo or implementation bug in one
   of the two code paths would not replicate in the other.
2. Free n-iamonds are enumerated iteratively from scratch via BFS
   ("add an edge-adjacent cell to any shape in the previous generation").
3. Canonical forms are computed by generating the full D_6 orbit of each
   candidate shape, normalising each orbit element (translating so the
   min cell is at (0, 0)), and taking the lexicographic minimum of the
   sorted-tuple representations.
4. For each reported container, containment of every free n-iamond is
   checked by independently iterating every D_6 orientation and every
   integer (dr, dc) translation, with a plain set-subset test.

Correctness is independently verified by:
  * Piece-count check: |free(n)| must equal A000577(n).
  * Group-order check: R_60^6 = I, F^2 = I, (F o R_60)^2 = I (dihedral relation).
  * Round-trip check: canonical(canonical(P)) = canonical(P).
"""

import json
import sys
import time

A000577 = {1: 1, 2: 1, 3: 1, 4: 3, 5: 4, 6: 12, 7: 24, 8: 66,
           9: 160, 10: 448, 11: 1186, 12: 3334, 13: 9235, 14: 26166}


# ---------- D_6 generators (independent reimplementation) ----------

def R60(cell):
    """60-degree clockwise generator (fresh reimplementation of paper's formula)."""
    r, c = cell
    t = (r + c) % 2
    # In Python, // is floor division.
    new_r = (r + c + 1) // 2
    new_c = (c - 3 * r - t - 2) // 2
    return (new_r, new_c)


def F(cell):
    """Reflection (fresh reimplementation of paper's formula)."""
    r, c = cell
    return (-r - 1, c)


def apply_D6(cell, element_index):
    """Apply D_6 element (0..11) to a cell.

    Elements 0..5: R_60^k for k = 0..5.
    Elements 6..11: F composed with R_60^k for k = 0..5.
    """
    result = cell
    if element_index >= 6:
        result = F(result)
        k = element_index - 6
    else:
        k = element_index
    for _ in range(k):
        result = R60(result)
    return result


def transform(shape, element_index):
    return frozenset(apply_D6(cell, element_index) for cell in shape)


def normalise(shape):
    """Translate shape so its min cell is at (0, 0) or (0, 1), using a
    translation with (dr + dc) even (parity-preserving). Returns sorted
    tuple. This is a fresh reimplementation of Section 2.2's shift_c formula.
    """
    if not shape:
        return tuple()
    rmin = min(r for r, _ in shape)
    cmin = min(c for _, c in shape)
    # shift_c = rmin + 2*floor((cmin - rmin) / 2), per Section 2.2.
    shift_c = rmin + 2 * ((cmin - rmin) // 2)
    return tuple(sorted((r - rmin, c - shift_c) for r, c in shape))


def canonical(shape):
    best = None
    for i in range(12):
        t = transform(shape, i)
        tup = normalise(t)
        if best is None or tup < best:
            best = tup
    return best


# ---------- Polyiamond enumeration ----------

def neighbours(cell):
    r, c = cell
    d = (r + c) % 2
    if d == 0:  # up-pointing cell
        return [(r, c - 1), (r, c + 1), (r - 1, c)]
    return [(r, c - 1), (r, c + 1), (r + 1, c)]


def enumerate_free(n):
    """Return set of canonical tuples of all free n-iamonds."""
    if n == 1:
        return {canonical(frozenset({(0, 0)}))}
    prev = enumerate_free(n - 1)
    result = set()
    for cf in prev:
        cells = set(cf)
        for cell in list(cells):
            for nb in neighbours(cell):
                if nb not in cells:
                    new_shape = frozenset(cells | {nb})
                    result.add(canonical(new_shape))
    return result


# ---------- Containment check ----------

def fits_in(piece_canonical, container_set):
    piece = set(piece_canonical)
    rmin_c = min(r for r, _ in container_set)
    rmax_c = max(r for r, _ in container_set)
    cmin_c = min(c for _, c in container_set)
    cmax_c = max(c for _, c in container_set)
    for i in range(12):
        orient = transform(piece, i)
        rmin_o = min(r for r, _ in orient)
        rmax_o = max(r for r, _ in orient)
        cmin_o = min(c for _, c in orient)
        cmax_o = max(c for _, c in orient)
        for dr in range(rmin_c - rmin_o, rmax_c - rmax_o + 1):
            for dc in range(cmin_c - cmin_o, cmax_c - cmax_o + 1):
                # Triangular grid: parity-preserving translations only
                # (odd dr+dc flips cell orientation, not a rigid motion)
                if (dr + dc) % 2 != 0:
                    continue
                translated = frozenset((r + dr, c + dc) for r, c in orient)
                if translated <= container_set:
                    return True
    return False


# ---------- Local optimality (cell-removal) check ----------

def is_connected(cells):
    """BFS connectivity test on the triangular edge-adjacency graph."""
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
    """Check whether any single cell can be removed while keeping the
    container valid (connected + contains all pieces).

    Returns a list of removable cells; empty list = locally optimal
    (no single-cell removal yields a smaller valid universal container).
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
        # Short-circuit: as soon as one piece fails, this cell is NOT removable
        all_fit = True
        for piece in pieces:
            if not fits_in(piece, test_frozen):
                all_fit = False
                break
        if all_fit:
            removable.append(cell)
    return removable


# ---------- Self-check of D_6 group ----------

def verify_group_relations():
    """Check R_60^6 = I, F^2 = I, (F R_60)^2 = I on sample cells."""
    samples = [(0, 0), (1, 2), (-3, 5), (7, -4), (0, 13)]
    for cell in samples:
        # R_60^6 = I
        x = cell
        for _ in range(6):
            x = R60(x)
        assert x == cell, f"R_60^6({cell}) = {x}, expected {cell}"
        # F^2 = I
        assert F(F(cell)) == cell, f"F^2({cell}) != {cell}"
        # (F . R_60)^2 = I   (dihedral relation: F R_60 F = R_60^{-1},
        # equivalently (F R_60)^2 = I)
        x = cell
        x = R60(x)
        x = F(x)
        x = R60(x)
        x = F(x)
        assert x == cell, f"(F R_60)^2({cell}) = {x}"
    print("D_6 group relations verified: R_60^6 = I, F^2 = I, (F R_60)^2 = I")


def verify_canonical_idempotent(n=4):
    """Check canonical(canonical_tuple_as_set) == canonical_tuple."""
    pieces = enumerate_free(n)
    for cf in pieces:
        cf2 = canonical(frozenset(cf))
        assert cf2 == cf, f"canonical idempotence failed: {cf} -> {cf2}"
    print(f"Canonical-form idempotence verified (n={n}, {len(pieces)} pieces)")


# ---------- Main ----------

def main(max_n=14):
    verify_group_relations()
    verify_canonical_idempotent(n=4)

    with open('research-outputs/paper-project/oeis-new-polyiamond-container/research/solver-results.json') as f:
        data = json.load(f)

    print()
    print("Independent verifier: containment + local optimality")
    print("=" * 82)
    print(f"{'n':>3} {'|C|':>5} {'A000577':>8} {'contain':<9} "
          f"{'removable':>10} {'opt?':<6} {'verify (s)':>11} {'optchk (s)':>11}")
    print("-" * 82)

    all_ok = True
    all_opt = True
    for n_str in sorted(data['solutions'].keys(), key=int):
        n = int(n_str)
        if n > max_n:
            break
        cells = frozenset(tuple(c) for c in data['solutions'][n_str]['cells'])
        expected = A000577[n]

        pieces = enumerate_free(n)
        if len(pieces) != expected:
            print(f"{n:>3} {len(cells):>5} {expected:>8} COUNTFAIL -- got {len(pieces)}")
            all_ok = False
            continue

        # Containment check
        t0 = time.time()
        missing = 0
        for piece in pieces:
            if not fits_in(piece, cells):
                missing += 1
        t_verify = time.time() - t0
        contain_status = "OK" if missing == 0 else f"FAIL({missing})"

        # Local optimality check (only if containment passes)
        if missing == 0:
            t0 = time.time()
            removable = check_local_optimality(cells, pieces)
            t_opt = time.time() - t0
            opt_status = "OK" if not removable else "FAIL"
            removable_str = f"{len(removable)}" if removable else "none"
            if removable:
                all_opt = False
        else:
            removable_str = "--"
            opt_status = "--"
            t_opt = 0.0
            all_opt = False

        print(f"{n:>3} {len(cells):>5} {expected:>8} {contain_status:<9} "
              f"{removable_str:>10} {opt_status:<6} {t_verify:>11.2f} {t_opt:>11.2f}")
        if missing != 0:
            all_ok = False

    print("=" * 82)
    if all_ok and all_opt:
        print("All containers: containment OK and locally optimal")
    else:
        if not all_ok:
            print("CONTAINMENT FAILURES DETECTED")
        if not all_opt:
            print("LOCAL-OPTIMALITY FAILURES DETECTED (container can be shrunk)")
    return 0 if (all_ok and all_opt) else 1


if __name__ == '__main__':
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    sys.exit(main(max_n=max_n))
