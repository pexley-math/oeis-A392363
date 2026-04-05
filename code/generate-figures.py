"""
Generate publication figures for the polyiamond container sequence.
Reads solver-results.json and produces a Typst PDF via DocumentBuilder.

Single-state binary rendering: every cell in the container is filled with
a uniform color. This is a "smallest covering shape" problem -- no
sub-structure to distinguish.

Usage:
    python generate-figures.py
"""

import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from figure_gen_utils.document_builder import DocumentBuilder

PROJ_DIR = Path(__file__).resolve().parent.parent

CONTAINER_COLOR = "#1ABC9C"  # teal


def main():
    results_path = PROJ_DIR / "research" / "solver-results.json"
    results = json.loads(results_path.read_text())

    terms = results["terms"]
    solutions = results["solutions"]
    details = results.get("details", {})

    # Order by n
    ns = sorted(int(k) for k in terms)
    seq_str = ", ".join(str(terms[str(n)]) for n in ns)

    doc = DocumentBuilder(
        title="Smallest Connected Polyiamond Containing All Free n-Iamonds",
        description=("a(n) = minimum triangular cells in a connected polyiamond "
                     "that contains every free n-iamond as a sub-pattern "
                     "(under translation, rotation, reflection)."),
        sequence_line=f"a(1..{len(terms)}) = {seq_str}",
    )

    for n in ns:
        n_str = str(n)
        a_n = terms[n_str]
        cells_raw = solutions[n_str]["cells"]
        cells = [tuple(c) for c in cells_raw]

        det = details.get(n_str, {})
        bbox = det.get("grid_size", [0, 0])
        pieces = det.get("num_free_polyiamonds", "?")
        elapsed = det.get("elapsed", "?")
        detail_text = (f"{a_n} cells, {pieces} free pieces, "
                       f"bbox {bbox[0]}x{bbox[1]}, solved in {elapsed}s")

        doc.add_triangle_figure(
            cells=cells,
            n=n,
            k=a_n,
            status="PROVED",
            method="SAT + CEGAR",
            fill_color=CONTAINER_COLOR,
            detail_text=detail_text,
        )

    output_typ = PROJ_DIR / "submission" / "polyiamond-container-figures.typ"
    output_typ.parent.mkdir(parents=True, exist_ok=True)
    doc.generate(str(output_typ))
    print(f"Generated: {output_typ}")

    pdf_path = PROJ_DIR / "submission" / "polyiamond-container-figures.pdf"
    try:
        doc.compile(pdf_path=str(pdf_path))
        print(f"Compiled: {pdf_path}")
    except Exception as e:
        print(f"Typst compile failed: {e}")
        print("  (typ source saved; compile manually)")


if __name__ == "__main__":
    main()
