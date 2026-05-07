#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Dict
from xml.sax.saxutils import escape

_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    _HAS_MPL = False


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#000000", sw: float = 1.0) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" />'


def _svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", rotate: float = 0.0) -> str:
    if rotate:
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
            f'transform="rotate({rotate:.1f},{x:.2f},{y:.2f})">{escape(text)}</text>'
        )
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}">{escape(text)}</text>'


def save_svg_fallback(
    out_svg: Path,
    bench: List[str],
    left_vals: List[float],
    left_title: str,
    selection_err: List[float],
    generation_fail: List[float],
) -> None:
    width, height = 1600, 680
    panel_w = 700
    panel_h = 420
    top = 80
    left_x = 70
    right_x = 830
    y0 = top + panel_h

    n = len(bench)
    bar_w = 52
    gap = (panel_w - n * bar_w) / (n + 1)
    max_left = max(1.0, max(left_vals) * 1.15)
    max_right = 100.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 36, "Diagnosing the Selection Bottleneck", size=22),
        _svg_text(left_x + panel_w / 2, 62, "(a) Oracle--TTAug gap", size=15),
        _svg_text(right_x + panel_w / 2, 62, "(b) TTAug error decomposition", size=15),
    ]

    # Axes boxes
    parts.append(_svg_rect(left_x, top, panel_w, panel_h, fill="none"))
    parts.append(_svg_rect(right_x, top, panel_w, panel_h, fill="none"))
    parts.append(_svg_text(left_x - 45, top + panel_h / 2, "Gap (points)", size=12, rotate=-90))
    parts.append(_svg_text(right_x - 45, top + panel_h / 2, "Error (points)", size=12, rotate=-90))

    # Left panel bars
    for i, v in enumerate(left_vals):
        x = left_x + gap + i * (bar_w + gap)
        h = (v / max_left) * panel_h
        y = y0 - h
        parts.append(_svg_rect(x, y, bar_w, h, fill="#3a86ff", stroke="#111111", sw=1.0))
        parts.append(_svg_text(x + bar_w / 2, y - 6, f"{v:.1f}", size=11))
        parts.append(_svg_text(x + bar_w / 2, y0 + 40, bench[i], size=11, rotate=-30))

    # Right panel stacked bars
    for i, (s, g) in enumerate(zip(selection_err, generation_fail)):
        x = right_x + gap + i * (bar_w + gap)
        hs = (s / max_right) * panel_h
        hg = (g / max_right) * panel_h
        ys = y0 - hs
        yg = ys - hg
        parts.append(_svg_rect(x, ys, bar_w, hs, fill="#ff6b6b", stroke="#111111", sw=1.0))
        parts.append(_svg_rect(x, yg, bar_w, hg, fill="#ffd166", stroke="#111111", sw=1.0))
        parts.append(_svg_text(x + bar_w / 2, y0 + 40, bench[i], size=11, rotate=-30))

    # Legend
    lx = right_x + 20
    ly = top + 20
    parts.append(_svg_rect(lx, ly, 18, 12, fill="#ff6b6b", stroke="#111111"))
    parts.append(_svg_text(lx + 24, ly + 11, "Selection error (Oracle - TTAug)", size=11, anchor="start"))
    parts.append(_svg_rect(lx, ly + 22, 18, 12, fill="#ffd166", stroke="#111111"))
    parts.append(_svg_text(lx + 24, ly + 33, "Generation failure (100 - Oracle)", size=11, anchor="start"))
    parts.append(_svg_text(
        width / 2,
        635,
        "Caption: Red = selection error (Oracle - TTAug), yellow = generation failure (100 - Oracle). "
        "Their sum equals total TTAug error (100 - TTAug).",
        size=12,
    ))

    parts.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot combined left-right figure: selection-gap bar + error-decomposition stacked bar."
    )
    ap.add_argument(
        "--input-csv",
        default="paper_neurips2026_artifacts/tables/oracle_selection_decomp_recalc_v91nocf_regenA.csv",
        help="CSV with columns from oracle_selection_decomp_recalc_v91nocf_regenA.csv",
    )
    ap.add_argument(
        "--output-prefix",
        default="paper_neurips2026_artifacts/figures/oracle_left_right_combined_v91nocf_regenA",
        help="Output file prefix (without extension).",
    )
    ap.add_argument(
        "--left-gap-mode",
        choices=["ttaug", "as"],
        default="ttaug",
        help="Left-panel gap definition: oracle-ttaug or oracle-as_tta.",
    )
    ap.add_argument(
        "--include-coco",
        action="store_true",
        help="Include COCO in plots. By default, COCO is excluded.",
    )
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    rows = load_rows(in_path)
    if not rows:
        raise RuntimeError(f"No data rows found in: {in_path}")

    if not args.include_coco:
        rows = [r for r in rows if str(r.get("benchmark", "")).upper() != "COCO"]
    if not rows:
        raise RuntimeError("No rows left after filtering (check --include-coco).")

    bench = [r["benchmark"] for r in rows]
    oracle = [to_float(r["oracle"]) for r in rows]
    ttaug = [to_float(r["ttaug"]) for r in rows]
    astta = [to_float(r["as_tta"]) for r in rows]

    if args.left_gap_mode == "ttaug":
        left_vals = [o - t for o, t in zip(oracle, ttaug)]
        left_title = "(a) Oracle--TTAug gap"
    else:
        left_vals = [o - a for o, a in zip(oracle, astta)]
        left_title = "(a) Oracle--AS-TTA gap"

    selection_err = [o - t for o, t in zip(oracle, ttaug)]
    generation_fail = [100.0 - o for o in oracle]
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not _HAS_MPL:
        out_svg = out_prefix.with_suffix(".svg")
        save_svg_fallback(out_svg, bench, left_vals, left_title, selection_err, generation_fail)
        print(out_svg)
        return

    x = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=180)

    # Left panel
    ax = axes[0]
    bars = ax.bar(x, left_vals, color="#3a86ff", edgecolor="black", linewidth=0.6)
    ax.set_title(left_title)
    ax.set_ylabel("Gap (points)")
    ax.set_xticks(x)
    ax.set_xticklabels(bench, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars, left_vals):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.6,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Right panel
    ax = axes[1]
    ax.bar(x, selection_err, color="#ff6b6b", edgecolor="black", linewidth=0.6, label="Selection error (Oracle - TTAug)")
    ax.bar(
        x,
        generation_fail,
        bottom=selection_err,
        color="#ffd166",
        edgecolor="black",
        linewidth=0.6,
        label="Generation failure (100 - Oracle)",
    )
    ax.set_title("(b) TTAug error decomposition")
    ax.set_ylabel("Error (points)")
    ax.set_xticks(x)
    ax.set_xticklabels(bench, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Diagnosing the Selection Bottleneck", y=1.04, fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Caption: Red = selection error (Oracle - TTAug), yellow = generation failure (100 - Oracle). "
        "Their sum equals total TTAug error (100 - TTAug).",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    print(out_prefix.with_suffix(".png"))
    print(out_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
