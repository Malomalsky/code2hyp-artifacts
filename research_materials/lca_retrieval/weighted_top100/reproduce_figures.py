"""Regenerate the three vector diagrams, without the manuscript or its captions."""

import json
from pathlib import Path
import re
from figure_kernels import problem_figure, cost_figure

ROOT = Path(__file__).resolve().parent
PREAMBLE = r"""\documentclass[border=3mm]{standalone}
\usepackage{fontspec}
\setmainfont{texgyretermes-regular.otf}[
  BoldFont=texgyretermes-bold.otf,
  ItalicFont=texgyretermes-italic.otf,
  BoldItalicFont=texgyretermes-bolditalic.otf]
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{figblue}{RGB}{0,114,178}
\definecolor{figorange}{RGB}{190,80,0}
\begin{document}
"""


def read(name):
    return json.loads((ROOT / name).read_text())


def generate():
    output = ROOT / "reproduced"
    output.mkdir(exist_ok=True)
    top = read("data/top100/result.json")
    cost = read("data/two_stage/result.json")
    early = read("data/earlier/reference_results.json")
    fresh = read("data/earlier/fresh_fusion_result.json")
    old_plot = (ROOT / "figures/reference/figure-3.tikz").read_text()
    methods = ("fusion_retained", "fusion_masked", "ast_nodes", "fragment_retained", "fragment_masked")
    maps = [early["rank_fusion"]["previous650"]["methods"][k]["MAP_at_8"] for k in methods]
    maps += [early["Table_8"]["methods"]["ast_transitions"]["MAP_at_8"]]
    maps += [fresh["methods"][k]["metrics"]["problem_macro_map_at_r"] for k in methods]
    blocks = re.findall(r"\\foreach \\score/\\row in \{(.*?)\}\{", old_plot, re.S)
    observed = [float(v) for b in blocks for v in re.findall(r"^\s*(0\.\d+)/", b, re.M)]
    assert observed == maps
    contrasts = [early["Table_8"]["contrasts"][k] for k in (
        "retained_minus_fragment_masked", "retained_minus_ast_nodes", "retained_minus_ast_transitions")]
    contrasts += [fresh["contrasts"][k] for k in ("ast_nodes", "fragment_retained", "fusion_masked")]
    observed = re.findall(r"^\s*(-?\d+\.\d+)/(-?\d+\.\d+)/(-?\d+\.\d+)/", old_plot, re.M)
    assert [tuple(map(float, r)) for r in observed] == [(r["mean"], *r["bonferroni_three_interval"]) for r in contrasts]
    plots = (problem_figure(top), cost_figure(cost), old_plot.strip())
    for i, plot in enumerate(plots, 1):
        reference = (ROOT / f"figures/reference/figure-{i}.tikz").read_text().strip()
        assert plot.strip() == reference, f"Figure {i} differs from the checked vector source"
        (output / f"figure-{i}.tex").write_text(PREAMBLE + plot.strip() + "\n\\end{document}\n")
    (output / "figure-data.json").write_text(json.dumps({
        "figure_1_problem_pairs": {k: top["methods"][k]["problem_ap8"] for k in ("D_BM25", "root_100")},
        "figure_1_primary": top["primary_root100_minus_denseBM25"],
        "figure_2": {k: {"MAP_at_8": v["MAP_at_8"], "gain_retained": v["fraction_full_MAP_gain_retained"],
                           "median_pass_mean_ms": v["timing_ms"]["median_pass_mean_total"]}
                     for k, v in cost["methods"].items()},
        "figure_3_MAP_values": maps, "figure_3_contrasts": contrasts}, indent=2, sort_keys=True) + "\n")
    print("PASS: 400 paired observations, both cost curves and 11 earlier MAP values / six intervals", flush=True)


if __name__ == "__main__":
    from verify_results import manifest
    manifest()
    generate()
