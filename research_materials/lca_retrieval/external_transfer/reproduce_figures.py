"""Three current vector figures from verified numerical data, without article prose."""
import json
import re
import numpy as np
from figure_kernels import problem_figure, cost_figure
from verify_results import ROOT, read, sha, close, arrays, manifest

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


def main():
    manifest()
    verified = read("reproduced/verification.json")
    assert verified["status"] == "passed" and verified["manifest_sha256"] == sha(ROOT/"MANIFEST.json")
    top = read("data/figures/top100_fresh_20261010/result.json")
    cost = read("data/figures/two_stage_20261010/result.json")
    assert sha(ROOT/"data/figures/top100_fresh_20261010/result.json") == "e116ce044a421bba43c9ad48877b8f42c166169ab7f93b4cf44a26e707078282"
    assert sha(ROOT/"data/figures/two_stage_20261010/result.json") == "98a1d7095dd6a2d4e690862c436cdf1e3f171df51a384f618aea0ae980c39034"
    cross_arrays = arrays("data/CodeContests_CodeRank/arrays.npz")
    from kernels import METHODS
    from verify_results import cohort, grouped
    groups = cohort("CodeContests")["groups"]
    native = grouped(cross_arrays["native"], groups)
    problem_ids = sorted(set(groups))
    for name in ("D_BM25", "root_100"):
        close(native[METHODS.index(name), :, 0], [top["methods"][name]["problem_ap8"][p] for p in problem_ids])
    delta = native[METHODS.index("root_100"), :, 0] - native[METHODS.index("D_BM25"), :, 0]
    rng = np.random.Generator(np.random.PCG64(20261010))
    sampled = delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(axis=1)
    primary = top["primary_root100_minus_denseBM25"]
    close(delta.mean(), primary["mean"])
    close(np.quantile(sampled, [.025, .975]), primary["percentile_95_interval"])
    timing = arrays("data/figures/two_stage_20261010/timing.npz")["milliseconds"]
    methods = ["D"] + [f"{family}_{k}" for family in ("candidate", "simpler_root") for k in ("full", 50, 100, 500)]
    for i, name in enumerate(methods):
        close(np.median(timing[:, :, i, -1].mean(axis=1)), cost["methods"][name]["timing_ms"]["median_pass_mean_total"])
    # The first two results are the exact earlier release inputs, not new measurements.
    plots = [problem_figure(top), cost_figure(cost)]
    transfer = (ROOT/"figures/reference/figure-3.tikz").read_text().strip()
    effects = verified["crossed"]["contrasts"]["root100_minus_D"]
    coordinates = []
    for name, row in (("CodeContests_CodeRank", 4), ("CPRet_CodeRank", 2),
                      ("CodeContests_Qwen", 3), ("CPRet_Qwen", 1)):
        e = effects[name]
        mean, (low, high) = e["mean"], e["percentile_95_interval"]
        coordinates.append(f"{mean:.15f} {row} {mean-low:.15f} {high-mean:.15f}")
    actual = iter(coordinates)
    transfer, count = re.subn(r"^-?\d+\.\d+ [1-4] \d+\.\d+ \d+\.\d+$", lambda _: next(actual), transfer, flags=re.M)
    assert count == 4
    plots.append(transfer)
    for i, plot in enumerate(plots, 1):
        assert plot.strip() == (ROOT/f"figures/reference/figure-{i}.tikz").read_text().strip(), i
        (ROOT/f"reproduced/figure-{i}.tex").write_text(PREAMBLE+plot.strip()+"\n\\end{document}\n")
    report = {"status": "passed", "figures": 3, "paired_primary_problems": 400,
              "crossed_effects": effects, "timing_samples": list(timing.shape),
              "manuscript_text_included": False, "pdf_rendering_requires_xelatex": True}
    (ROOT/"reproduced/figures.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("PASS: all three current figure sources, 400 paired observations, saved timing samples and four crossed intervals")


if __name__ == "__main__":
    main()
