"""Generate the current numerical figures and cost table, without manuscript text."""

import csv
import json
import re
import numpy as np
from verify_results import ROOT, read, verify_manifest
from figure_kernels import problem_figure, cost_figure

PREAMBLE = r"""\documentclass[border=3mm]{standalone}
\usepackage{fontspec}
\setmainfont{texgyretermes-regular.otf}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{figblue}{RGB}{0,114,178}
\definecolor{figorange}{RGB}{190,80,0}
\begin{document}
"""


def run():
    binding = verify_manifest()
    result = read("reproduced/result.json")
    assert result["status"] == "passed" and result["manifest_sha256"] == binding
    plots = [problem_figure(read("data/figures/primary_result.json")),
             cost_figure(result["sessions"]["A"])]
    transfer = (ROOT / "data/figures/transfer.tikz").read_text()
    cross = read("data/figures/crossed_summary.json")
    # The transfer graphic is unchanged; its numerical replay is in the external-transfer release.
    points = re.findall(r"^(-?\d+\.\d+) ([1-4]) (\d+\.\d+) (\d+\.\d+)$", transfer, re.M)
    assert len(points) == 4
    cells = ("CodeContests_CodeRank", "CodeContests_Qwen", "CPRet_CodeRank", "CPRet_Qwen")
    for mean, row, minus, plus in points:
        expected = cross["contrasts"]["root100_minus_D"][cells[4-int(row)]]
        np.testing.assert_allclose([float(mean), float(mean)-float(minus), float(mean)+float(plus)],
                                  [expected["mean"], *expected["percentile_95_interval"]], rtol=0, atol=2e-15)
    plots.append(transfer)
    out = ROOT / "reproduced"
    for number, plot in enumerate(plots, 1):
        picture = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", plot, re.S).group()
        (out / f"figure-{number}.tex").write_text(PREAMBLE+picture+"\n\\end{document}\n")
        (out / f"figure-{number}.tikz").write_text(picture+"\n")
    with (out / "cost_table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["session", "method", "MAP@8", "MRR", "Success@10", "mean_ms", "median_ms",
                         "p95_ms", "startup_s", "peak_reserved_GiB", "peak_allocated_GiB", "peak_RAM_GiB"])
        for session, record in result["sessions"].items():
            for method, row in record["methods"].items():
                memory = row["memory"]
                writer.writerow([session, method, *row["quality"].values(),
                    *[1000*row["latency"][k] for k in ("mean_seconds", "median_seconds", "p95_seconds")],
                    row["index"]["ready_startup_excluding_network_seconds"],
                    memory.get("peak_cuda_reserved_bytes",0)/2**30,
                    memory.get("peak_cuda_allocated_bytes",0)/2**30,
                    memory["peak_host_rss_bytes_before_quality"]/2**30])
    (out / "figures.json").write_text(json.dumps({"status":"passed", "figures":3,
        "cost_sessions_separate":True, "manuscript_included":False,
        "scope":"Current cost plot from raw-record replay; primary bootstrap checked; unchanged transfer plot from previously verified release"}, indent=2)+"\n")
    print("PASS: three figure sources and separate-session cost table; compile with XeLaTeX")


if __name__ == "__main__":
    run()
