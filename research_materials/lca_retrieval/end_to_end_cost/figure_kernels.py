"""Unchanged numerical plot functions from the local figure verifier."""

import numpy as np

def problem_figure(result):
    methods = result["methods"]
    names = ("D_BM25", "root_100")
    scores = {name: methods[name]["problem_ap8"] for name in names}
    problems = sorted(scores["D_BM25"])
    assert len(problems) == 400
    assert all(set(scores[name]) == set(problems) for name in names)
    # Eight queries, eight relevant items, ranks 1..8: exact AP means lie
    # on the 1/(8*8*lcm(1..8)) grid. This avoids counting roundoff as a gain.
    denominator = 53760
    ticks = {name: {p: round(scores[name][p] * denominator) for p in problems}
             for name in names}
    for name in names:
        assert all(abs(scores[name][p] * denominator - ticks[name][p]) < 1e-8
                   for p in problems)
        assert abs(sum(scores[name].values()) / 400
                   - methods[name]["MAP_at_8"]) < 1e-12
    deltas = [ticks["root_100"][p] - ticks["D_BM25"][p] for p in problems]
    counts = tuple(sum((d > 0, d < 0, d == 0)[i] for d in deltas) for i in range(3))
    assert counts == (270, 128, 2)
    assert counts == tuple(result["primary_problem_counts"][k] for k in ("better", "worse", "tied"))
    contrast = result["primary_root100_minus_denseBM25"]
    assert abs(sum(deltas) / (400 * denominator) - contrast["mean"]) < 1e-12
    values = np.array([scores["root_100"][p] - scores["D_BM25"][p] for p in problems])
    indices = np.random.Generator(np.random.PCG64(20261010)).integers(400, size=(20000, 400))
    np.testing.assert_allclose(np.quantile(values[indices].mean(axis=1), [.025, .975]),
                               contrast["percentile_95_interval"], rtol=0, atol=1e-12)

    # Zero-centred bins include every problem, including losses and exact ties.
    indices = [(d * 200 + denominator) // (2 * denominator) for d in deltas]
    centres = list(range(min(indices), max(indices) + 1))
    bins = {i: 0 for i in centres}
    for index in indices:
        assert index in bins
        bins[index] += 1
    assert sum(bins.values()) == 400
    ymax = ((max(bins.values()) + 29) // 20) * 20
    scatter = []
    styles = (
        (1, "only marks,mark=*,mark size=0.8pt,draw=figblue,fill=figblue,opacity=0.65"),
        (-1, "only marks,mark=triangle*,mark size=1.05pt,draw=figorange,fill=figorange,opacity=0.8"),
        (0, "only marks,mark=o,mark size=1.3pt,draw=black,mark options={solid}"),
    )
    for sign, style in styles:
        coordinates = []
        for p, d in zip(problems, deltas):
            if (d > 0) - (d < 0) == sign:
                coordinates.append(f"({scores['D_BM25'][p]:.12f},{scores['root_100'][p]:.12f})")
        scatter.append("\\addplot[" + style + "] coordinates {\n"
                       + "\n".join(coordinates) + "\n};")
    template = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\pgfplotsset{problem axis/.style={
  scale only axis,width=3.90cm,height=3.90cm,
  font=\footnotesize,tick label style={font=\footnotesize},
  label style={font=\footnotesize},
  axis line style={black!55,line width=0.4pt},
  tick style={black!55},tick align=outside,
  grid=major,grid style={black!10,line width=0.3pt},
  title style={font=\footnotesize\bfseries,at={(0,1.07)},anchor=west},
  legend style={font=\footnotesize,draw=none,fill=white,inner sep=2pt,
    at={(0.03,0.97)},anchor=north west},legend cell align=left}}
\begin{axis}[problem axis,at={(0,0)},anchor=south west,
  title={(a) Scores for each problem},
  xmin=-0.02,xmax=1.02,ymin=-0.02,ymax=1.02,
  xtick={0,0.25,0.5,0.75,1},ytick={0,0.25,0.5,0.75,1},
  xlabel={D + BM25: mean AP@8},ylabel={AST reranker: mean AP@8}]
\addplot[black!55,dashed,line width=0.6pt,forget plot]
  coordinates {(0,0) (1,1)};
SCATTER
\legend{Higher: 270,Lower: 128,Tied: 2}
\end{axis}
\begin{axis}[problem axis,at={(5.3cm,0)},anchor=south west,
  title={(b) Paired differences},
  xmin=XMIN,xmax=XMAX,ymin=0,ymax=YMAX,
  xtick distance=0.1,ytick distance=20,scaled x ticks=false,
  xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},
  xlabel={Reranker minus baseline},ylabel={Number of problems}]
\addplot[ybar interval,fill=figblue!55,draw=figblue!80,line width=0.3pt,
  forget plot] coordinates {
HISTOGRAM
};
\addplot[black!65,densely dotted,line width=0.65pt,forget plot]
  coordinates {(0,0) (0,YMAX)};
\addplot[figorange,dashed,line width=0.8pt]
  coordinates {(MEAN,0) (MEAN,YMAX)};
\legend{Mean: $+0.02275$}
\end{axis}
\end{tikzpicture}
\caption{Primary comparison on 400 unused problems. (a) Problem-average AP@8 for CodeRankEmbed (D) + BM25 and the top-100 AST reranker; the diagonal marks equality. (b) All paired AP@8 differences, with bin width 0.01. Dotted and dashed lines mark zero and the mean. Mean gain: 0.02275; paired problem-bootstrap 95\% interval: \mbox{[0.01754, 0.02793]}.}
\label{fig:problem-effects}
\end{figure}"""
    # ybar interval consumes left edges plus a closing edge, not centres.
    histogram = "\n".join(f"({i / 100 - 0.005:.4f},{bins[i]})" for i in centres)
    histogram += f"\n({centres[-1] / 100 + 0.005:.4f},0)"
    return (template.replace("SCATTER", "\n".join(scatter))
            .replace("HISTOGRAM", histogram).replace("YMAX", str(ymax))
            .replace("MEAN", f"{contrast['mean']:.15f}")
            .replace("XMIN", f"{centres[0] / 100 - .01:.4f}")
            .replace("XMAX", f"{centres[-1] / 100 + .01:.4f}"))

def cost_figure(result):
    methods = result["methods"]
    template = r"""\begin{figure}[!htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
  scale only axis,width=8.8cm,height=5.0cm,
  font=\footnotesize,tick label style={font=\footnotesize},
  label style={font=\footnotesize},
  axis line style={black!55,line width=0.4pt},
  tick align=outside,grid=major,grid style={black!10},
  xmode=log,xmin=0.8,xmax=50000,ymin=0,ymax=1,
  xtick={1,10,100,1000,10000},
  xticklabels={1,10,100,{1,000},{10,000}},
  ytick={0,0.2,0.4,0.6,0.8,1},
  xlabel={Raw-query latency (ms; logarithmic scale)},
  ylabel={MAP@8 on 3,200 queries}]
POINTS
\end{axis}
\end{tikzpicture}
\caption{Session A quality--latency comparison. Points: means; lines: mean to 95th percentile, not confidence bounds. D: CodeRankEmbed; Qwen-E/R: 0.6B embedding/reranking models. Quality uses 3,200 queries (saved pair scores for Qwen-R); timings use 400 queries in three passes. Startup and memory: Table~\ref{tab:cost}.}
\label{fig:reranking-cost}
\end{figure}"""
    points = []
    for key, label, style, offset in (
        ("BM25", "BM25", "black!65,mark=triangle*", "above right"),
        ("D_BM25", "D + BM25", "black!65,mark=square*", "below left,yshift=-3pt"),
        ("D_AST", "D + AST", "figblue,mark=*", "above left,yshift=3pt"),
        ("QwenEmbedding", "Qwen-E", "figorange,mark=diamond*", "above left"),
        ("D_QwenReranker", "D + Qwen-R", "figorange,mark=triangle*", "above left"),
    ):
        row = methods[key]
        quality = row["quality"] or row["previous_full_quality_not_new_host_receipt"]
        mean = 1000 * row["latency"]["mean_seconds"]
        p95 = 1000 * row["latency"]["p95_seconds"]
        assert 0 < mean < p95 and 0 <= quality["MAP_at_8"] <= 1
        y = quality["MAP_at_8"]
        points.append(f"\\addplot[{style},mark indices={{1}},mark size=2.2pt,line width=0.8pt] "
                      f"coordinates {{({mean:.12f},{y:.12f}) ({p95:.12f},{y:.12f})}};")
        points.append(f"\\node[{offset},font=\\footnotesize] at "
                      f"(axis cs:{mean:.12f},{y:.12f}) {{{label}}};")
    return template.replace("POINTS", "\n".join(points))
