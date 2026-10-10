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
        (1, "only marks,mark=*,mark size=0.65pt,draw=figblue,fill=figblue,opacity=0.55"),
        (-1, "only marks,mark=triangle*,mark size=0.85pt,draw=figorange,fill=figorange,opacity=0.7"),
        (0, "only marks,mark=o,mark size=1pt,draw=black!70,mark options={solid}"),
    )
    for sign, style in styles:
        coordinates = []
        for p, d in zip(problems, deltas):
            if (d > 0) - (d < 0) == sign:
                coordinates.append(f"({scores['D_BM25'][p]:.12f},{scores['root_100'][p]:.12f})")
        scatter.append("\\addplot[" + style + "] coordinates {\n"
                       + "\n".join(coordinates) + "\n};")
    template = r"""\begin{tikzpicture}
\pgfplotsset{problem axis/.style={
  scale only axis,width=3.90cm,height=4.05cm,
  font=\footnotesize,tick label style={font=\scriptsize},
  label style={font=\footnotesize},
  axis line style={black!55,line width=0.4pt},
  tick style={black!55},tick align=outside,
  grid=major,grid style={black!10,line width=0.3pt},
  title style={font=\footnotesize\bfseries,at={(0,1.07)},anchor=west},
  legend style={font=\scriptsize,draw=none,fill=white,inner sep=2pt,
    at={(0.03,0.97)},anchor=north west},legend cell align=left}}
\begin{axis}[problem axis,at={(0,0)},anchor=south west,
  title={(a) Paired problem scores},
  xmin=-0.02,xmax=1.02,ymin=-0.02,ymax=1.02,
  xtick={0,0.25,0.5,0.75,1},ytick={0,0.25,0.5,0.75,1},
  xlabel={D + BM25: mean AP@8},ylabel={AST reranker: mean AP@8}]
\addplot[black!55,dashed,line width=0.6pt,forget plot]
  coordinates {(0,0) (1,1)};
SCATTER
\legend{Improved: 270,Reduced: 128,Unchanged: 2}
\end{axis}
\begin{axis}[problem axis,at={(5.3cm,0)},anchor=south west,
  title={(b) Paired differences},
  xmin=XMIN,xmax=XMAX,ymin=0,ymax=YMAX,
  xtick distance=0.1,ytick distance=20,scaled x ticks=false,
  xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},
  xlabel={Change in mean AP@8},ylabel={Number of problems}]
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
"""
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
    template = r"""\begin{tikzpicture}
\pgfplotsset{cost axis/.style={
  scale only axis,width=4.05cm,height=4.05cm,
  font=\footnotesize,tick label style={font=\scriptsize},
  label style={font=\footnotesize},
  axis line style={black!55,line width=0.4pt},
  tick align=outside,grid=major,grid style={black!10},
  xmin=0.75,xmax=4.25,xtick={1,2,3,4},xticklabels={50,100,500,Full},
  xlabel={Candidate budget},
  title style={font=\footnotesize\bfseries,at={(0,1.07)},anchor=west},
  legend style={font=\scriptsize,draw=none,fill=white,inner sep=2pt},
  legend cell align=left}}
\begin{axis}[cost axis,at={(0,0)},anchor=south west,
  title={(a) Gain preserved},ymin=90,ymax=102,ytick={90,94,98,100},
  ylabel={Full MAP@8 gain retained (\%)},
  legend style={at={(0.03,0.03)},anchor=south west}]
\addplot[figblue,mark=*,line width=0.7pt] coordinates {ROOTGAIN};
\addplot[figorange,mark=square*,dashed,line width=0.7pt] coordinates {IDFGAIN};
\legend{Fragments,Fragments + IDF}
\end{axis}
\begin{axis}[cost axis,at={(5.3cm,0)},anchor=south west,
  title={(b) Warm CPU ranking},ymin=0,ymax=6,ytick={0,2,4,6},
  ylabel={Time per query (ms)},
  legend style={at={(0.03,0.97)},anchor=north west}]
\addplot[figblue,mark=*,line width=0.7pt] coordinates {ROOTTIME};
\addplot[figorange,mark=square*,dashed,line width=0.7pt] coordinates {IDFTIME};
\addplot[black!65,densely dotted,line width=0.8pt] coordinates {(1,DENSETIME) (4,DENSETIME)};
\legend{Fragments,Fragments + IDF,Dense alone}
\end{axis}
\end{tikzpicture}
"""
    for prefix, tag in (("simpler_root", "ROOT"), ("candidate", "IDF")):
        rows = [methods[prefix + "_" + k] for k in ("50", "100", "500", "full")]
        full_gain = rows[-1]["MAP_at_8"] - methods["D"]["MAP_at_8"]
        for row in rows:
            assert abs((row["MAP_at_8"] - methods["D"]["MAP_at_8"]) / full_gain
                       - row["fraction_full_MAP_gain_retained"]) < 1e-12
            assert np.median(row["timing_ms"]["per_pass_mean_total"]) == row["timing_ms"]["median_pass_mean_total"]
        for suffix, values in (("GAIN", [100 * r["fraction_full_MAP_gain_retained"] for r in rows]),
                               ("TIME", [r["timing_ms"]["median_pass_mean_total"] for r in rows])):
            template = template.replace(tag + suffix, " ".join(f"({i},{v:.12f})" for i, v in enumerate(values, 1)))
    return template.replace("DENSETIME", f"{methods['D']['timing_ms']['median_pass_mean_total']:.12f}")
