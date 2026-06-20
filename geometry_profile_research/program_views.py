from __future__ import annotations

from dataclasses import dataclass

from .ast_features import build_ast_graph
from .graphs import SimpleGraph


@dataclass(frozen=True)
class ProgramView:
    """A named graph view of a program."""

    kind: str
    graph: SimpleGraph
    description: str


@dataclass(frozen=True)
class ProgramViews:
    """Container for code views with explicit unsupported-view markers."""

    ast: ProgramView
    cfg: ProgramView | None = None
    dfg: ProgramView | None = None
    cpg: ProgramView | None = None

    def available_views(self) -> list[str]:
        views = []
        if self.ast is not None:
            views.append("ast")
        if self.cfg is not None:
            views.append("cfg")
        if self.dfg is not None:
            views.append("dfg")
        if self.cpg is not None:
            views.append("cpg")
        return views


def extract_python_program_views(code: str) -> ProgramViews:
    """Extract currently implemented Python program views.

    The MVP intentionally exposes only AST. CFG, DFG and CPG require separate
    validated extractors; returning `None` is safer than creating misleading
    proxy graphs and later treating them as real control/data-flow views.
    """
    return ProgramViews(
        ast=ProgramView(
            kind="ast",
            graph=build_ast_graph(code),
            description="Python standard-library abstract syntax tree",
        )
    )
