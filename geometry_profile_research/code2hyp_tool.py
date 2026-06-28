from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch

from geometry_profile_research.constant_curvature import (
    ConstantCurvatureProduct,
    ProductMeasure,
    median_positive_cost_scale,
    scaled_sinkhorn_epsilon,
)
from geometry_profile_research.python_raw_ast import parse_python_ast_tree
from geometry_profile_research.raw_ast import NodeId, RawAstPath, RawAstTree, terminal_to_terminal_paths

LabelMode = Literal["scalar_hash", "categorical", "none"]
PYTHON_AST_LABELS = tuple(
    sorted(
        name
        for name, value in vars(ast).items()
        if isinstance(value, type) and issubclass(value, ast.AST)
    )
)
LABEL_VOCAB = {label: index for index, label in enumerate((*PYTHON_AST_LABELS, "TerminalToken", "__OTHER__"))}


@dataclass(frozen=True)
class PathDescriptor:
    """Human-readable metadata for one LCA-product AST path object."""

    path_index: int
    lca_label: str
    start_label: str
    end_label: str
    lca_depth: int
    path_length: int
    start_source_span: str
    end_source_span: str
    lca_source_span: str


@dataclass(frozen=True)
class EncodedProgram:
    """A source file represented as a measure over AST path objects."""

    item_id: str
    path: str
    language: str
    measure: ProductMeasure
    descriptors: tuple[PathDescriptor, ...]


@dataclass(frozen=True)
class SearchResult:
    """One structural retrieval result."""

    path: str
    distance: float
    path_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Code2HypConfig:
    """Frozen public configuration for the deterministic Code2Hyp-v1 tool."""

    model_name: str = "code2hyp-v1"
    language: str = "python"
    max_paths: int = 128
    point_dim: int = 4
    curvature: float = 0.0
    side_weight: float = 1.0
    label_mode: LabelMode = "scalar_hash"
    sinkhorn_kappa: float = 0.05
    sinkhorn_iterations: int = 128


class Code2Hyp:
    """Deterministic structural retrieval model over raw Python ASTs.

    The model follows the object-order-aware representation used in the
    Code2Hyp experiments: AST nodes are points, terminal-to-terminal paths are
    LCA-anchored product objects ``(LCA, start, end)``, and a file is a finite
    measure over these path objects.
    """

    def __init__(self, config: Code2HypConfig | None = None) -> None:
        self.config = config or Code2HypConfig()
        if self.config.label_mode not in {"scalar_hash", "categorical", "none"}:
            raise ValueError(f"unknown label_mode: {self.config.label_mode!r}")
        self.geometry = ConstantCurvatureProduct(
            curvature=self.config.curvature,
            factor_weights=(1.0, 1.0, 1.0),
            side_weight=self.config.side_weight,
            unoriented=True,
        )

    @classmethod
    def load(cls, model_name: str = "code2hyp-v1") -> Code2Hyp:
        if model_name != "code2hyp-v1":
            raise ValueError(f"unknown Code2Hyp model: {model_name}")
        return cls(Code2HypConfig(model_name=model_name))

    def encode_file(self, path: str | Path) -> EncodedProgram:
        file_path = Path(path)
        if file_path.suffix != ".py":
            raise ValueError("Code2Hyp-v1 currently supports Python .py files")
        source = file_path.read_text(encoding="utf-8")
        tree = parse_python_ast_tree(source)
        return self.encode_tree(tree, item_id=file_path.stem, path=str(file_path))

    def encode_tree(self, tree: RawAstTree, *, item_id: str, path: str) -> EncodedProgram:
        paths = terminal_to_terminal_paths(tree, max_paths=self.config.max_paths)
        if not paths:
            paths = _fallback_root_paths(tree)
        node_points = _node_points(tree, point_dim=self.config.point_dim, label_mode=self.config.label_mode)
        product_points = torch.stack(
            [
                torch.stack(
                    (
                        node_points[path_object.lca(tree)],
                        node_points[path_object.start],
                        node_points[path_object.end],
                    ),
                    dim=0,
                )
                for path_object in paths
            ],
            dim=0,
        )
        side_features = torch.stack(
            [_path_side_features(tree, path_object, label_mode=self.config.label_mode) for path_object in paths],
            dim=0,
        )
        mass = torch.ones(len(paths), dtype=torch.float32) / float(len(paths))
        measure = ProductMeasure(
            points=self.geometry.project(product_points),
            mass=mass,
            side_features=side_features,
        )
        descriptors = tuple(
            _path_descriptor(tree, path_index=index, path_object=path_object)
            for index, path_object in enumerate(paths)
        )
        return EncodedProgram(
            item_id=item_id,
            path=path,
            language=self.config.language,
            measure=measure,
            descriptors=descriptors,
        )

    def index_directory(
        self,
        root: str | Path,
        *,
        pattern: str = "*.py",
        recursive: bool = True,
    ) -> Code2HypIndex:
        root_path = Path(root)
        iterator = root_path.rglob(pattern) if recursive else root_path.glob(pattern)
        entries = tuple(self.encode_file(path) for path in sorted(iterator) if path.is_file())
        return Code2HypIndex(model=self, entries=entries)

    def audit_directory(
        self,
        root: str | Path,
        *,
        pattern: str = "*.py",
        recursive: bool = True,
        max_pairs: int = 256,
    ) -> dict[str, Any]:
        index = self.index_directory(root, pattern=pattern, recursive=recursive)
        return self.audit_index(index, max_pairs=max_pairs)

    def audit_index(self, index: Code2HypIndex, *, max_pairs: int = 256) -> dict[str, Any]:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        pairs = _entry_pairs(index.entries, max_pairs=max_pairs)
        full_costs: list[torch.Tensor] = []
        point_costs: list[torch.Tensor] = []
        side_costs: list[torch.Tensor] = []
        for left, right in pairs:
            full_cost, point_cost, side_cost = _cost_components(self.geometry, left.measure, right.measure)
            full_costs.append(full_cost)
            point_costs.append(point_cost)
            side_costs.append(side_cost)
        full_total = _sum_matrices(full_costs)
        point_total = _sum_matrices(point_costs)
        side_total = _sum_matrices(side_costs)
        denominator = full_total if full_total > 0.0 else 1.0
        return {
            "model": self.config.model_name,
            "entries": len(index.entries),
            "pair_count": len(pairs),
            "path_objects": sum(len(entry.descriptors) for entry in index.entries),
            "point_cost_share": point_total / denominator,
            "side_cost_share": side_total / denominator,
            "median_positive_full_cost": median_positive_cost_scale(full_costs, fallback=0.0),
        }

    def compare_files(self, left: str | Path, right: str | Path) -> dict[str, Any]:
        left_program = self.encode_file(left)
        right_program = self.encode_file(right)
        distance = self.distance(left_program, right_program)
        return {
            "model": self.config.model_name,
            "left": left_program.path,
            "right": right_program.path,
            "distance": distance,
            "left_path_count": len(left_program.descriptors),
            "right_path_count": len(right_program.descriptors),
        }

    def explain_files(self, left: str | Path, right: str | Path, *, top_k: int = 10) -> dict[str, Any]:
        left_program = self.encode_file(left)
        right_program = self.encode_file(right)
        return self.explain(left_program, right_program, top_k=top_k)

    def distance(self, left: EncodedProgram, right: EncodedProgram) -> float:
        scale = self._cost_scale(left.measure, right.measure)
        epsilon = scaled_sinkhorn_epsilon(scale, kappa=self.config.sinkhorn_kappa)
        value = self.geometry.sinkhorn_divergence(
            left.measure,
            right.measure,
            epsilon=epsilon,
            iterations=self.config.sinkhorn_iterations,
        )
        return round(max(float(value.detach()), 0.0), 12)

    def explain(self, left: EncodedProgram, right: EncodedProgram, *, top_k: int = 10) -> dict[str, Any]:
        scale = self._cost_scale(left.measure, right.measure)
        epsilon = scaled_sinkhorn_epsilon(scale, kappa=self.config.sinkhorn_kappa)
        local_cost = self.geometry.path_cost_matrix(left.measure, right.measure)
        plan = self.geometry.transport_plan(
            left.measure,
            right.measure,
            epsilon=epsilon,
            iterations=self.config.sinkhorn_iterations,
        )
        alignments = _top_transport_alignments(
            left,
            right,
            plan=plan,
            local_cost=local_cost,
            top_k=top_k,
        )
        return {
            "model": self.config.model_name,
            "query": left.path,
            "candidate": right.path,
            "distance": self.distance(left, right),
            "alignments": alignments,
        }

    def _cost_scale(self, left: ProductMeasure, right: ProductMeasure) -> float:
        return median_positive_cost_scale(
            [
                self.geometry.path_cost_matrix(left, right),
                self.geometry.path_cost_matrix(left, left),
                self.geometry.path_cost_matrix(right, right),
            ],
            fallback=1.0,
        )


class Code2HypIndex:
    """In-memory structural code index."""

    def __init__(self, *, model: Code2Hyp, entries: Sequence[EncodedProgram]) -> None:
        self.model = model
        self.entries = tuple(entries)
        self.model_name = model.config.model_name

    def search(self, query: str | Path | EncodedProgram, *, top_k: int = 20) -> list[SearchResult]:
        query_program = query if isinstance(query, EncodedProgram) else self.model.encode_file(query)
        results = [
            SearchResult(
                path=entry.path,
                distance=self.model.distance(query_program, entry),
                path_count=len(entry.descriptors),
            )
            for entry in self.entries
        ]
        results.sort(key=lambda result: (result.distance, result.path != query_program.path, result.path))
        return results[:top_k]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Code2HypIndex:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": asdict(self.model.config),
            "entries": [_encoded_program_to_json(entry) for entry in self.entries],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Code2HypIndex:
        config_values = payload.get("config", {})
        model_name = payload.get("model_name", config_values.get("model_name", "code2hyp-v1"))
        config = Code2HypConfig(**{**asdict(Code2HypConfig(model_name=model_name)), **config_values})
        model = Code2Hyp(config)
        entries = tuple(_encoded_program_from_json(entry) for entry in payload.get("entries", []))
        return cls(model=model, entries=entries)


def _encoded_program_to_json(program: EncodedProgram) -> dict[str, Any]:
    return {
        "item_id": program.item_id,
        "path": program.path,
        "language": program.language,
        "points": program.measure.points.detach().cpu().tolist(),
        "mass": program.measure.mass.detach().cpu().tolist(),
        "side_features": (
            None
            if program.measure.side_features is None
            else program.measure.side_features.detach().cpu().tolist()
        ),
        "descriptors": [asdict(descriptor) for descriptor in program.descriptors],
    }


def _encoded_program_from_json(payload: dict[str, Any]) -> EncodedProgram:
    measure = ProductMeasure(
        points=torch.tensor(payload["points"], dtype=torch.float32),
        mass=torch.tensor(payload["mass"], dtype=torch.float32),
        side_features=(
            None
            if payload.get("side_features") is None
            else torch.tensor(payload["side_features"], dtype=torch.float32)
        ),
    )
    return EncodedProgram(
        item_id=payload["item_id"],
        path=payload["path"],
        language=payload.get("language", "python"),
        measure=measure,
        descriptors=tuple(PathDescriptor(**descriptor) for descriptor in payload.get("descriptors", [])),
    )


def _node_points(tree: RawAstTree, *, point_dim: int, label_mode: LabelMode = "scalar_hash") -> dict[NodeId, torch.Tensor]:
    order = tree.preorder()
    index_by_node = {node: index for index, node in enumerate(order)}
    max_depth = max((tree.depth(node) for node in order), default=1) or 1
    max_index = max(len(order) - 1, 1)
    max_siblings = max(
        (len(tree.children_by_node.get(parent, ())) for parent in order),
        default=1,
    ) or 1
    points: dict[NodeId, torch.Tensor] = {}
    for node in order:
        parent = tree.parent(node)
        if parent is None:
            sibling_position = 0.0
        else:
            siblings = tree.children_by_node.get(parent, ())
            sibling_position = siblings.index(node) / max(max_siblings - 1, 1)
        label_feature = _stable_label_value(tree.labels.get(node, "")) if label_mode == "scalar_hash" else 0.0
        features = [
            tree.depth(node) / max_depth,
            index_by_node[node] / max_index,
            sibling_position,
            label_feature,
        ]
        points[node] = torch.tensor(features[:point_dim], dtype=torch.float32)
    return points


def _stable_label_value(label: str) -> float:
    if not label:
        return 0.0
    value = sum((index + 1) * ord(char) for index, char in enumerate(label))
    return (value % 997) / 996.0


def _path_side_features(tree: RawAstTree, path_object: RawAstPath, *, label_mode: LabelMode = "scalar_hash") -> torch.Tensor:
    lca = path_object.lca(tree)
    max_depth = max((tree.depth(node) for node in tree.preorder()), default=1) or 1
    max_length = max(1, len(tree.parent_by_node) - 1)
    numeric = [
        path_object.length / max_length,
        tree.depth(lca) / max_depth,
        tree.depth(path_object.start) / max_depth,
        tree.depth(path_object.end) / max_depth,
    ]
    if label_mode == "scalar_hash":
        numeric.extend(
            [
                _stable_label_value(tree.labels.get(lca, "")),
                _stable_label_value(tree.labels.get(path_object.start, "")),
                _stable_label_value(tree.labels.get(path_object.end, "")),
            ]
        )
        return torch.tensor(numeric, dtype=torch.float32)
    if label_mode == "none":
        return torch.tensor(numeric, dtype=torch.float32)
    if label_mode == "categorical":
        categorical = torch.cat(
            (
                _label_one_hot(tree.labels.get(lca, "")),
                _label_one_hot(tree.labels.get(path_object.start, "")),
                _label_one_hot(tree.labels.get(path_object.end, "")),
            )
        )
        return torch.cat((torch.tensor(numeric, dtype=torch.float32), categorical))
    raise ValueError(f"unknown label_mode: {label_mode!r}")


def _label_one_hot(label: str) -> torch.Tensor:
    vector = torch.zeros(len(LABEL_VOCAB), dtype=torch.float32)
    vector[LABEL_VOCAB.get(label, LABEL_VOCAB["__OTHER__"])] = 1.0
    return vector


def _path_descriptor(tree: RawAstTree, *, path_index: int, path_object: RawAstPath) -> PathDescriptor:
    lca = path_object.lca(tree)
    return PathDescriptor(
        path_index=path_index,
        lca_label=_display_label(tree, lca),
        start_label=_display_label(tree, path_object.start),
        end_label=_display_label(tree, path_object.end),
        lca_depth=tree.depth(lca),
        path_length=path_object.length,
        start_source_span=_source_span(tree, path_object.start),
        end_source_span=_source_span(tree, path_object.end),
        lca_source_span=_source_span(tree, lca),
    )


def _source_span(tree: RawAstTree, node: NodeId) -> str:
    return tree.attributes.get(node, {}).get("source_span", "")


def _display_label(tree: RawAstTree, node: NodeId) -> str:
    label = tree.labels.get(node, "")
    attributes = tree.attributes.get(node, {})
    terminal = attributes.get("terminal_token")
    if terminal:
        return f"token:{_short_label_value(terminal)}"
    name = attributes.get("name")
    if name:
        return f"{label}:{_short_label_value(name)}"
    return label


def _short_label_value(value: str, *, limit: int = 24) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def _fallback_root_paths(tree: RawAstTree) -> tuple[RawAstPath, ...]:
    order = tree.preorder()
    if len(order) == 1:
        return (RawAstPath(start=tree.root_id, end=tree.root_id, nodes=(tree.root_id,)),)
    return (tree.path_between(order[0], order[-1]),)


def _top_transport_alignments(
    left: EncodedProgram,
    right: EncodedProgram,
    *,
    plan: torch.Tensor,
    local_cost: torch.Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    pairs: list[tuple[float, float, int, int]] = []
    for left_index, right_index in _matrix_indices(plan):
        mass = float(plan[left_index, right_index].detach())
        if mass <= 0.0:
            continue
        cost = float(local_cost[left_index, right_index].detach())
        pairs.append((mass, cost, left_index, right_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    alignments: list[dict[str, Any]] = []
    for mass, cost, left_index, right_index in pairs[:top_k]:
        left_descriptor = left.descriptors[left_index]
        right_descriptor = right.descriptors[right_index]
        alignments.append(
            {
                "transport_mass": mass,
                "local_cost": cost,
                "query_path_index": left_descriptor.path_index,
                "candidate_path_index": right_descriptor.path_index,
                "query_lca_label": left_descriptor.lca_label,
                "candidate_lca_label": right_descriptor.lca_label,
                "query_start_label": left_descriptor.start_label,
                "candidate_start_label": right_descriptor.start_label,
                "query_end_label": left_descriptor.end_label,
                "candidate_end_label": right_descriptor.end_label,
                "query_source_span": left_descriptor.lca_source_span,
                "candidate_source_span": right_descriptor.lca_source_span,
                "query_path": asdict(left_descriptor),
                "candidate_path": asdict(right_descriptor),
            }
        )
    return alignments


def _entry_pairs(
    entries: Sequence[EncodedProgram],
    *,
    max_pairs: int,
) -> list[tuple[EncodedProgram, EncodedProgram]]:
    pairs: list[tuple[EncodedProgram, EncodedProgram]] = []
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1 :]:
            pairs.append((left, right))
            if len(pairs) >= max_pairs:
                return pairs
    if not pairs and entries:
        pairs.append((entries[0], entries[0]))
    return pairs


def _cost_components(
    geometry: ConstantCurvatureProduct,
    left: ProductMeasure,
    right: ProductMeasure,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    point_geometry = ConstantCurvatureProduct(
        curvature=geometry.curvature,
        factor_weights=geometry.factor_weights,
        side_weight=0.0,
        unoriented=geometry.unoriented,
        eps=geometry.eps,
    )
    full_cost = geometry.path_cost_matrix(left, right)
    point_cost = point_geometry.path_cost_matrix(left, right)
    side_cost = torch.clamp(full_cost - point_cost, min=0.0)
    return full_cost, point_cost, side_cost


def _sum_matrices(matrices: Sequence[torch.Tensor]) -> float:
    if not matrices:
        return 0.0
    return float(sum(matrix.detach().sum() for matrix in matrices))


def _matrix_indices(matrix: torch.Tensor) -> Iterable[tuple[int, int]]:
    rows, columns = matrix.shape
    for row in range(rows):
        for column in range(columns):
            yield row, column
