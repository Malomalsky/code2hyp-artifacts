from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch import Tensor

from .code2hyp_torch import Code2HypBatch, Code2HypTorchConfig, tree_context_features

LexicalAblationMode = Literal["original", "obfuscated", "record_obfuscated", "structural_only"]
LEXICAL_MASK_TOKEN = "<LEXICAL_MASK>"


@dataclass(frozen=True)
class RawCode2VecContext:
    start_token: str
    ast_path: tuple[str, ...]
    end_token: str


@dataclass(frozen=True)
class RawCode2VecRecord:
    label: str
    contexts: tuple[RawCode2VecContext, ...]


@dataclass(frozen=True)
class EncodedCode2HypDataset:
    batch: Code2HypBatch
    labels: Tensor
    model_config: Code2HypTorchConfig
    token_vocab: "Vocabulary"
    ast_node_vocab: "Vocabulary"
    label_vocab: "Vocabulary"


@dataclass(frozen=True)
class EncodedCode2HypMultiLabelDataset:
    batch: Code2HypBatch
    labels: Tensor
    target_sizes: Tensor
    model_config: Code2HypTorchConfig
    token_vocab: "Vocabulary"
    ast_node_vocab: "Vocabulary"
    target_vocab: "Vocabulary"


class Vocabulary:
    pad_token = "<PAD>"
    unk_token = "<UNK>"

    def __init__(self, special_tokens: bool = True) -> None:
        if special_tokens:
            self._token_to_id = {self.pad_token: 0, self.unk_token: 1}
            self._id_to_token = [self.pad_token, self.unk_token]
        else:
            self._token_to_id = {}
            self._id_to_token = []

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self._id_to_token)

    def add(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._id_to_token)
            self._id_to_token.append(token)
        return self._token_to_id[token]

    def lookup(self, token: str) -> int:
        if self.unk_token in self._token_to_id:
            return self._token_to_id.get(token, self.unk_id)
        return self._token_to_id[token]

    def token(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def contains(self, token: str) -> bool:
        return token in self._token_to_id


def parse_code2vec_line(line: str) -> RawCode2VecRecord:
    """Parse one code2vec-style line: `label token,path,node ...`."""
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty code2vec line")
    label = parts[0]
    contexts: list[RawCode2VecContext] = []
    for raw_context in parts[1:]:
        fields = raw_context.split(",")
        if len(fields) != 3:
            continue
        start_token, raw_path, end_token = fields
        ast_path = tuple(node for node in raw_path.split("|") if node)
        if not start_token or not end_token or not ast_path:
            continue
        contexts.append(
            RawCode2VecContext(
                start_token=start_token,
                ast_path=ast_path,
                end_token=end_token,
            )
        )
    if not contexts:
        raise ValueError(f"line has no valid path contexts: {line[:80]}")
    return RawCode2VecRecord(label=label, contexts=tuple(contexts))


def iter_code2vec_records(lines: Iterable[str]) -> Iterable[RawCode2VecRecord]:
    """Yield parsed records from code2vec/code2seq preprocessed text lines."""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        yield parse_code2vec_line(stripped)


def _stable_obfuscated_token(token: str) -> str:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
    return f"LEX_{digest}"


def apply_lexical_ablation(
    records: Iterable[RawCode2VecRecord],
    mode: LexicalAblationMode,
) -> list[RawCode2VecRecord]:
    """Transform endpoint tokens while leaving labels and AST paths unchanged."""
    if mode == "original":
        return list(records)
    if mode not in {"obfuscated", "record_obfuscated", "structural_only"}:
        raise ValueError(
            "lexical ablation mode must be one of: original, obfuscated, record_obfuscated, structural_only"
        )

    transformed_records: list[RawCode2VecRecord] = []
    for record in records:
        transformed_contexts: list[RawCode2VecContext] = []
        local_token_ids: dict[str, str] = {}
        for context in record.contexts:
            if mode == "obfuscated":
                start_token = _stable_obfuscated_token(context.start_token)
                end_token = _stable_obfuscated_token(context.end_token)
            elif mode == "record_obfuscated":
                if context.start_token not in local_token_ids:
                    local_token_ids[context.start_token] = f"LEX_LOCAL_{len(local_token_ids)}"
                if context.end_token not in local_token_ids:
                    local_token_ids[context.end_token] = f"LEX_LOCAL_{len(local_token_ids)}"
                start_token = local_token_ids[context.start_token]
                end_token = local_token_ids[context.end_token]
            else:
                start_token = LEXICAL_MASK_TOKEN
                end_token = LEXICAL_MASK_TOKEN
            transformed_contexts.append(
                RawCode2VecContext(
                    start_token=start_token,
                    ast_path=context.ast_path,
                    end_token=end_token,
                )
            )
        transformed_records.append(
            RawCode2VecRecord(
                label=record.label,
                contexts=tuple(transformed_contexts),
            )
        )
    return transformed_records


def load_code2vec_records(path: str | Path, limit: int | None = None) -> list[RawCode2VecRecord]:
    """Load real preprocessed code2vec/code2seq records from a `.c2v` split file."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    records: list[RawCode2VecRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for record in iter_code2vec_records(handle):
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def split_label_subtokens(label: str) -> tuple[str, ...]:
    subtokens = tuple(subtoken for subtoken in label.split("|") if subtoken)
    return subtokens or (label,)


def build_vocabularies(records: Iterable[RawCode2VecRecord]) -> tuple[Vocabulary, Vocabulary, Vocabulary]:
    token_vocab = Vocabulary()
    ast_node_vocab = Vocabulary()
    label_vocab = Vocabulary(special_tokens=False)
    for record in records:
        label_vocab.add(record.label)
        for context in record.contexts:
            token_vocab.add(context.start_token)
            token_vocab.add(context.end_token)
            for node in context.ast_path:
                ast_node_vocab.add(node)
    return token_vocab, ast_node_vocab, label_vocab


def build_multilabel_vocabularies(
    records: Iterable[RawCode2VecRecord],
) -> tuple[Vocabulary, Vocabulary, Vocabulary]:
    token_vocab = Vocabulary()
    ast_node_vocab = Vocabulary()
    target_vocab = Vocabulary(special_tokens=False)
    for record in records:
        for subtoken in split_label_subtokens(record.label):
            target_vocab.add(subtoken)
        for context in record.contexts:
            token_vocab.add(context.start_token)
            token_vocab.add(context.end_token)
            for node in context.ast_path:
                ast_node_vocab.add(node)
    return token_vocab, ast_node_vocab, target_vocab


def filter_records_by_known_label_subtokens(
    records: Iterable[RawCode2VecRecord],
    target_vocab: Vocabulary,
) -> list[RawCode2VecRecord]:
    return [
        record
        for record in records
        if all(target_vocab.contains(subtoken) for subtoken in split_label_subtokens(record.label))
    ]


def _attach_tree_features(batch: Code2HypBatch) -> Code2HypBatch:
    return Code2HypBatch(
        start_tokens=batch.start_tokens,
        end_tokens=batch.end_tokens,
        ast_paths=batch.ast_paths,
        ast_path_mask=batch.ast_path_mask,
        context_mask=batch.context_mask,
        context_tree_features=tree_context_features(batch),
    )


def encode_records_to_batch(
    records: list[RawCode2VecRecord],
    max_contexts: int,
    max_path_length: int,
    token_dim: int = 32,
    structural_dim: int = 32,
    curvature: float = 1.0,
    path_encoder: str = "mean",
    representation_transform: str = "identity",
) -> EncodedCode2HypDataset:
    if not records:
        raise ValueError("records must not be empty")
    if max_contexts <= 0:
        raise ValueError("max_contexts must be positive")
    if max_path_length <= 0:
        raise ValueError("max_path_length must be positive")

    token_vocab, ast_node_vocab, label_vocab = build_vocabularies(records)
    examples = len(records)
    start_tokens = torch.zeros(examples, max_contexts, dtype=torch.long)
    end_tokens = torch.zeros(examples, max_contexts, dtype=torch.long)
    ast_paths = torch.zeros(examples, max_contexts, max_path_length, dtype=torch.long)
    ast_path_mask = torch.zeros(examples, max_contexts, max_path_length, dtype=torch.bool)
    context_mask = torch.zeros(examples, max_contexts, dtype=torch.bool)
    labels = torch.zeros(examples, dtype=torch.long)

    for record_index, record in enumerate(records):
        labels[record_index] = label_vocab.lookup(record.label)
        for context_index, context in enumerate(record.contexts[:max_contexts]):
            context_mask[record_index, context_index] = True
            start_tokens[record_index, context_index] = token_vocab.lookup(context.start_token)
            end_tokens[record_index, context_index] = token_vocab.lookup(context.end_token)
            for path_index, node in enumerate(context.ast_path[:max_path_length]):
                ast_paths[record_index, context_index, path_index] = ast_node_vocab.lookup(node)
                ast_path_mask[record_index, context_index, path_index] = True

    model_config = Code2HypTorchConfig(
        token_vocab_size=len(token_vocab),
        ast_node_vocab_size=len(ast_node_vocab),
        label_vocab_size=len(label_vocab),
        token_dim=token_dim,
        structural_dim=structural_dim,
        curvature=curvature,
        path_encoder=path_encoder,  # type: ignore[arg-type]
        representation_transform=representation_transform,  # type: ignore[arg-type]
    )
    batch = Code2HypBatch(
        start_tokens=start_tokens,
        end_tokens=end_tokens,
        ast_paths=ast_paths,
        ast_path_mask=ast_path_mask,
        context_mask=context_mask,
    )
    return EncodedCode2HypDataset(
        batch=_attach_tree_features(batch),
        labels=labels,
        model_config=model_config,
        token_vocab=token_vocab,
        ast_node_vocab=ast_node_vocab,
        label_vocab=label_vocab,
    )


def encode_records_to_multilabel_batch(
    records: list[RawCode2VecRecord],
    max_contexts: int,
    max_path_length: int,
    token_dim: int = 32,
    structural_dim: int = 32,
    curvature: float = 1.0,
    path_encoder: str = "mean",
    representation_transform: str = "identity",
    token_vocab: Vocabulary | None = None,
    ast_node_vocab: Vocabulary | None = None,
    target_vocab: Vocabulary | None = None,
) -> EncodedCode2HypMultiLabelDataset:
    if not records:
        raise ValueError("records must not be empty")
    if max_contexts <= 0:
        raise ValueError("max_contexts must be positive")
    if max_path_length <= 0:
        raise ValueError("max_path_length must be positive")

    if token_vocab is None or ast_node_vocab is None or target_vocab is None:
        token_vocab, ast_node_vocab, target_vocab = build_multilabel_vocabularies(records)

    examples = len(records)
    start_tokens = torch.zeros(examples, max_contexts, dtype=torch.long)
    end_tokens = torch.zeros(examples, max_contexts, dtype=torch.long)
    ast_paths = torch.zeros(examples, max_contexts, max_path_length, dtype=torch.long)
    ast_path_mask = torch.zeros(examples, max_contexts, max_path_length, dtype=torch.bool)
    context_mask = torch.zeros(examples, max_contexts, dtype=torch.bool)
    labels = torch.zeros(examples, len(target_vocab), dtype=torch.float32)
    target_sizes = torch.zeros(examples, dtype=torch.long)

    for record_index, record in enumerate(records):
        subtokens = split_label_subtokens(record.label)
        target_sizes[record_index] = len(subtokens)
        for subtoken in subtokens:
            labels[record_index, target_vocab.lookup(subtoken)] = 1.0
        for context_index, context in enumerate(record.contexts[:max_contexts]):
            context_mask[record_index, context_index] = True
            start_tokens[record_index, context_index] = token_vocab.lookup(context.start_token)
            end_tokens[record_index, context_index] = token_vocab.lookup(context.end_token)
            for path_index, node in enumerate(context.ast_path[:max_path_length]):
                ast_paths[record_index, context_index, path_index] = ast_node_vocab.lookup(node)
                ast_path_mask[record_index, context_index, path_index] = True

    model_config = Code2HypTorchConfig(
        token_vocab_size=len(token_vocab),
        ast_node_vocab_size=len(ast_node_vocab),
        label_vocab_size=len(target_vocab),
        token_dim=token_dim,
        structural_dim=structural_dim,
        curvature=curvature,
        path_encoder=path_encoder,  # type: ignore[arg-type]
        representation_transform=representation_transform,  # type: ignore[arg-type]
    )
    batch = Code2HypBatch(
        start_tokens=start_tokens,
        end_tokens=end_tokens,
        ast_paths=ast_paths,
        ast_path_mask=ast_path_mask,
        context_mask=context_mask,
    )
    return EncodedCode2HypMultiLabelDataset(
        batch=_attach_tree_features(batch),
        labels=labels,
        target_sizes=target_sizes,
        model_config=model_config,
        token_vocab=token_vocab,
        ast_node_vocab=ast_node_vocab,
        target_vocab=target_vocab,
    )
