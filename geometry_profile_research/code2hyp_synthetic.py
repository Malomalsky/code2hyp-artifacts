from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .code2hyp_torch import Code2HypBatch, Code2HypTorchConfig


@dataclass(frozen=True)
class SyntheticCode2HypConfig:
    examples: int = 128
    contexts_per_method: int = 6
    max_path_length: int = 5
    branches: int = 4
    token_vocab_size: int = 16
    token_dim: int = 8
    structural_dim: int = 8
    seed: int = 13


@dataclass(frozen=True)
class SyntheticCode2HypDataset:
    batch: Code2HypBatch
    labels: Tensor
    model_config: Code2HypTorchConfig
    description: str


def make_synthetic_code2hyp_dataset(config: SyntheticCode2HypConfig) -> SyntheticCode2HypDataset:
    """Create a controlled AST-path method-label dataset.

    The label is determined by the top AST branch node. Lexical tokens are
    sampled independently, so the task is intentionally structure-driven.
    """
    if config.examples <= 0:
        raise ValueError("examples must be positive")
    if config.contexts_per_method <= 0:
        raise ValueError("contexts_per_method must be positive")
    if config.max_path_length < 2:
        raise ValueError("max_path_length must be at least 2")
    if config.branches <= 1:
        raise ValueError("branches must be greater than one")
    if config.token_vocab_size <= 1:
        raise ValueError("token_vocab_size must be greater than one")

    generator = torch.Generator().manual_seed(config.seed)
    labels = torch.randint(0, config.branches, (config.examples,), generator=generator)
    start_tokens = torch.randint(
        1,
        config.token_vocab_size,
        (config.examples, config.contexts_per_method),
        generator=generator,
    )
    end_tokens = torch.randint(
        1,
        config.token_vocab_size,
        (config.examples, config.contexts_per_method),
        generator=generator,
    )

    root_node = 1
    branch_node_offset = 2
    detail_node_offset = branch_node_offset + config.branches
    detail_vocab_size = config.branches * max(config.max_path_length, 3) + config.contexts_per_method + 4
    ast_node_vocab_size = detail_node_offset + detail_vocab_size

    ast_paths = torch.zeros(
        config.examples,
        config.contexts_per_method,
        config.max_path_length,
        dtype=torch.long,
    )
    ast_path_mask = torch.zeros_like(ast_paths, dtype=torch.bool)

    for example_index, label in enumerate(labels.tolist()):
        branch_node = branch_node_offset + label
        for context_index in range(config.contexts_per_method):
            path_length = int(torch.randint(2, config.max_path_length + 1, (1,), generator=generator).item())
            ast_paths[example_index, context_index, 0] = root_node
            ast_paths[example_index, context_index, 1] = branch_node
            ast_path_mask[example_index, context_index, :path_length] = True

            for depth in range(2, path_length):
                detail_bucket = (label * config.max_path_length + context_index + depth) % detail_vocab_size
                ast_paths[example_index, context_index, depth] = detail_node_offset + detail_bucket

    context_mask = torch.ones(config.examples, config.contexts_per_method, dtype=torch.bool)
    model_config = Code2HypTorchConfig(
        token_vocab_size=config.token_vocab_size,
        ast_node_vocab_size=ast_node_vocab_size,
        label_vocab_size=config.branches,
        token_dim=config.token_dim,
        structural_dim=config.structural_dim,
    )

    return SyntheticCode2HypDataset(
        batch=Code2HypBatch(
            start_tokens=start_tokens,
            end_tokens=end_tokens,
            ast_paths=ast_paths,
            ast_path_mask=ast_path_mask,
            context_mask=context_mask,
        ),
        labels=labels,
        model_config=model_config,
        description=(
            "Synthetic structure-driven AST-path classification: labels are "
            "encoded by the first branch node below root; lexical tokens are noise."
        ),
    )
