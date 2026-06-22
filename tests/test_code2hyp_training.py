from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from geometry_profile_research.code2hyp_data import (
    encode_records_to_multilabel_batch,
    parse_code2vec_line,
)
from geometry_profile_research.code2hyp_synthetic import (
    SyntheticCode2HypConfig,
    make_synthetic_code2hyp_dataset,
)
from geometry_profile_research.code2hyp_training import (
    ADAPTIVE_RANK_MAX_WEIGHT,
    ADAPTIVE_RANK_MIN_WEIGHT,
    _structural_regularizer_loss,
    compute_multilabel_pos_weight,
    evaluate_accuracy,
    evaluate_multilabel_metrics,
    fit_multilabel_supervised,
    fit_supervised,
    make_minibatches,
    multilabel_metrics_from_logits,
    scheduled_structural_loss_weight,
    slice_batch,
    train_step,
    train_multilabel_step,
)
from geometry_profile_research.code2hyp_torch import Code2HypBatch, Code2HypTorchConfig, Code2HypTorchModel


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.offset = 0

    def forward(self, batch: Code2HypBatch):  # type: ignore[override]
        batch_size = int(batch.start_tokens.shape[0])
        start = self.offset
        end = start + batch_size
        self.offset = end
        return SimpleNamespace(logits=self.logits[start:end])


def _empty_batch(examples: int, contexts: int = 1, path_length: int = 1) -> Code2HypBatch:
    return Code2HypBatch(
        start_tokens=torch.zeros(examples, contexts, dtype=torch.long),
        end_tokens=torch.zeros(examples, contexts, dtype=torch.long),
        ast_paths=torch.zeros(examples, contexts, path_length, dtype=torch.long),
        ast_path_mask=torch.ones(examples, contexts, path_length, dtype=torch.bool),
        context_mask=torch.ones(examples, contexts, dtype=torch.bool),
    )


class Code2HypSyntheticTrainingTests(unittest.TestCase):
    def test_synthetic_dataset_has_expected_shapes_and_structural_labels(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(
            SyntheticCode2HypConfig(
                examples=24,
                contexts_per_method=4,
                max_path_length=5,
                branches=3,
                seed=5,
            )
        )

        self.assertEqual(dataset.batch.start_tokens.shape, (24, 4))
        self.assertEqual(dataset.batch.ast_paths.shape, (24, 4, 5))
        self.assertEqual(dataset.labels.shape, (24,))
        self.assertEqual(dataset.model_config.label_vocab_size, 3)
        self.assertTrue(torch.all(dataset.labels >= 0))
        self.assertTrue(torch.all(dataset.labels < 3))
        self.assertTrue(torch.all(dataset.batch.context_mask.sum(dim=1) == 4))

    def test_make_minibatches_preserves_all_examples(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=23, seed=7))

        minibatches = list(make_minibatches(dataset.batch, dataset.labels, batch_size=5, shuffle=False))

        self.assertEqual(sum(len(labels) for _, labels in minibatches), 23)
        torch.testing.assert_close(torch.cat([labels for _, labels in minibatches]), dataset.labels)

    def test_slice_batch_preserves_precomputed_tree_features(self) -> None:
        cached_features = torch.arange(4 * 2 * 4, dtype=torch.float32).reshape(4, 2, 4)
        batch = Code2HypBatch(
            start_tokens=torch.zeros(4, 2, dtype=torch.long),
            end_tokens=torch.zeros(4, 2, dtype=torch.long),
            ast_paths=torch.zeros(4, 2, 3, dtype=torch.long),
            ast_path_mask=torch.ones(4, 2, 3, dtype=torch.bool),
            context_mask=torch.ones(4, 2, dtype=torch.bool),
            context_tree_features=cached_features,
        )

        sliced = slice_batch(batch, torch.tensor([2, 0], dtype=torch.long))

        self.assertIsNotNone(sliced.context_tree_features)
        torch.testing.assert_close(sliced.context_tree_features, cached_features[[2, 0]])

    def test_single_train_step_updates_parameters_and_returns_metrics(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=11))
        model = Code2HypTorchModel(dataset.model_config, variant="product")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        before = model.decoder.weight.detach().clone()

        metrics = train_step(model, optimizer, dataset.batch, dataset.labels, structural_loss_weight=0.25)

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreater(metrics["task_loss"], 0.0)
        self.assertGreaterEqual(metrics["structural_loss"], 0.0)
        self.assertGreaterEqual(metrics["loss"], metrics["task_loss"])
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertFalse(torch.allclose(before, model.decoder.weight.detach()))

    def test_fit_supervised_learns_structural_synthetic_task(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(
            SyntheticCode2HypConfig(
                examples=96,
                contexts_per_method=5,
                max_path_length=5,
                branches=4,
                seed=13,
            )
        )
        model = Code2HypTorchModel(dataset.model_config, variant="product")

        history = fit_supervised(
            model,
            dataset.batch,
            dataset.labels,
            epochs=35,
            batch_size=24,
            learning_rate=0.05,
            structural_loss_weight=0.0,
            seed=19,
        )
        final_accuracy = evaluate_accuracy(model, dataset.batch, dataset.labels, batch_size=32)

        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreaterEqual(final_accuracy, 0.90)

    def test_product_model_with_trainable_curvature_learns_and_keeps_curvature_positive(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=64, branches=4, seed=23))
        config = Code2HypTorchConfig(
            token_vocab_size=dataset.model_config.token_vocab_size,
            ast_node_vocab_size=dataset.model_config.ast_node_vocab_size,
            label_vocab_size=dataset.model_config.label_vocab_size,
            token_dim=dataset.model_config.token_dim,
            structural_dim=dataset.model_config.structural_dim,
            trainable_curvature=True,
        )
        model = Code2HypTorchModel(config, variant="product")

        fit_supervised(
            model,
            dataset.batch,
            dataset.labels,
            epochs=20,
            batch_size=16,
            learning_rate=0.04,
            structural_loss_weight=0.1,
            seed=29,
        )
        output = model(dataset.batch)

        self.assertGreater(float(output.curvature.detach()), 0.0)
        self.assertTrue(torch.isfinite(model.raw_curvature.detach()))

    def test_multilabel_metrics_use_target_subtoken_count_as_top_k(self) -> None:
        logits = torch.tensor([[5.0, 4.0, 0.0], [0.0, 5.0, 4.0]])
        labels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        target_sizes = torch.tensor([2, 2])

        metrics = multilabel_metrics_from_logits(logits, labels, target_sizes)

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["predicted_positive_count_mean"], 2.0)

    def test_multilabel_metrics_support_non_oracle_selection_protocols(self) -> None:
        logits = torch.tensor([[5.0, 4.0, 0.0], [0.0, 5.0, 4.0]])
        labels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        target_sizes = torch.tensor([2, 2])

        fixed_top1 = multilabel_metrics_from_logits(logits, labels, target_sizes, selection="fixed_topk", fixed_k=1)
        threshold = multilabel_metrics_from_logits(logits, labels, target_sizes, selection="threshold", threshold=4.5)

        self.assertEqual(fixed_top1["precision"], 1.0)
        self.assertEqual(fixed_top1["recall"], 0.5)
        self.assertEqual(fixed_top1["predicted_positive_count_mean"], 1.0)
        self.assertEqual(threshold["precision"], 1.0)
        self.assertEqual(threshold["recall"], 0.5)
        self.assertEqual(threshold["predicted_positive_count_mean"], 1.0)

    def test_evaluate_multilabel_metrics_keeps_target_sizes_aligned_across_minibatches(self) -> None:
        labels = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],
                [10.0, 9.0, 0.0, 0.0],
                [10.0, 9.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 10.0],
            ]
        )
        target_sizes = torch.tensor([1, 2, 3, 1])
        model = _FixedLogitModel(logits)

        metrics = evaluate_multilabel_metrics(
            model, _empty_batch(examples=4), labels, target_sizes, batch_size=2
        )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_multilabel_positive_weights_counter_class_imbalance(self) -> None:
        labels = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        pos_weight = compute_multilabel_pos_weight(labels, max_weight=10.0)

        torch.testing.assert_close(pos_weight, torch.tensor([3.0, 3.0, 1.0]))

    def test_multilabel_train_step_updates_model_on_code2seq_targets(self) -> None:
        dataset = encode_records_to_multilabel_batch(
            [
                parse_code2vec_line("to|lower|case obj,Name|Call,value"),
                parse_code2vec_line("to|string obj,Name|Call,value"),
                parse_code2vec_line("hash|code self,Name|Call,result"),
            ],
            max_contexts=1,
            max_path_length=2,
            token_dim=8,
            structural_dim=8,
        )
        model = Code2HypTorchModel(dataset.model_config, variant="product")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        before = model.decoder.weight.detach().clone()
        pos_weight = compute_multilabel_pos_weight(dataset.labels, max_weight=5.0)

        metrics = train_multilabel_step(
            model,
            optimizer,
            dataset.batch,
            dataset.labels,
            dataset.target_sizes,
            structural_loss_weight=0.1,
            pos_weight=pos_weight,
        )
        eval_metrics = evaluate_multilabel_metrics(
            model,
            dataset.batch,
            dataset.labels,
            dataset.target_sizes,
            batch_size=2,
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreaterEqual(eval_metrics["f1"], 0.0)
        self.assertLessEqual(eval_metrics["f1"], 1.0)
        self.assertFalse(torch.allclose(before, model.decoder.weight.detach()))

    def test_multilabel_train_step_accepts_rank_structural_regularizer(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=41))
        model = Code2HypTorchModel(dataset.model_config, variant="product")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

        metrics = train_multilabel_step(
            model,
            optimizer,
            dataset.batch,
            torch.nn.functional.one_hot(dataset.labels, num_classes=dataset.model_config.label_vocab_size).float(),
            torch.ones_like(dataset.labels),
            structural_loss_weight=0.1,
            structural_regularizer="rank",
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["structural_loss"], 0.0)

    def test_multilabel_train_step_accepts_neighbor_distribution_regularizer(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=50))
        model = Code2HypTorchModel(dataset.model_config, variant="product")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

        metrics = train_multilabel_step(
            model,
            optimizer,
            dataset.batch,
            torch.nn.functional.one_hot(dataset.labels, num_classes=dataset.model_config.label_vocab_size).float(),
            torch.ones_like(dataset.labels),
            structural_loss_weight=0.1,
            structural_regularizer="neighbor_distribution",
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["structural_loss"], 0.0)

    def test_multilabel_train_step_accepts_path_attention_monotone_regularizer(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=42))
        model = Code2HypTorchModel(dataset.model_config, variant="hyperbolic_path_attention_message_passing")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

        metrics = train_multilabel_step(
            model,
            optimizer,
            dataset.batch,
            torch.nn.functional.one_hot(dataset.labels, num_classes=dataset.model_config.label_vocab_size).float(),
            torch.ones_like(dataset.labels),
            structural_loss_weight=0.1,
            structural_regularizer="path_attention_monotone",
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["structural_loss"], 0.0)

    def test_multilabel_train_step_accepts_path_attention_tree_distance_regularizer(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=44))
        model = Code2HypTorchModel(dataset.model_config, variant="hyperbolic_path_attention_message_passing")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

        metrics = train_multilabel_step(
            model,
            optimizer,
            dataset.batch,
            torch.nn.functional.one_hot(dataset.labels, num_classes=dataset.model_config.label_vocab_size).float(),
            torch.ones_like(dataset.labels),
            structural_loss_weight=0.1,
            structural_regularizer="path_attention_tree_distance",
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["structural_loss"], 0.0)

    def test_multilabel_train_step_accepts_path_dual_attention_separation_regularizer(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=45))
        model = Code2HypTorchModel(dataset.model_config, variant="hyperbolic_path_dual_attention_message_passing")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

        metrics = train_multilabel_step(
            model,
            optimizer,
            dataset.batch,
            torch.nn.functional.one_hot(dataset.labels, num_classes=dataset.model_config.label_vocab_size).float(),
            torch.ones_like(dataset.labels),
            structural_loss_weight=0.1,
            structural_regularizer="path_dual_attention_separation",
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["structural_loss"], 0.0)

    def test_multilabel_train_step_accepts_path_dual_attention_separation_rank_regularizer(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=46))
        model = Code2HypTorchModel(dataset.model_config, variant="hyperbolic_path_dual_attention_message_passing")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

        metrics = train_multilabel_step(
            model,
            optimizer,
            dataset.batch,
            torch.nn.functional.one_hot(dataset.labels, num_classes=dataset.model_config.label_vocab_size).float(),
            torch.ones_like(dataset.labels),
            structural_loss_weight=0.1,
            structural_regularizer="path_dual_attention_separation_rank",
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["structural_loss"], 0.0)

    def test_dual_attention_soft_rank_regularizer_uses_fractional_rank_weight(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=16, seed=48))
        model = Code2HypTorchModel(dataset.model_config, variant="hyperbolic_path_dual_attention_message_passing")
        output = model(dataset.batch)

        separation_loss = _structural_regularizer_loss(
            output,
            dataset.batch,
            "path_dual_attention_separation",
        )
        full_rank_loss = _structural_regularizer_loss(
            output,
            dataset.batch,
            "path_dual_attention_separation_rank",
        )
        soft_rank_loss = _structural_regularizer_loss(
            output,
            dataset.batch,
            "path_dual_attention_separation_soft_rank",
        )

        expected_soft_loss = separation_loss + 0.25 * (full_rank_loss - separation_loss)
        torch.testing.assert_close(soft_rank_loss, expected_soft_loss)

    def test_dual_attention_adaptive_rank_regularizer_keeps_nonzero_rank_floor(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=16, seed=49))
        model = Code2HypTorchModel(dataset.model_config, variant="hyperbolic_path_dual_attention_message_passing")
        output = model(dataset.batch)

        separation_loss = _structural_regularizer_loss(
            output,
            dataset.batch,
            "path_dual_attention_separation",
        )
        full_rank_loss = _structural_regularizer_loss(
            output,
            dataset.batch,
            "path_dual_attention_separation_rank",
        )
        adaptive_rank_loss = _structural_regularizer_loss(
            output,
            dataset.batch,
            "path_dual_attention_separation_adaptive_rank",
        )

        rank_loss = full_rank_loss - separation_loss
        adaptive_weight = torch.clamp(
            separation_loss.detach() / torch.clamp(
                separation_loss.detach() + rank_loss.detach(),
                min=1e-12,
            ),
            min=ADAPTIVE_RANK_MIN_WEIGHT,
            max=ADAPTIVE_RANK_MAX_WEIGHT,
        )
        expected_adaptive_loss = separation_loss + adaptive_weight * rank_loss
        self.assertGreaterEqual(float(adaptive_weight.detach()), ADAPTIVE_RANK_MIN_WEIGHT)
        self.assertLessEqual(float(adaptive_weight.detach()), ADAPTIVE_RANK_MAX_WEIGHT)
        self.assertGreater(float(adaptive_rank_loss.detach()), float(separation_loss.detach()))
        torch.testing.assert_close(adaptive_rank_loss, expected_adaptive_loss)

    def test_dual_attention_adaptive_rank_regularizer_caps_dominant_rank_weight(self) -> None:
        separation_loss = torch.tensor(10.0)
        rank_loss = torch.tensor(1.0)

        adaptive_weight = torch.clamp(
            separation_loss.detach() / torch.clamp(
                separation_loss.detach() + rank_loss.detach(),
                min=1e-12,
            ),
            min=ADAPTIVE_RANK_MIN_WEIGHT,
            max=ADAPTIVE_RANK_MAX_WEIGHT,
        )

        self.assertAlmostEqual(float(adaptive_weight), ADAPTIVE_RANK_MAX_WEIGHT)

    def test_dual_attention_adaptive_rank_regularizer_floors_vanishing_separation_weight(self) -> None:
        separation_loss = torch.tensor(0.0)
        rank_loss = torch.tensor(1.0)

        adaptive_weight = torch.clamp(
            separation_loss.detach() / torch.clamp(
                separation_loss.detach() + rank_loss.detach(),
                min=1e-12,
            ),
            min=ADAPTIVE_RANK_MIN_WEIGHT,
            max=ADAPTIVE_RANK_MAX_WEIGHT,
        )

        self.assertAlmostEqual(float(adaptive_weight), ADAPTIVE_RANK_MIN_WEIGHT)

    def test_fit_multilabel_supervised_can_anneal_structural_loss_weight(self) -> None:
        dataset = make_synthetic_code2hyp_dataset(SyntheticCode2HypConfig(examples=32, seed=43))
        labels = torch.nn.functional.one_hot(
            dataset.labels,
            num_classes=dataset.model_config.label_vocab_size,
        ).float()
        target_sizes = torch.ones_like(dataset.labels)
        model = Code2HypTorchModel(dataset.model_config, variant="product")

        history = fit_multilabel_supervised(
            model,
            dataset.batch,
            labels,
            target_sizes,
            epochs=3,
            batch_size=16,
            learning_rate=0.03,
            structural_loss_weight=0.3,
            structural_loss_schedule="linear",
            structural_regularizer="rank",
            seed=47,
        )

        weights = [round(row["structural_loss_weight"], 4) for row in history]
        self.assertEqual(weights, [0.1, 0.2, 0.3])

    def test_delayed_linear_structural_loss_schedule_keeps_first_epoch_unregularized(self) -> None:
        weights = [
            scheduled_structural_loss_weight(
                base_weight=0.3,
                epoch_index=epoch_index,
                epochs=3,
                schedule="delayed_linear",
            )
            for epoch_index in range(3)
        ]

        self.assertEqual([round(weight, 4) for weight in weights], [0.0, 0.15, 0.3])

    def test_cosine_structural_loss_schedule_smoothly_reaches_base_weight(self) -> None:
        weights = [
            scheduled_structural_loss_weight(
                base_weight=0.3,
                epoch_index=epoch_index,
                epochs=5,
                schedule="cosine",
            )
            for epoch_index in range(5)
        ]

        self.assertEqual([round(weight, 4) for weight in weights], [0.0, 0.0439, 0.15, 0.2561, 0.3])

    def test_warmup_decay_structural_loss_schedule_peaks_then_relaxes(self) -> None:
        weights = [
            scheduled_structural_loss_weight(
                base_weight=0.3,
                epoch_index=epoch_index,
                epochs=5,
                schedule="warmup_decay",
            )
            for epoch_index in range(5)
        ]

        self.assertEqual([round(weight, 4) for weight in weights], [0.0, 0.15, 0.3, 0.15, 0.0])


if __name__ == "__main__":
    unittest.main()
