from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from geometry_profile_research.code2hyp_data import encode_records_to_multilabel_batch, parse_code2vec_line
from geometry_profile_research.code2hyp_supervised import (
    Code2HypSupervisedConfig,
    Code2VecCompatibleCode2Hyp,
    train_supervised_c2s_variants,
)


class Code2HypSupervisedTests(unittest.TestCase):
    def test_forward_returns_logits_and_attention_over_valid_contexts(self) -> None:
        records = [
            parse_code2vec_line("to|lower obj,Name|Call,value this,Name|Return,out"),
            parse_code2vec_line("to|string obj,Name|Member,value"),
        ]
        encoded = encode_records_to_multilabel_batch(records, max_contexts=3, max_path_length=4)
        model = Code2VecCompatibleCode2Hyp(
            Code2HypSupervisedConfig(
                token_vocab_size=len(encoded.token_vocab),
                ast_node_vocab_size=len(encoded.ast_node_vocab),
                target_vocab_size=len(encoded.target_vocab),
                token_dim=8,
                structural_dim=8,
                geometry="poincare",
                curvature=1.0,
            )
        )

        output = model(encoded.batch)

        self.assertEqual(output.logits.shape, encoded.labels.shape)
        self.assertEqual(output.attention.shape, encoded.batch.context_mask.shape)
        torch.testing.assert_close(
            output.attention.sum(dim=1),
            torch.ones(output.attention.shape[0]),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(torch.all(output.attention[~encoded.batch.context_mask] == 0.0))

    def test_poincare_and_euclidean_controls_have_same_parameter_count(self) -> None:
        common = {
            "token_vocab_size": 11,
            "ast_node_vocab_size": 13,
            "target_vocab_size": 7,
            "token_dim": 8,
            "structural_dim": 8,
            "curvature": 1.0,
        }
        euclidean = Code2VecCompatibleCode2Hyp(Code2HypSupervisedConfig(**common, geometry="euclidean"))
        poincare = Code2VecCompatibleCode2Hyp(Code2HypSupervisedConfig(**common, geometry="poincare"))

        euclidean_params = sum(parameter.numel() for parameter in euclidean.parameters())
        poincare_params = sum(parameter.numel() for parameter in poincare.parameters())

        self.assertEqual(euclidean_params, poincare_params)

    def test_train_supervised_c2s_variants_writes_reproducible_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train.c2s"
            val = root / "val.c2s"
            train.write_text(
                "\n".join(
                    [
                        "to|lower obj,Name|Call,value this,Name|Return,out",
                        "to|string obj,Name|Member,value",
                        "get|name this,Name|Call,name",
                    ]
                ),
                encoding="utf-8",
            )
            val.write_text(
                "\n".join(
                    [
                        "to|lower obj,Name|Call,value",
                        "get|name this,Name|Call,name",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "result.json"

            payload = train_supervised_c2s_variants(
                train_path=train,
                validation_path=val,
                output_path=output,
                variants=("euclidean", "poincare_near_zero"),
                train_limit=3,
                validation_limit=2,
                max_contexts=3,
                max_path_length=4,
                token_dim=8,
                structural_dim=8,
                epochs=1,
                batch_size=2,
                learning_rate=0.01,
                seed=123,
            )

            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"], "code2vec_compatible_code2hyp_supervised")
        self.assertEqual(saved["config"]["train_limit"], 3)
        self.assertEqual([run["variant"] for run in saved["runs"]], ["euclidean", "poincare_near_zero"])
        self.assertEqual(saved["runs"][0]["parameter_count"], saved["runs"][1]["parameter_count"])
        self.assertIn("validation_f1", saved["runs"][0])
        self.assertIn("validation_precision", saved["runs"][0])
        self.assertIn("validation_recall", saved["runs"][0])


if __name__ == "__main__":
    unittest.main()
