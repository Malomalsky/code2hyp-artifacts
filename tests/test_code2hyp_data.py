from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from geometry_profile_research.code2hyp_data import (
    Vocabulary,
    apply_lexical_ablation,
    build_multilabel_vocabularies,
    encode_records_to_batch,
    encode_records_to_multilabel_batch,
    filter_records_by_known_label_subtokens,
    load_code2vec_records,
    parse_code2vec_line,
    split_label_subtokens,
)
from geometry_profile_research.code2hyp_torch import Code2HypBatch, tree_context_features


class Code2HypDataTests(unittest.TestCase):
    def test_parse_code2vec_line_extracts_label_and_path_contexts(self) -> None:
        line = "get_value obj,Name|Member|Return,value this,Name|Call,get"

        record = parse_code2vec_line(line)

        self.assertEqual(record.label, "get_value")
        self.assertEqual(len(record.contexts), 2)
        self.assertEqual(record.contexts[0].start_token, "obj")
        self.assertEqual(record.contexts[0].ast_path, ("Name", "Member", "Return"))
        self.assertEqual(record.contexts[0].end_token, "value")

    def test_vocabulary_uses_zero_for_padding_and_one_for_unknown(self) -> None:
        vocab = Vocabulary()

        first = vocab.add("alpha")
        second = vocab.add("beta")

        self.assertEqual(vocab.pad_id, 0)
        self.assertEqual(vocab.unk_id, 1)
        self.assertEqual(first, 2)
        self.assertEqual(second, 3)
        self.assertEqual(vocab.lookup("missing"), 1)
        self.assertEqual(vocab.lookup("alpha"), 2)

    def test_encode_records_to_batch_limits_contexts_and_paths(self) -> None:
        records = [
            parse_code2vec_line("get_value obj,Name|Member|Return,value this,Name|Call,get"),
            parse_code2vec_line("set_value obj,Name|Assign|Return,value"),
        ]

        encoded = encode_records_to_batch(records, max_contexts=2, max_path_length=2)

        self.assertEqual(encoded.batch.start_tokens.shape, (2, 2))
        self.assertEqual(encoded.batch.ast_paths.shape, (2, 2, 2))
        self.assertEqual(encoded.labels.shape, (2,))
        self.assertEqual(encoded.model_config.label_vocab_size, 2)
        self.assertTrue(torch.equal(encoded.batch.context_mask[0], torch.tensor([True, True])))
        self.assertTrue(torch.equal(encoded.batch.context_mask[1], torch.tensor([True, False])))
        self.assertTrue(torch.equal(encoded.batch.ast_path_mask[0, 0], torch.tensor([True, True])))

    def test_encode_records_to_batch_precomputes_tree_features(self) -> None:
        records = [
            parse_code2vec_line("get_value obj,Name|Member|Return,value this,Name|Call,get"),
            parse_code2vec_line("set_value obj,Name|Assign|Return,value"),
        ]

        encoded = encode_records_to_batch(records, max_contexts=2, max_path_length=3)
        uncached_batch = Code2HypBatch(
            start_tokens=encoded.batch.start_tokens,
            end_tokens=encoded.batch.end_tokens,
            ast_paths=encoded.batch.ast_paths,
            ast_path_mask=encoded.batch.ast_path_mask,
            context_mask=encoded.batch.context_mask,
        )

        self.assertIsNotNone(encoded.batch.context_tree_features)
        self.assertEqual(encoded.batch.context_tree_features.shape, (2, 2, 4))
        torch.testing.assert_close(encoded.batch.context_tree_features, tree_context_features(uncached_batch))
        torch.testing.assert_close(encoded.batch.context_tree_features[1, 1], torch.zeros(4))

    def test_load_code2vec_records_reads_real_preprocessed_format_with_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.c2v"
            path.write_text(
                "\n".join(
                    [
                        "to|lower|case value,Name|MethodInvocation|Name,result",
                        "equals left,Name|BinaryExpr|Name,right",
                        "ignored start,Name,end",
                    ]
                ),
                encoding="utf-8",
            )

            records = load_code2vec_records(path, limit=2)

        self.assertEqual([record.label for record in records], ["to|lower|case", "equals"])
        self.assertEqual(records[0].contexts[0].ast_path, ("Name", "MethodInvocation", "Name"))

    def test_split_label_subtokens_uses_code2seq_pipe_convention(self) -> None:
        self.assertEqual(split_label_subtokens("to|lower|case"), ("to", "lower", "case"))
        self.assertEqual(split_label_subtokens("equals"), ("equals",))

    def test_encode_records_to_multilabel_batch_predicts_target_subtokens(self) -> None:
        records = [
            parse_code2vec_line("to|lower|case obj,Name|Call,value"),
            parse_code2vec_line("to|string obj,Name|Call,value"),
        ]

        encoded = encode_records_to_multilabel_batch(records, max_contexts=1, max_path_length=2)

        self.assertEqual(encoded.labels.shape, (2, 4))
        self.assertEqual(encoded.model_config.label_vocab_size, 4)
        self.assertEqual(encoded.target_sizes.tolist(), [3, 2])
        self.assertEqual(encoded.target_vocab.token(int(torch.argmax(encoded.labels[0]))), "to")

    def test_encode_records_to_multilabel_batch_precomputes_tree_features(self) -> None:
        records = [
            parse_code2vec_line("to|lower|case obj,Name|Call,value"),
            parse_code2vec_line("to|string obj,Name|Member|Return,value this,Name|Call,get"),
        ]

        encoded = encode_records_to_multilabel_batch(records, max_contexts=2, max_path_length=3)
        uncached_batch = Code2HypBatch(
            start_tokens=encoded.batch.start_tokens,
            end_tokens=encoded.batch.end_tokens,
            ast_paths=encoded.batch.ast_paths,
            ast_path_mask=encoded.batch.ast_path_mask,
            context_mask=encoded.batch.context_mask,
        )

        self.assertIsNotNone(encoded.batch.context_tree_features)
        self.assertEqual(encoded.batch.context_tree_features.shape, (2, 2, 4))
        torch.testing.assert_close(encoded.batch.context_tree_features, tree_context_features(uncached_batch))

    def test_encode_records_to_multilabel_batch_preserves_curvature_config(self) -> None:
        records = [
            parse_code2vec_line("to|lower|case obj,Name|Call,value"),
            parse_code2vec_line("to|string obj,Name|Call,value"),
        ]

        encoded = encode_records_to_multilabel_batch(
            records,
            max_contexts=1,
            max_path_length=2,
            curvature=0.3,
        )

        self.assertEqual(encoded.model_config.curvature, 0.3)

    def test_filter_records_by_known_label_subtokens_for_validation_split(self) -> None:
        train_records = [
            parse_code2vec_line("to|lower obj,Name|Call,value"),
        ]
        val_records = [
            parse_code2vec_line("to|lower obj,Name|Call,value"),
            parse_code2vec_line("unknown|lower obj,Name|Call,value"),
        ]
        _, _, target_vocab = build_multilabel_vocabularies(train_records)

        filtered = filter_records_by_known_label_subtokens(val_records, target_vocab)

        self.assertEqual([record.label for record in filtered], ["to|lower"])

    def test_apply_lexical_ablation_obfuscates_endpoint_tokens_but_preserves_structure(self) -> None:
        records = [
            parse_code2vec_line("to|lower foo,Name|Call,bar foo,Name|Return,foo"),
            parse_code2vec_line("hash|code bar,Name|Call,foo"),
        ]

        obfuscated = apply_lexical_ablation(records, "obfuscated")

        self.assertEqual([record.label for record in obfuscated], ["to|lower", "hash|code"])
        self.assertEqual(obfuscated[0].contexts[0].ast_path, ("Name", "Call"))
        first_foo = obfuscated[0].contexts[0].start_token
        second_foo = obfuscated[0].contexts[1].end_token
        bar = obfuscated[0].contexts[0].end_token
        self.assertTrue(first_foo.startswith("LEX_"))
        self.assertEqual(first_foo, second_foo)
        self.assertNotEqual(first_foo, bar)
        self.assertNotIn("foo", {first_foo, bar})

    def test_apply_lexical_ablation_record_obfuscates_tokens_locally(self) -> None:
        records = [
            parse_code2vec_line("to|lower foo,Name|Call,bar foo,Name|Return,foo"),
            parse_code2vec_line("hash|code bar,Name|Call,foo"),
        ]

        obfuscated = apply_lexical_ablation(records, "record_obfuscated")

        first_record_foo = obfuscated[0].contexts[0].start_token
        repeated_first_record_foo = obfuscated[0].contexts[1].end_token
        first_record_bar = obfuscated[0].contexts[0].end_token
        second_record_foo = obfuscated[1].contexts[0].end_token
        second_record_bar = obfuscated[1].contexts[0].start_token

        self.assertEqual(first_record_foo, repeated_first_record_foo)
        self.assertNotEqual(first_record_foo, first_record_bar)
        self.assertTrue(first_record_foo.startswith("LEX_LOCAL_"))
        self.assertNotIn("foo", {first_record_foo, first_record_bar})
        self.assertNotEqual(first_record_foo, second_record_foo)
        self.assertNotEqual(first_record_bar, second_record_bar)

    def test_apply_lexical_ablation_structural_only_masks_all_endpoint_tokens(self) -> None:
        records = [
            parse_code2vec_line("to|lower foo,Name|Call,bar foo,Name|Return,baz"),
        ]

        masked = apply_lexical_ablation(records, "structural_only")

        self.assertEqual(masked[0].label, "to|lower")
        self.assertEqual(masked[0].contexts[0].ast_path, ("Name", "Call"))
        for context in masked[0].contexts:
            self.assertEqual(context.start_token, "<LEXICAL_MASK>")
            self.assertEqual(context.end_token, "<LEXICAL_MASK>")


if __name__ == "__main__":
    unittest.main()
