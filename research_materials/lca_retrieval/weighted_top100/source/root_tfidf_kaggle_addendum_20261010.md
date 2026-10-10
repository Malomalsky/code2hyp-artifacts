# Kaggle execution addendum, 2026-10-10

The user explicitly requested Kaggle instead of Colab for the separate frozen
root-TF-IDF new-problem replication. This addendum is fixed before any new-cohort
neural encoding or retrieval results. It does not authorize changes to Stage B,
CPRet, Java test access, selection, features, hypotheses, weights, or analysis.

## Execution-only changes

Use a private Kaggle dataset and private script kernel on NvidiaTeslaT4, with
explicit machine_shape and --accelerator. Only free GPU quota is permitted.
The live CLI reported 23.18 of 30 GPU hours remaining on 2026-10-10, refresh
2026-10-17 00:00 UTC; Stage B seed18 is RUNNING and is not modified. There is
no existing root-tfidf kernel or dataset. Check again immediately before launch.

The preserved coderank_colab.py is the source for coderank_kaggle_20261010.py.
Changes are the Kaggle-only environment guard, fixed new source/design binding,
output directory, T4 assertion, and model-file verification before loading.
The model, revision, float32, batch 16, max length 512, CLS pooling, normalization,
empty query prefix, pilot and 128-program checkpoints remain unchanged.
The preserved Colab adapter is not used. Original files/receipts stay intact.

Pin sentence-transformers 3.1.1, transformers 4.45.1 and numpy 2.1.3.
Use the installed CUDA PyTorch and record its actual version. Do not claim
bitwise or retrieval equivalence to the earlier Colab runtime. All compared
methods in this new-cohort evaluation share the same new dense embeddings.
The model snapshot must exactly match every saved old model-file hash.

Write outputs/checkpoints only under /kaggle/working/root_tfidf_encoding.
On interruption, retrieve the checkpoint and resume through a private input
dataset with the same settings binding. Do not repeat completed encoding.
The 64-program resource pilot is not a quality test; full encoding starts only
if projected encoding is below one hour. Encoding also has a one-hour bound.
No local neural inference, training, purchases, or public uploads.

## Analysis remains frozen

root_tfidf_evaluate_kaggle_20261010.py is a preserved-copy adaptation of the
already frozen evaluator. Only encoder/provenance checks, separate output
directory, and execution-code receipt paths differ. Its mixed() scoring and
metric/bootstrap/decision logic must remain unchanged, checked in tests.
All fourteen rankings and all prespecified contrasts are reported regardless
of sign. No new weight selection, exclusions, or model choices are allowed.
The original scientific design and evaluation-code receipts remain binding,
alongside a separate Kaggle execution receipt fixed before launch.

This is a same-domain replication on previously unused problems, not an
independent domain or a proven pretraining-clean benchmark. Missing old full
Python inventory remains an explicit limitation. Success@10 is the correct
name for the legacy recall_at_10 field. MAP@8 remains primary.
