# Approved completion of the interrupted cost study

The author approved a paired continuation on 2026-10-15 after being told
that the original VM cannot be restored. Complete the missing CodeRankEmbed
control with D+AST measured on the same new VM. Keep the two sessions separate.
This changes the execution scope, not the retrieval methods or weights.

Session A retains its five systems and all 6000 measured requests without
replacement. Session B measures D+AST followed by D, preserving their relative
order in the original fixed schedule. Each uses the identical 400 query IDs,
three query orders, complete raw-query path, warmup, batch/precision settings
and isolated processes. Original benchmark, manifest, models and feature
code remain byte-identical. No timing is pooled or scaled between sessions.

The missing reranker quality computation is a replay of the original saved
candidate IDs and pair scores. The earlier assertion incorrectly required
identical floating-point dense ordering before associating those scores.
Bind scores to their saved IDs, report any recalculated candidate-order or
membership differences, and retain the fixed old candidate pool. This is not
new pair inference and does not retroactively complete session A.

Require two complete worker receipts, 2400 new timings, full quality for both
workers and a separately identified 3200-query cached reranker replay. Verify
all input/output hashes, model settings, query identities and metrics before
publication. Do not issue a six-method single-VM completion receipt.

Use only a private free Kaggle Tesla T4 with explicit NvidiaTeslaT4 accelerator.
The read-only quota check reports 4.29 GPU hours remaining before the
2026-10-17 reset. The continuation controller allows 2400 seconds after setup;
the hard kernel limit is 3600 seconds, retaining more than one hour of quota.
Repeat status, duplicate and quota checks before launch. No Stage B, Java test,
new model training, new feature/weight selection or journal submission.

The public package may contain code, numerical data and instructions only.
Original program bodies, model weights, credentials and manuscript files are
excluded; provide corpus reconstruction instructions and source hashes instead.
