# Code2Hyp explainability case

Dataset: DTA Zenodo materialized Python subset.
Task: `task-01`.
Query: `data/dta_zenodo_7799972/materialized_python_balanced_64/task-01/task_01_0005_41dc04bd21003824.py`.
Candidate: `data/dta_zenodo_7799972/materialized_python_balanced_64/task-01/task_01_0021_e29fdc9bf57eefd3.py`.
Distance: `0.067685085024`.

| Rank | Mass | Cost | Query path | Candidate path |
|---:|---:|---:|---|---|
| 1 | 0.0078 | 0.0201 | token:math -> alias:math -> token:None (LCA depth 2, length 2) | token:math -> alias:math -> token:None (LCA depth 2, length 2) |
| 2 | 0.0076 | 0.0295 | token:math -> Module -> Mult (LCA depth 0, length 9) | token:math -> Module -> Mult (LCA depth 0, length 9) |
| 3 | 0.0069 | 0.0367 | token:math -> Module -> Load (LCA depth 0, length 12) | token:math -> Module -> Load (LCA depth 0, length 10) |
| 4 | 0.0063 | 0.0051 | token:math -> Module -> token:None (LCA depth 0, length 5) | token:math -> Module -> token:None (LCA depth 0, length 5) |
| 5 | 0.0063 | 0.0166 | token:math -> Module -> token:None (LCA depth 0, length 5) | token:math -> Module -> token:None (LCA depth 0, length 5) |
| 6 | 0.0062 | 0.0363 | token:None -> Module -> Load (LCA depth 0, length 8) | token:None -> Module -> Load (LCA depth 0, length 8) |
| 7 | 0.0057 | 0.1040 | token:math -> Module -> Mult (LCA depth 0, length 14) | token:math -> Module -> Mult (LCA depth 0, length 10) |
| 8 | 0.0056 | 0.0248 | token:math -> Module -> Load (LCA depth 0, length 8) | token:math -> Module -> Load (LCA depth 0, length 8) |
