# Code2Hyp explainability case

Dataset: BugNet Python slice.
Task: `task-025_p02924`.
Query: `data/external_task_corpus/bugnet_python_train_pass_16x32/task-025_p02924/p02924_0006_170348327683a173.py`.
Candidate: `data/external_task_corpus/bugnet_python_train_pass_16x32/task-025_p02924/p02924_0008_51f2577e1acd7ba7.py`.
Distance: `1.048507769206`.

| Rank | Mass | Cost | Query path | Candidate path |
|---:|---:|---:|---|---|
| 1 | 0.0062 | 1.5678 | Load -> Call -> Load (depth 4, len 4) | Load -> Call -> Load (depth 3, len 5) |
| 2 | 0.0060 | 1.3645 | token:range -> Call -> token:n (depth 4, len 4) | token:int -> Call -> token:input (depth 3, len 5) |
| 3 | 0.0059 | 1.4661 | Load -> Call -> token:n (depth 4, len 4) | Load -> Call -> token:input (depth 3, len 5) |
| 4 | 0.0051 | 0.8667 | token:input -> Name -> Load (depth 4, len 2) | token:n -> Name -> Store (depth 3, len 2) |
| 5 | 0.0051 | 0.8684 | token:range -> Name -> Load (depth 5, len 2) | token:input -> Name -> Load (depth 5, len 2) |
| 6 | 0.0049 | 0.6730 | Load -> Assign -> token:None (depth 1, len 5) | Load -> FunctionDef:main -> token:None (depth 1, len 5) |
| 7 | 0.0044 | 0.1724 | token:int -> Call -> token:input (depth 2, len 5) | token:n -> Assign -> token:int (depth 2, len 5) |
| 8 | 0.0041 | 0.6418 | Load -> Call -> Load (depth 2, len 5) | Store -> Assign -> Load (depth 2, len 5) |
| 9 | 0.0040 | 1.9825 | token:n -> Name -> Load (depth 5, len 2) | token:int -> Call -> Load (depth 3, len 5) |
| 10 | 0.0038 | 0.5499 | Load -> Call -> token:input (depth 2, len 5) | Store -> Assign -> token:int (depth 2, len 5) |
