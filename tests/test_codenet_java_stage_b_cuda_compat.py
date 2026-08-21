import torch

from scripts.run_codenet_java_stage_b_cuda_compat import _set_current_cuda_index


def test_unindexed_cuda_device_resolves_to_current_index() -> None:
    observed = []

    _set_current_cuda_index(observed.append, lambda: 3, torch.device("cuda"))
    _set_current_cuda_index(observed.append, lambda: 3, torch.device("cuda:1"))
    _set_current_cuda_index(observed.append, lambda: 3, 2)

    assert observed == [torch.device("cuda:3"), torch.device("cuda:1"), 2]
