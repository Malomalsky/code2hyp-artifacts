from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, Callable

import torch


def _set_current_cuda_index(
    set_device: Callable[[Any], None],
    current_device: Callable[[], int],
    value: Any,
) -> None:
    if isinstance(value, int):
        set_device(value)
        return
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", current_device())
    set_device(device)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_codenet_java_stage_b_cuda_compat.py RUNNER [RUNNER_ARGS ...]")
    runner = Path(sys.argv.pop(1)).resolve()
    if not runner.is_file():
        raise SystemExit(f"runner does not exist: {runner}")

    original_set_device = torch.cuda.set_device
    torch.cuda.set_device = lambda value: _set_current_cuda_index(
        original_set_device,
        torch.cuda.current_device,
        value,
    )
    sys.argv[0] = str(runner)
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
