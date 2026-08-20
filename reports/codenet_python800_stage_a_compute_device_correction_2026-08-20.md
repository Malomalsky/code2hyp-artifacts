# CodeNet Python800 Stage A compute-device correction

Date: 2026-08-20

The RunPod validation and test pods exposed an NVIDIA GeForce RTX 3090, but
runner v4 did not move the model or encoded measures to CUDA. PyTorch therefore
executed training and Sinkhorn evaluation on the CPU. The CUDA-enabled PyTorch
build and the presence of a pod GPU do not constitute evidence of GPU use.

This correction changes the hardware description, not the frozen Stage A
results. The stored distance matrices, recomputed metrics, validation selection,
test-opening record and seals remain the outputs of the registered runner. No
Stage A claim should describe those calculations as GPU-accelerated or use them
as a GPU runtime benchmark.

The independent Java Stage B runner fixes the issue before registration. Its
design explicitly requires `compute_device=cuda`; the process fails before data
materialization when CUDA is unavailable, disables TF32 and cuDNN benchmarking,
enables deterministic PyTorch algorithms, records the CUDA device and runtime,
and saves the final distance matrices as CPU `float64` tensors for independent
recomputation.
