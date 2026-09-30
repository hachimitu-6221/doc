"""Bisect mojo_v8 CUDA port bottleneck on a Modal B200.

Builds exp_v8.cu in three variants (base / -DSKIP_EPILOGUE / -DSKIP_MMA) and
times a K sweep to separate per-iteration vs per-tile costs.

Usage:
    .venv/bin/modal run kernel/cuda_kernel/run_exp_v8_modal.py
"""

from pathlib import Path

import modal

SOURCE_DIR = Path(__file__).resolve().parent
app = modal.App("mojo-v8-exp")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file(SOURCE_DIR / "exp_v8.cu", "/root/exp_v8.cu")
)


@app.function(image=image, gpu="B200", timeout=1800)
def run_experiments() -> None:
    import subprocess

    def run(command: list[str], timeout: int = 300) -> None:
        print("$ " + " ".join(command), flush=True)
        try:
            process = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout
            )
            print(process.stdout, flush=True)
            if process.returncode != 0:
                print(f"!! exit code {process.returncode}", flush=True)
        except subprocess.TimeoutExpired:
            print("!! TIMEOUT", flush=True)

    run(["nvidia-smi"])
    variants = [
        ("base", []),
        ("bb4", ["-DBIGBOX_N=4"]),
        ("bb4promo", ["-DBIGBOX_N=4", "-DL2_PROMO"]),
        ("t256", ["-DBLOCK_N=128", "-DMMA_N=256", "-DAB_STAGES=6"]),
        ("t256_bb2", ["-DBLOCK_N=128", "-DMMA_N=256", "-DAB_STAGES=6", "-DBIGBOX_N=2"]),
        ("t256_bb2_promo", ["-DBLOCK_N=128", "-DMMA_N=256", "-DAB_STAGES=6", "-DBIGBOX_N=2", "-DL2_PROMO"]),
        ("t256_bb3_promo", ["-DBLOCK_N=128", "-DMMA_N=256", "-DAB_STAGES=6", "-DBIGBOX_N=3", "-DL2_PROMO"]),
    ]
    for name, defines in variants:
        run([
            "nvcc", "-std=c++17", "-O3", "-arch=sm_100a", *defines,
            "/root/exp_v8.cu", "-o", f"/root/exp_{name}",
            "-L/usr/local/cuda/lib64/stubs", "-lcuda",
        ])
    for name, _ in variants:
        run([f"/root/exp_{name}", "4096", "4096", "4096"])


@app.local_entrypoint()
def main() -> None:
    run_experiments.remote()
