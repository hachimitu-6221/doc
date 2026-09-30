"""Run the faithful Mojo v4 CUDA port and cuBLAS baseline on a Modal B200."""

from pathlib import Path

import modal


SOURCE_DIR = Path(__file__).resolve().parent
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
SANITIZER = "compute-sanitizer"
app = modal.App("mojo-v4-cuda-cublas")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .add_local_file(SOURCE_DIR / "v4.cu", "/root/v4.cu")
    .add_local_file(SOURCE_DIR / "benchmark_v4.cu", "/root/benchmark_v4.cu")
    .add_local_file(SOURCE_DIR.parent / "mojo_kernel" / "v4.mojo", "/root/v4.mojo")
)


@app.function(image=image, gpu="B200", timeout=1200)
def benchmark(m: int, n: int, k: int, sanitize: bool) -> dict:
    import datetime
    import hashlib
    import json
    import os
    import signal
    import subprocess

    logs = []

    def run(command: list[str], timeout: int = 180) -> str:
        print("$ " + " ".join(command), flush=True)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            print(output, flush=True)
            raise
        print(output, flush=True)
        logs.append({"command": command, "output": output})
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output)
        return output

    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    run([
        "nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
        "-Xptxas=-v", "/root/benchmark_v4.cu", "-o", "/root/benchmark_v4",
        "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas",
    ])
    for shape in [(256, 256, 64), (512, 512, 192), (768, 768, 576), (4096, 4096, 64)]:
        run(["/root/benchmark_v4", "--correctness-only", *map(str, shape)])
    if sanitize:
        run([SANITIZER, "--version"])
        for tool in ["memcheck", "synccheck", "racecheck"]:
            run([
                SANITIZER, "--tool", tool, "--error-exitcode", "1",
                "/root/benchmark_v4", "--correctness-only", "768", "512", "576",
            ], timeout=300)
        run([
            SANITIZER, "--tool", "racecheck", "--error-exitcode", "1",
            "/root/benchmark_v4", "--correctness-only", "4096", "4096", "576",
        ], timeout=300)
    run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu", "--format=csv"])
    output = run(["/root/benchmark_v4", str(m), str(n), str(k)], timeout=300)
    run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu", "--format=csv"])
    result = next(
        json.loads(line.removeprefix("RESULT_JSON="))
        for line in output.splitlines() if line.startswith("RESULT_JSON=")
    )
    result.update(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cuda_image=CUDA_IMAGE,
        sanitized=sanitize,
        sanitizer_scope="unfiltered: custom kernels and cuBLAS" if sanitize else "not run",
        sanitizer_package="CUDA 12.8.1 bundled" if sanitize else None,
        timing="CUDA Graph, 50 alternating warmup replays each, 100 GEMMs/replay, CUDA events, alternating order, median of 7 rounds",
        baseline="cublasGemmEx, CUBLAS_COMPUTE_32F, reduced-precision reduction disabled",
        source_sha256={
            name: hashlib.sha256(Path("/root", name).read_bytes()).hexdigest()
            for name in ["v4.cu", "benchmark_v4.cu", "v4.mojo"]
        },
        logs=logs,
    )
    return result


@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096, sanitize: bool = False):
    import json

    if m <= 0 or n <= 0 or k <= 0 or m % 256 or n % 256 or k % 64:
        raise ValueError("Required: positive M,N,K; M % 256 = N % 256 = K % 64 = 0")
    result = benchmark.remote(m, n, k, sanitize)
    result_path = SOURCE_DIR / f"results_v4_b200_{m}x{n}x{k}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {result_path}")
