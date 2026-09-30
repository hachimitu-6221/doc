"""Run the faithful Mojo v1 CUDA port and cuBLAS baseline on a Modal B200."""

from pathlib import Path

import modal


SOURCE_DIR = Path(__file__).resolve().parent
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
app = modal.App("mojo-v1-cuda-cublas")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .add_local_file(SOURCE_DIR / "v1.cu", "/root/v1.cu")
    .add_local_file(SOURCE_DIR / "benchmark.cu", "/root/benchmark.cu")
)


@app.function(image=image, gpu="B200", timeout=1200)
def benchmark(m: int, n: int, k: int, sanitize: bool) -> dict:
    import datetime
    import hashlib
    import json
    import subprocess

    logs = []

    def run(command: list[str], timeout: int = 180) -> str:
        print("$ " + " ".join(command), flush=True)
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        print(process.stdout, flush=True)
        logs.append({"command": command, "output": process.stdout})
        process.check_returncode()
        return process.stdout

    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    run([
        "nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
        "-Xptxas=-v", "/root/benchmark.cu", "-o", "/root/benchmark",
        "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas",
    ])
    for shape in [(64, 256, 64), (128, 512, 192), (192, 256, 512)]:
        run(["/root/benchmark", "--correctness-only", *map(str, shape)])
    if sanitize:
        for tool in ["memcheck", "synccheck", "racecheck"]:
            run([
                "compute-sanitizer", "--tool", tool, "--error-exitcode", "1",
                "/root/benchmark", "--correctness-only", "128", "512", "192",
            ], timeout=300)
    output = run(["/root/benchmark", str(m), str(n), str(k)], timeout=300)
    result = next(
        json.loads(line.removeprefix("RESULT_JSON="))
        for line in output.splitlines() if line.startswith("RESULT_JSON=")
    )
    result.update(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cuda_image=CUDA_IMAGE,
        timing="CUDA Graph, 100 GEMMs/replay, CUDA events, alternating order, median of 7 rounds",
        baseline="cublasGemmEx, CUBLAS_COMPUTE_32F, reduced-precision reduction disabled",
        source_sha256={
            name: hashlib.sha256(Path("/root", name).read_bytes()).hexdigest()
            for name in ["v1.cu", "benchmark.cu"]
        },
        logs=logs,
    )
    return result


@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096, sanitize: bool = False):
    import json

    if m <= 0 or n <= 0 or k <= 0 or m % 64 or n % 256 or k % 64:
        raise ValueError("Required: positive M,N,K; M % 64 = N % 256 = K % 64 = 0")
    result = benchmark.remote(m, n, k, sanitize)
    result_path = SOURCE_DIR / f"results_b200_{m}x{n}x{k}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {result_path}")
