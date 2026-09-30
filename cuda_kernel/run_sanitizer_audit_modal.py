"""Audit v4/v5/v6 unfiltered checks without changing the CUDA 12.8 build/baseline."""

from pathlib import Path

import modal
from cuda_sanitizer import CUDA_IMAGE, PACKAGE, PACKAGE_SHA256, SANITIZER, make_image


SOURCE_DIR = Path(__file__).resolve().parent
app = modal.App("mojo-v56-sanitizer-audit")
image = make_image()
SOURCES = ["v4.cu", "v5.cu", "v6.cu", "benchmark_v4.cu", "benchmark_v5.cu", "benchmark_v6.cu"]
for source in SOURCES:
    image = image.add_local_file(SOURCE_DIR / source, "/root/" + source)


@app.function(image=image, gpu="B200", timeout=1800)
def audit(repeats: int, compare_old: bool, bundled: bool) -> dict:
    import datetime
    import hashlib
    import os
    import signal
    import subprocess
    import time

    logs = []
    checker = "compute-sanitizer" if bundled else SANITIZER

    def run(command: list[str], timeout: int = 120) -> dict:
        print("$ " + " ".join(command), flush=True)
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        timed_out = False
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        record = dict(command=command, output=output, returncode=process.returncode,
                      timed_out=timed_out, seconds=time.monotonic() - start)
        print(output, flush=True)
        print(f"AUDIT STATUS: returncode={process.returncode} timed_out={timed_out}", flush=True)
        logs.append(record)
        return record

    def required(command: list[str]) -> None:
        record = run(command)
        if record["returncode"] or record["timed_out"]:
            raise RuntimeError("Audit preparation failed: " + " ".join(command))

    required(["nvidia-smi"])
    required(["/usr/local/cuda-12.8/bin/nvcc", "--version"])
    required(["compute-sanitizer", "--version"])
    required([SANITIZER, "--version"])
    binaries = []
    for version, mma_n in [(4, 256), (5, 256), (6, 256), (6, 128)]:
        binary = f"/root/benchmark_v{version}_mma{mma_n}"
        required([
            "/usr/local/cuda-12.8/bin/nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a",
            f"-DMOJO_V6_MMA_N={mma_n}", f"/root/benchmark_v{version}.cu", "-o", binary,
            "-L/usr/local/cuda-12.8/lib64/stubs", "-lcuda", "-lcublas",
        ])
        binaries.append(binary)

    comparison_checks = []
    if compare_old:
        for binary in binaries:
            for repeat in range(min(repeats, 3)):
                comparison_checks.append(run([
                    "compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "1",
                    binary, "--correctness-only", "4096", "4096", "576",
                ], timeout=60))

    commands = []
    for binary in binaries:
        for tool in ["memcheck", "synccheck", "racecheck"]:
            for repeat in range(repeats if tool == "racecheck" else 1):
                commands.append([checker, "--tool", tool, "--error-exitcode", "1",
                                 binary, "--correctness-only", "4096", "4096", "576"])
        commands.append([checker, "--tool", "racecheck", "--error-exitcode", "1",
                         binary, "--correctness-only", "4096", "4096", "4096"])
    checks = []
    for command in commands:
        record = run(command)
        checks.append(record)
        if record["timed_out"] or record["returncode"]:
            break
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                cuda_image=CUDA_IMAGE, sanitizer_package="CUDA 12.8.1 bundled" if bundled else PACKAGE,
                sanitizer_package_sha256=None if bundled else PACKAGE_SHA256,
                scope="unfiltered: custom kernels and cuBLAS", repeats=repeats,
                passed=len(checks) == len(commands) and all(
                    not record["timed_out"] and record["returncode"] == 0
                    for record in checks + comparison_checks),
                source_sha256={name: hashlib.sha256(Path('/root', name).read_bytes()).hexdigest() for name in SOURCES},
                logs=logs)


@app.local_entrypoint()
def main(repeats: int = 5, compare_old: bool = False, bundled: bool = True):
    import json

    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be between 1 and 10")
    if bundled and compare_old:
        raise ValueError("Use --no-bundled with --compare-old")
    result = audit.remote(repeats, compare_old, bundled)
    suffix = "" if bundled else "_standalone"
    output = SOURCE_DIR / f"sanitizer_v456_audit{suffix}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}")
    if not result["passed"]:
        raise RuntimeError("Unfiltered sanitizer audit failed; see saved logs")
