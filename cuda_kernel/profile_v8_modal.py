from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("mojo-v8-diagnose")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file(ROOT / "v8.cu", "/root/v8.cu")
    .add_local_file(ROOT / "benchmark_v8.cu", "/root/benchmark_v8.cu")
)


@app.function(image=image, gpu="B200", timeout=1200)
def diagnose() -> dict[str, bytes]:
    import subprocess

    def run(command, timeout=300, required=True):
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        print(" ".join(command) + f" -> exit {process.returncode}, {len(process.stdout)} chars", flush=True)
        if required:
            process.check_returncode()
        return process.stdout

    build = run(["nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_100a", "--keep", "--keep-dir", "/tmp", "-Xptxas=-v", "/root/benchmark_v8.cu", "-o", "/root/benchmark_v8", "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lcublas"])
    sass = run(["cuobjdump", "--dump-sass", "/root/benchmark_v8"])
    profile = run(["ncu", "--set", "full", "--clock-control", "none", "--kernel-name-base", "demangled", "--kernel-name", "regex:mojo_v8", "--launch-count", "1", "--export", "/tmp/v8", "--force-overwrite", "/root/benchmark_v8", "--correctness-only", "4096", "4096", "4096"], timeout=600, required=False)
    details = run(["ncu", "--import", "/tmp/v8.ncu-rep", "--page", "details"], required=False)
    source = run(["ncu", "--import", "/tmp/v8.ncu-rep", "--page", "source", "--print-source", "cuda,sass"], required=False)
    result = {"build.txt": build.encode(), "sass.txt": sass.encode(), "profile.txt": (profile + details).encode(), "source.txt": source.encode()}
    for path in Path("/tmp").glob("*.ptx"):
        result[path.name] = path.read_bytes()
    report = Path("/tmp/v8.ncu-rep")
    if report.exists():
        result[report.name] = report.read_bytes()
    return result


@app.local_entrypoint()
def main(label: str = "fixed"):
    if not label.replace("_", "").isalnum():
        raise ValueError("Use an alphanumeric profile label")
    results = diagnose.remote()
    output = ROOT / f"v8_profile_{label}"
    output.mkdir(exist_ok=True)
    for name, content in results.items():
        (output / name).write_bytes(content)
    print(f"Saved {output}")
