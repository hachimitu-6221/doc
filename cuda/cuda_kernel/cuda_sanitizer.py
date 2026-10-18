"""Pinned standalone checker; compilation and cuBLAS remain on CUDA 12.8."""

from pathlib import Path

import modal


CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
PACKAGE = "cuda-sanitizer-13-0_13.0.85-1_amd64.deb"
PACKAGE_SHA256 = "5913520009ecc86be1c62b5793b032f81fdffdfcd4493da6212e14c3dc1f35a4"
PACKAGE_URL = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/" + PACKAGE
SANITIZER = "/opt/sanitizer/usr/local/cuda-13.0/compute-sanitizer/compute-sanitizer"


def make_image() -> modal.Image:
    return modal.Image.from_registry(CUDA_IMAGE, add_python="3.12").run_commands(
        f"python -c \"import urllib.request; urllib.request.urlretrieve('{PACKAGE_URL}', '/tmp/{PACKAGE}')\"",
        f"echo '{PACKAGE_SHA256}  /tmp/{PACKAGE}' | sha256sum -c -",
        f"dpkg-deb -x /tmp/{PACKAGE} /opt/sanitizer && rm /tmp/{PACKAGE}",
    ).add_local_file(Path(__file__), "/root/cuda_sanitizer.py")
