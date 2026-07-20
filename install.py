"""
ComfyUI-Manager install / update hook.

Keeps nvidia-ml-py (pynvml) at the latest PyPI release for the
General Manage VRAM startup NVML patch (v2.4.0+).
"""
import subprocess
import sys


def main():
    print("[ComfyUI-VRAM-Manager] install.py: upgrading nvidia-ml-py to latest...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "nvidia-ml-py"],
    )
    print("[ComfyUI-VRAM-Manager] install.py: nvidia-ml-py ready.")


if __name__ == "__main__":
    main()
