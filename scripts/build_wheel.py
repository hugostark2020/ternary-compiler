"""
Build pre-compiled wheels for Windows and Linux.
Usage:
    python scripts/build_wheel.py              # Build pure Python wheel
    python scripts/build_wheel.py --cpu-ext    # Build with C++ CPU extension
"""
import os
import sys
import subprocess
import shutil


def build_pure_wheel():
    """Build a pure Python wheel (no C++ extension)."""
    print("Building pure Python wheel...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "build", "wheel"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        check=True,
    )
    # Find the built wheel
    dist_dir = "dist"
    wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
    if wheels:
        print(f"✅ Built: {wheels[-1]}")
        return os.path.join(dist_dir, wheels[-1])
    return None


def build_with_cpu_ext():
    """Build wheel with C++ CPU extension (AVX-512)."""
    print("Building wheel with C++ CPU extension...")
    # First build the C++ extension
    subprocess.run(
        [sys.executable, "setup_cpu.py", "build_ext", "--inplace"],
        check=True,
    )
    # Then build the wheel
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        check=True,
    )
    dist_dir = "dist"
    wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
    if wheels:
        print(f"✅ Built with CPU ext: {wheels[-1]}")
        return os.path.join(dist_dir, wheels[-1])
    return None


def install_wheel(wheel_path):
    """Install the built wheel."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", wheel_path],
        check=True,
    )
    print(f"✅ Installed: {wheel_path}")


if __name__ == "__main__":
    build_cpu = "--cpu-ext" in sys.argv

    # Clean previous builds
    for d in ["build", "dist", "*.egg-info"]:
        shutil.rmtree(d, ignore_errors=True)

    if build_cpu:
        wheel = build_with_cpu_ext()
    else:
        wheel = build_pure_wheel()

    if wheel:
        install_wheel(wheel)
        print("\n🎯 Wheel ready for distribution!")
        print(f"   File: {wheel}")
        print(f"   Install: pip install {wheel}")