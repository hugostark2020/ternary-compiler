#!/bin/bash
# Build the C++ CPU extension inside WSL
set -e

echo "=== Step 1: Copy project to ~/tc (no spaces in path) ==="
cp -r "/mnt/c/Users/LENOVO/Desktop/Insha ALLAH/ternary-compiler" ~/tc

echo "=== Step 2: Create virtual environment ==="
cd ~/tc
python3 -m venv venv-linux
source venv-linux/bin/activate

echo "=== Step 3: Install dependencies ==="
pip install -r requirements.txt -q 2>/dev/null || true
pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || true

echo "=== Step 4: Build C++ extension ==="
python setup_cpu.py build_ext --inplace 2>&1

echo "=== Step 5: Test extension ==="
python -c "import ternary_cpu_ext; print('✅ C++ CPU extension loaded')"

echo "=== Step 6: Run benchmark ==="
python examples/benchmark.py --model distilgpt2 --samples 10 --runs 3 --device cpu

echo "=== DONE ==="