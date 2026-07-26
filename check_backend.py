"""Check which kernel backend is active."""
import sys
sys.path.insert(0, "src")
from kernels import get_backend

print(f"Active backend: {get_backend()}")