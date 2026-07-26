"""
CPU-optimized backend for ternary matrix multiplication.

Supports:
- Numba JIT with parallel acceleration (auto-detects CPU features)
- Native PyTorch fallback

For maximum performance, a C++ extension with AVX-512/NEON intrinsics
can be built via `python setup.py build_ext`.
"""
import torch
import numpy as np


def _get_cpu_capabilities():
    """Detect CPU capabilities for kernel selection."""
    try:
        import cpuinfo  # optional: py-cpuinfo

        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        return {
            "avx512": any(f in flags for f in ["avx512f", "avx512"]),
            "avx2": "avx2" in flags,
            "neon": "neon" in flags,
        }
    except ImportError:
        # Default conservative detection
        import platform

        is_arm = "aarch" in platform.machine().lower()
        return {"avx512": False, "avx2": False, "neon": is_arm}


def _ternary_matmul_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated ternary matrix multiplication.

    A is ternary {-1, 0, +1} values, B is FP32.
    Uses parallel loops for CPU speedup.
    """
    try:
        from numba import njit, prange

        @njit(parallel=True, fastmath=True, cache=True)
        def _kernel(a, b, out):
            M, K = a.shape
            K, N = b.shape
            for i in prange(M):
                for j in range(N):
                    s = 0.0
                    for k in range(K):
                        s += a[i, k] * b[k, j]
                    out[i, j] = s

        M, K = a.shape
        _, N = b.shape
        out = np.zeros((M, N), dtype=np.float32)
        _kernel(a, b, out)
        return out
    except ImportError:
        return None


class CPUTernaryBackend:
    """
    CPU backend that auto-selects the fastest available implementation.
    """

    def __init__(self):
        self.caps = _get_cpu_capabilities()
        print(
            f"CPU backend: AVX512={self.caps['avx512']}, "
            f"AVX2={self.caps['avx2']}, NEON={self.caps['neon']}"
        )
        self._use_numba = False
        try:
            import numba  # type: ignore

            self._use_numba = True
        except ImportError:
            pass

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute ternary matmul on CPU with best available backend.

        Args:
            a: Ternary weight matrix (M, K) with values in {-1, 0, +1}
            b: Activation matrix (K, N) in FP16/FP32

        Returns:
            Output matrix (M, N)
        """
        assert a.shape[1] == b.shape[0]

        # Use Numba if available (fastest on CPU)
        if self._use_numba and a.device.type == "cpu":
            a_np = a.cpu().numpy().astype(np.float32)
            b_np = b.cpu().numpy().astype(np.float32)
            result = _ternary_matmul_numba(a_np, b_np)
            if result is not None:
                return torch.from_numpy(result).to(b.dtype)

        # Fallback: native PyTorch
        return torch.matmul(a.to(b.dtype), b)


def get_cpu_backend():
    """Initialize and return the CPU backend."""
    return CPUTernaryBackend()