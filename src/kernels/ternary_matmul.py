"""
Ternary matrix multiplication kernel with support for:
- Triton (Linux/CUDA) — fastest, custom JIT kernels
- torch.compile (Windows/Linux) — PyTorch JIT compilation
- PyTorch native (fallback) — always works
"""
import torch


def _get_available_backend():
    """Detect the best available kernel backend on the current platform."""
    # Check for Triton (Linux only typically)
    try:
        import triton  # type: ignore
        return "triton"
    except ImportError:
        pass

    # Check for torch.compile (available in PyTorch 2.0+ on all platforms)
    if hasattr(torch, "compile") and torch.cuda.is_available():
        try:
            # Test if torch.compile actually works
            test_fn = lambda x: x * 2
            compiled = torch.compile(test_fn, backend="inductor")
            t = torch.tensor([1.0], device="cuda")
            compiled(t)
            return "torch_compile"
        except Exception:
            pass

    # Fallback to native PyTorch
    return "native"


def _ternary_matmul_native(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch implementation of ternary matmul.

    A is ternary {-1, 0, +1}, B is FP16. Uses optimized torch.matmul.
    """
    return torch.matmul(a.to(b.dtype), b)


def _ternary_matmul_compiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.compile-optimized version of ternary matmul.

    Uses PyTorch's inductor backend to JIT-compile the computation graph
    for faster execution on CUDA.
    """
    # The core computation
    result = torch.matmul(a.to(b.dtype), b)
    return result


# Apply torch.compile if available for the CUDA path
if hasattr(torch, "compile") and torch.cuda.is_available():
    try:
        _ternary_matmul_compiled = torch.compile(
            _ternary_matmul_compiled,
            backend="inductor",
            mode="max-autotune",
            fullgraph=True,
        )
    except Exception:
        pass


# Detect backend
BACKEND = _get_available_backend()


def ternary_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix multiplication where A is ternary {-1, 0, +1} and B is FP16.

    Automatically selects the best available backend:
    - Triton (Linux/CUDA): Custom JIT kernel for max performance
    - torch.compile (Windows/Linux + CUDA): PyTorch JIT compilation
    - Native PyTorch (any platform): Always works

    Args:
        a: Ternary weight matrix of shape (M, K) with values in {-1, 0, +1}
        b: Activation matrix of shape (K, N) in FP16

    Returns:
        Output matrix of shape (M, N) in FP16
    """
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} x {b.shape}"

    if BACKEND == "triton":
        return _ternary_matmul_triton(a, b)
    elif BACKEND == "torch_compile":
        return _ternary_matmul_compiled(a, b)
    else:
        return _ternary_matmul_native(a, b)


def _ternary_matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated ternary matmul (Linux only)."""
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    assert a.is_cuda and b.is_cuda, "Triton requires CUDA tensors"
    assert b.dtype == torch.float16, "B must be FP16 for Triton"

    M, K = a.shape
    K, N = b.shape

    @triton.jit
    def _kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = (
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        )
        tl.store(c_ptrs, c, mask=offs_cm[:, None] < M and offs_cn[None, :] < N)

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    _kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return c


def get_backend() -> str:
    """Return the active kernel backend name."""
    return BACKEND


__all__ = ["ternary_matmul", "get_backend"]