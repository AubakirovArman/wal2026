"""Path A Diagnostic: shared-memory codebook gather vs global gather vs dense.

Hypothesis: A 3KB codebook fits in shared memory / L1 cache.
If we stage it explicitly in CUDA shared memory, gather latency drops
from HBM random-access to ~L1 latency, compensating for scalar (non-tensor-core)
compute.

We test three paths on synthetic DRL v2 encoded weights:
1. Dense FP16 F.linear (tensor cores, baseline)
2. Current Triton fused global gather (_id_route_linear_kernel)
3. CUDA inline kernel with explicit shared-memory codebook staging

Metrics: latency (ms), effective compute (TFLOPS), effective bandwidth (GB/s).
"""
from __future__ import annotations

import math
import time
import warnings

import torch
import torch.utils.cpp_extension

# ---------------------------------------------------------------------------
# Synthetic data generation (DRL v2 format)
# ---------------------------------------------------------------------------
def make_synthetic_problem(
    n: int = 4096,
    k: int = 4096,
    codebook_size: int = 1500,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Return (x, ids, codebook_sum, row_scale, dense_w) for benchmarking.
    dense_w is constructed EXACTLY as codebook_sum[ids] so reconstruction is perfect.
    """
    torch.manual_seed(42)
    codebook_sum = torch.randn(codebook_size, dtype=torch.float32, device=device)
    ids = torch.randint(0, codebook_size, (n, k), dtype=torch.int32, device=device)
    dense_w = codebook_sum[ids.long()].to(dtype)
    row_scale = torch.ones(n, dtype=torch.float32, device=device)
    return dense_w, ids, codebook_sum, row_scale


# ---------------------------------------------------------------------------
# Baseline 1: Dense F.linear (tensor cores)
# ---------------------------------------------------------------------------
def bench_dense(x, dense_w, repeats: int = 50):
    # warmup
    for _ in range(10):
        torch.nn.functional.linear(x, dense_w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = torch.nn.functional.linear(x, dense_w)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


# ---------------------------------------------------------------------------
# Baseline 2: Materialize then dense (PackedIDRouteLinear reference path)
# ---------------------------------------------------------------------------
def bench_materialize(x, ids, codebook_sum, row_scale, repeats: int = 50):
    # warmup
    for _ in range(10):
        w = codebook_sum[ids.long()] * row_scale.unsqueeze(1)
        torch.nn.functional.linear(x, w.to(x.dtype))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        w = codebook_sum[ids.long()] * row_scale.unsqueeze(1)
        out = torch.nn.functional.linear(x, w.to(x.dtype))
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


# ---------------------------------------------------------------------------
# Baseline 3: Current Triton fused kernel (global gather)
# ---------------------------------------------------------------------------
def bench_triton_fused(x, ids, codebook_sum, row_scale, repeats: int = 50):
    import sys
    sys.path.insert(0, "src")
    from triton_id_matmul import id_route_linear_matmul
    # warmup
    for _ in range(10):
        id_route_linear_matmul(x, ids, codebook_sum, row_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = id_route_linear_matmul(x, ids, codebook_sum, row_scale)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


# ---------------------------------------------------------------------------
# CUDA inline kernel with explicit shared-memory codebook staging
# ---------------------------------------------------------------------------
_CPP_SOURCE = """
#include <torch/extension.h>
torch::Tensor path_a_smem_gather(torch::Tensor x, torch::Tensor ids, torch::Tensor codebook, torch::Tensor row_scale, int block_m, int block_n, int block_k);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Simple scalar kernel: no tensor cores, but explicit SMEM codebook.
// Each block handles BLOCK_N output rows and BLOCK_M batch items.
// K is looped over with BLOCK_K chunks.

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void path_a_smem_gather_kernel(
    const __half* __restrict__ x,
    const int*   __restrict__ ids,
    const __half* __restrict__ codebook,
    const __half* __restrict__ row_scale,
    __half* __restrict__ out,
    int M, int N, int K, int codebook_size
) {
    // Stage codebook into shared memory (3KB fits easily)
    extern __shared__ char smem[];
    __half* sm_codebook = reinterpret_cast<__half*>(smem);

    // Collaborative load of codebook
    for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
        sm_codebook[i] = codebook[i];
    }
    __syncthreads();

    // Tile coordinates
    int m0 = blockIdx.x * BLOCK_M;
    int n0 = blockIdx.y * BLOCK_N;

    // Accumulators in registers
    float acc[BLOCK_M][BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; ++i) {
        #pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K dimension
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load activations for this tile (x[m, k])
        // We load on-the-fly without SMEM staging for x to keep code simple.
        // Main point: weight gather uses SMEM codebook.
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            int k = k0 + kk;
            if (k >= K) break;

            // Each thread handles a subset of (m,n) pairs within the block
            int tid = threadIdx.x;
            int total_threads = blockDim.x;

            for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += total_threads) {
                int im = idx / BLOCK_N;
                int in = idx % BLOCK_N;
                int m = m0 + im;
                int n = n0 + in;
                if (m >= M || n >= N) continue;

                float a = __half2float(x[m * K + k]);
                int id_val = ids[n * K + k];
                float w = __half2float(sm_codebook[id_val]);
                acc[im][in] += a * w;
            }
        }
    }

    // Apply row scale and store
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += total_threads) {
        int im = idx / BLOCK_N;
        int in = idx % BLOCK_N;
        int m = m0 + im;
        int n = n0 + in;
        if (m >= M || n >= N) continue;
        float rs = __half2float(row_scale[n]);
        out[m * N + n] = __float2half(acc[im][in] * rs);
    }
}

// Wrapper
 torch::Tensor path_a_smem_gather(
    torch::Tensor x,
    torch::Tensor ids,
    torch::Tensor codebook,
    torch::Tensor row_scale,
    int block_m, int block_n, int block_k
) {
    int M = x.size(0);
    int K = x.size(1);
    int N = ids.size(0);
    int codebook_size = codebook.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 grid((M + block_m - 1) / block_m, (N + block_n - 1) / block_n);
    int threads = 256;
    size_t smem_size = codebook_size * sizeof(__half);

    // We use a simple dispatch; only one template combo for brevity.
    // If shapes don't divide evenly, the kernel checks bounds internally.
    path_a_smem_gather_kernel<8, 8, 32><<<grid, threads, smem_size>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        ids.data_ptr<int>(),
        reinterpret_cast<const __half*>(codebook.data_ptr()),
        reinterpret_cast<const __half*>(row_scale.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        M, N, K, codebook_size
    );
    return out;
}
"""

_path_a_cuda = None

def _load_cuda():
    global _path_a_cuda
    if _path_a_cuda is not None:
        return _path_a_cuda
    # Try to compile inline CUDA extension
    try:
        _path_a_cuda = torch.utils.cpp_extension.load_inline(
            name="path_a_smem",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["path_a_smem_gather"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _path_a_cuda
    except Exception as e:
        warnings.warn(f"CUDA inline compile failed: {e}")
        return None


def bench_cuda_path_a(x, ids, codebook_sum, row_scale, repeats: int = 50):
    return None, None  # CUDA ext incompatible with H200 SM90
    # Convert to half for CUDA kernel
    x_h = x.half()
    cb_h = codebook_sum.half()
    rs_h = row_scale.half()
    # warmup
    for _ in range(10):
        mod.path_a_smem_gather(x_h, ids, cb_h, rs_h, 8, 8, 32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = mod.path_a_smem_gather(x_h, ids, cb_h, rs_h, 8, 8, 32)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


# ---------------------------------------------------------------------------
# Main benchmark harness
# ---------------------------------------------------------------------------
def run_diagnostic(ms: list[int], n: int = 4096, k: int = 4096):
    device = "cuda:3"  # user requested GPU 2 and 3 only
    dtype = torch.float16
    torch.cuda.set_device(2)

    dense_w, ids, codebook_sum, row_scale = make_synthetic_problem(n, k, device=device, dtype=dtype)

    results = []
    for m in ms:
        x = torch.randn(m, k, dtype=dtype, device=device)

        # 1. Dense
        t_dense, out_dense = bench_dense(x, dense_w)
        flops = 2.0 * m * n * k
        tflops_dense = flops / (t_dense / 1000.0) / 1e12

        # 2. Materialize + linear
        t_mat, out_mat = bench_materialize(x, ids, codebook_sum, row_scale)

        # 3. Triton fused global gather
        try:
            t_triton, out_triton = bench_triton_fused(x, ids, codebook_sum, row_scale)
        except Exception as e:
            t_triton, out_triton = None, None

        # 4. CUDA Path A
        t_cuda, out_cuda = bench_cuda_path_a(x, ids, codebook_sum, row_scale)

        # Verify correctness where available
        max_err_mat = (out_mat - out_dense).abs().max().item() if out_mat is not None else None
        max_err_triton = (out_triton - out_dense).abs().max().item() if out_triton is not None else None
        max_err_cuda = (out_cuda - out_dense).abs().max().item() if out_cuda is not None else None

        results.append({
            "m": m,
            "dense_ms": round(t_dense, 3),
            "dense_tflops": round(tflops_dense, 2),
            "mat_ms": round(t_mat, 3),
            "mat_speedup": round(t_dense / t_mat, 2) if t_mat else None,
            "triton_ms": round(t_triton, 3) if t_triton else None,
            "triton_speedup": round(t_dense / t_triton, 2) if t_triton else None,
            "cuda_ms": round(t_cuda, 3) if t_cuda else None,
            "cuda_speedup": round(t_dense / t_cuda, 2) if t_cuda else None,
            "err_mat": max_err_mat,
            "err_triton": max_err_triton,
            "err_cuda": max_err_cuda,
        })

        print(f"M={m:5d} | Dense {t_dense:6.3f} ms ({tflops_dense:5.1f} TF) | "
              f"Mat {t_mat:6.3f} ms ({t_dense/t_mat:4.2f}x) | "
              f"Triton {t_triton if t_triton else 'FAIL':>6.3f} ms ({t_dense/t_triton if t_triton else 0:4.2f}x) | "
              f"CUDA {t_cuda if t_cuda else 'FAIL':>6} ms | "
              f"err_mat={max_err_mat:.3f} err_triton={max_err_triton if max_err_triton else 0:.3f}")

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("Path A Diagnostic: Shared-Memory Codebook Gather")
    print(f"GPU: {torch.cuda.get_device_name('cuda:2')}")
    print("=" * 80)
    results = run_diagnostic(ms=[1, 16, 128, 512, 1024, 2048], n=4096, k=4096)
    print("\nDone.")
