# M144 / Track 7: Graph-WAL Probe

**Date:** 2026-04-20
**Status:** ❌ Negative result
**Goal:** Test if graph-based Fourier transform improves WAL encode/decode.

## Hypothesis

> Hidden dimensions may not have natural ordering. Graph Fourier (via channel similarity graph) may be more appropriate than ordinary FFT/DCT.

## Method

```
For each layer:
  1. Build channel similarity graph (cosine similarity of rows/columns)
  2. Compute graph Laplacian: L = D - A
  3. Eigenvectors of L = graph Fourier basis
  4. Project weight matrix: W_gft = U^T @ W @ U
  5. Uniform quantize in graph spectral space
  6. Reconstruct and compare
```

**Transforms tested:**
- Raw (baseline)
- DCT2
- RandOrth
- GraphRow (graph Fourier on rows only)
- Graph2D (graph Fourier on both rows and columns)

## Results

### Aggregate

| Transform | Avg MSE | Best Count | vs Raw |
|-----------|---------|------------|--------|
| **RandOrth** | **0.00000006** | 4 | **24× better** |
| DCT2 | 0.00000025 | 1 | 6× better |
| Raw | 0.00000142 | 0 | 1× |
| **GraphRow** | **0.00001016** | 0 | **7× worse** |
| **Graph2D** | **0.00015700** | 0 | **110× worse** |

**Graph-WAL wins: 0/5 layers**

### Per Layer

| Layer | Best | GraphRow MSE | Graph2D MSE |
|-------|------|-------------|-------------|
| q_proj | RandOrth | 0.000014 | 0.000218 |
| k_proj | RandOrth | 0.000003 | 0.000062 |
| v_proj | DCT2 | 0.00000007 | 0.000003 |
| o_proj | RandOrth | 0.00000045 | 0.000002 |
| gate_proj | RandOrth | 0.000007 | 0.000035 |

## Analysis

### Why Graph-WAL Fails

1. **Graph basis is data-dependent:** Eigenvectors of the Laplacian depend on the specific weight matrix. Small quantization errors in spectral space create large errors when projected back through the graph basis.

2. **Error propagation:** In graph Fourier space, each quantized coefficient contributes to **all vertices** of the graph. Local quantization errors become global reconstruction errors.

3. **Noisy graph structure:** K-nearest-neighbors graph built on cosine similarity is sensitive to noise. The graph structure itself is not stable under quantization.

4. **Graph2D compounds the problem:** Using two graphs (rows + columns) multiplies the error propagation. MSE is 110× worse than Raw.

### Why RandOrth Still Wins

Random Orthogonal transform:
- Is **not data-dependent** — same random matrix works for all layers
- Spreads quantization error **uniformly** across all outputs
- Does not create error propagation patterns

### Comparison with Track 5 (Transform-WAL)

| Transform Type | Result | Reason |
|---------------|--------|--------|
| Fixed basis (DCT, RandOrth) | ✅ Works | Stable, uniform error |
| Data-dependent graph basis | ❌ Fails | Error propagation, noisy |

## Conclusion

**Graph-WAL does not improve over ordinary transforms.**

The hypothesis that "hidden dimensions lack natural order, so graph Fourier is better" is **rejected**. For LLM weight matrices:
- Fixed orthogonal transforms (RandOrth) work best
- Data-dependent graph transforms amplify quantization errors

## Implications

For WAL v2, the optimal pipeline is:
```
W → Random Orthogonal Transform → Scalar WAL Quantize → Inverse Transform
```

Not:
```
W → Graph Fourier → Quantize → Inverse Graph Fourier ❌
```

## Artifacts

- `experiments/m144_graph_wal_probe.py`
- `experiments/m144_graph_wal_probe.json`
