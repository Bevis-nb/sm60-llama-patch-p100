#pragma once

#include "common.cuh"

// Q4_1 GEMV specialised for sm_60 (GP100): no DP4A, but full-rate HFMA2.
// Returns false when the shape or type does not qualify; the caller then falls back to plain MMVQ.
bool ggml_cuda_mmvq_f16_sm60_supported(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst,
    const void * fusion);

void ggml_cuda_mmvq_f16_sm60(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
