// gguf_dequant.cuh - Dequantization kernels for GGUF K-quants

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "layerskip_loader.h"

#define QK_K 256
#define K_SCALE_SIZE 12

// ================================================================
// K-quant block structures
// ================================================================

struct block_q2_K {
    uint8_t scales[QK_K/16];  // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];       // quants
    half d;                    // super-block scale for quantized scales
    half dmin;                 // super-block scale for quantized mins
};

struct block_q3_K {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[12];        // scales, quantized with 6 bits
    half d;                    // super-block scale
};

struct block_q4_K {
    half d;                    // super-block scale for quantized scales
    half dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
};

struct block_q5_K {
    half d;                    // super-block scale for quantized scales
    half dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];        // quants, high bit
    uint8_t qs[QK_K/2];        // quants, low 4 bits
};

struct block_q6_K {
    uint8_t ql[QK_K/2];        // quants, lower 4 bits
    uint8_t qh[QK_K/4];        // quants, upper 2 bits
    int8_t scales[QK_K/16];    // scales, quantized with 8 bits
    half d;                    // super-block scale
};

// ================================================================
// Q2_K Dequantization
// ================================================================

__global__ void dequantize_block_q2_K(const void* __restrict__ vx, half* __restrict__ y, int k) {
    const int i = blockIdx.x;
    const block_q2_K* x = (const block_q2_K*)vx;

    const int tid = threadIdx.x;
    const int n = tid / 32;
    const int l = tid - 32 * n;
    const int is = 8 * n;

    const uint8_t q = x[i].qs[32 * n + l];
    half* dst = y + i * QK_K + 128 * n;

    float dall = __half2float(x[i].d);
    float dmin = __half2float(x[i].dmin);

    dst[l +  0] = __float2half_rn(dall * (x[i].scales[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is + 0] >> 4));
    dst[l + 32] = __float2half_rn(dall * (x[i].scales[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is + 2] >> 4));
    dst[l + 64] = __float2half_rn(dall * (x[i].scales[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is + 4] >> 4));
    dst[l + 96] = __float2half_rn(dall * (x[i].scales[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is + 6] >> 4));
}

void dequantize_q2_K(const void* vx, half* y, int k) {
    int num_blocks = k / QK_K;
    dequantize_block_q2_K<<<num_blocks, 64>>>(vx, y, k);
}

// ================================================================
// Q3_K Dequantization
// ================================================================

__global__ void dequantize_block_q3_K(const void* __restrict__ vx, half* __restrict__ y, int k) {
    const int i = blockIdx.x;
    const block_q3_K* x = (const block_q3_K*)vx;

    const int r = threadIdx.x / 4;
    const int tid = r / 2;
    const int is0 = r % 2;
    const int l0 = 16 * is0 + 4 * (threadIdx.x % 4);
    const int n = tid / 4;
    const int j = tid - 4 * n;

    uint8_t m = 1 << (4 * n + j);
    int is = 8 * n + 2 * j + is0;

    const uint8_t* q = x[i].qs + 32 * n + l0;
    const uint8_t* hm = x[i].hmask;
    half* dst = y + i * QK_K + 128 * n + 32 * j;

    const float d = __half2float(x[i].d);
    const uint8_t sc = x[i].scales[is];

    for (int l = 0; l < 4; ++l) {
        uint8_t h = hm[l0 + l] & m ? 1 : 0;
        dst[l] = __float2half_rn(d * sc * ((int8_t)((q[l] >> 0) & 3) - ((h << 2) & 0x4 ? 0 : 4)));
    }
}

void dequantize_q3_K(const void* vx, half* y, int k) {
    int num_blocks = k / QK_K;
    dequantize_block_q3_K<<<num_blocks, 64>>>(vx, y, k);
}

// ================================================================
// Q4_K Dequantization
// ================================================================

__global__ void dequantize_block_q4_K(const void* __restrict__ vx, half* __restrict__ y, int k) {
    const block_q4_K* x = (const block_q4_K*)vx;

    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ir = tid % 8;
    const int is = 2 * il;
    const int n = 4;

    half* dst = y + i * QK_K + 64 * il + n * ir;

    const float dall = __half2float(x[i].d);
    const float dmin = __half2float(x[i].dmin);

    const uint8_t* q = x[i].qs + 32 * il + n * ir;

    uint8_t sc, m;
    sc = x[i].scales[is] & 0x3f;
    m = x[i].scales[is + 1] & 0x3f;

    const float d1 = dall * sc;
    const float m1 = dmin * m;

    for (int l = 0; l < n; ++l) {
        dst[l] = __float2half_rn(d1 * (q[l] & 0xF) - m1);
    }

    dst += 32;
    sc = (x[i].scales[is] >> 6) | ((x[i].scales[is + 1] >> 6) << 2);
    m = (x[i].scales[is + 2] >> 6) | ((x[i].scales[is + 3] >> 6) << 2);

    const float d2 = dall * sc;
    const float m2 = dmin * m;

    for (int l = 0; l < n; ++l) {
        dst[l] = __float2half_rn(d2 * (q[l] >> 4) - m2);
    }
}

void dequantize_q4_K(const void* vx, half* y, int k) {
    int num_blocks = k / QK_K;
    dequantize_block_q4_K<<<num_blocks, 32>>>(vx, y, k);
}

// ================================================================
// Q5_K Dequantization
// ================================================================

__global__ void dequantize_block_q5_K(const void* __restrict__ vx, half* __restrict__ y, int k) {
    const block_q5_K* x = (const block_q5_K*)vx;

    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il = tid / 16;
    const int ir = tid % 16;
    const int is = 2 * il;

    half* dst = y + i * QK_K + 64 * il + 2 * ir;

    const float dall = __half2float(x[i].d);
    const float dmin = __half2float(x[i].dmin);

    const uint8_t* ql = x[i].qs + 32 * il + 2 * ir;
    const uint8_t* qh = x[i].qh + 2 * ir;

    uint8_t sc, m;
    sc = x[i].scales[is] & 0x3f;
    m = x[i].scales[is + 1] & 0x3f;

    const float d1 = dall * sc;
    const float m1 = dmin * m;

    uint8_t hm = 1 << (2 * il);

    dst[0] = __float2half_rn(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
    dst[1] = __float2half_rn(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);

    hm <<= 1;
    dst[32] = __float2half_rn(d1 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m1);
    dst[33] = __float2half_rn(d1 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m1);
}

void dequantize_q5_K(const void* vx, half* y, int k) {
    int num_blocks = k / QK_K;
    dequantize_block_q5_K<<<num_blocks, 64>>>(vx, y, k);
}

// ================================================================
// Q6_K Dequantization
// ================================================================

__global__ void dequantize_block_q6_K(const void* __restrict__ vx, half* __restrict__ y, int k) {
    const block_q6_K* x = (const block_q6_K*)vx;

    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int ip = tid / 32;
    const int il = tid - 32 * ip;
    const int is = 8 * ip + il / 16;

    half* dst = y + i * QK_K + 128 * ip + il;

    const float d = __half2float(x[i].d);

    const uint8_t* ql = x[i].ql + 64 * ip + il;
    const uint8_t qh = x[i].qh[32 * ip + il];
    const int8_t* sc = x[i].scales + is;

    dst[0]  = __float2half_rn(d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    dst[32] = __float2half_rn(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    dst[64] = __float2half_rn(d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    dst[96] = __float2half_rn(d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

void dequantize_q6_K(const void* vx, half* y, int k) {
    int num_blocks = k / QK_K;
    dequantize_block_q6_K<<<num_blocks, 64>>>(vx, y, k);
}

// ================================================================
// Generic dequantization dispatcher
// ================================================================

void dequantize_row(
    const void* x,
    half* y,
    int k,
    layerskip_loader::GGMLType type)
{
    switch (type) {
        case layerskip_loader::GGML_TYPE_F16:
            cudaMemcpy(y, x, k * sizeof(half), cudaMemcpyDeviceToDevice);
            break;
        case layerskip_loader::GGML_TYPE_Q2_K:
            dequantize_q2_K(x, y, k);
            break;
        case layerskip_loader::GGML_TYPE_Q3_K:
            dequantize_q3_K(x, y, k);
            break;
        case layerskip_loader::GGML_TYPE_Q4_K:
            dequantize_q4_K(x, y, k);
            break;
        case layerskip_loader::GGML_TYPE_Q5_K:
            dequantize_q5_K(x, y, k);
            break;
        case layerskip_loader::GGML_TYPE_Q6_K:
            dequantize_q6_K(x, y, k);
            break;
        default:
            printf("Unsupported quantization type: %d\n", type);
            break;
    }
}

// ================================================================
// Fused dequantization + matrix multiplication
// ================================================================

void dequant_matmul(
    cublasHandle_t handle,
    half* y,
    const void* W_quant,
    layerskip_loader::GGMLType quant_type,
    const half* x,
    int m, int n,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    // Allocate temporary buffer for dequantized weights
    half* W_fp16;
    size_t W_size = (size_t)m * n * sizeof(half);
    cudaMalloc(&W_fp16, W_size);
    
    // Dequantize the entire weight matrix
    dequantize_row(W_quant, W_fp16, m * n, quant_type);
    
    // Perform FP16 matrix multiplication
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 m, 1, n,
                 &alpha,
                 W_fp16, CUDA_R_16F, n,
                 x, CUDA_R_16F, n,
                 &beta,
                 y, CUDA_R_16F, m,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaFree(W_fp16);
}