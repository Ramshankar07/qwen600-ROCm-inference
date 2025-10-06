// layerskip_model.cuh - CUDA kernels for LayerSkip-Llama3.2-1B

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include "config.h"
#include "layerskip_loader.h"
#include "profiling.cuh"

#define EXIT_SUCCESS 0
constexpr int THREADS_PER_BLOCK = 256;

using TransformerWeights = layerskip_loader::LayerSkipWeights;

// ================================================================
// Run State
// ================================================================

typedef struct {
    half *x;       // activation (DIM,)
    half *xb;      // buffer for residual (DIM,)
    half *xb2;     // additional buffer (DIM,)
    half *hb;      // hidden FFN buffer (HIDDEN_DIM,)
    half *hb2;     // hidden FFN buffer (HIDDEN_DIM,)
    half *q;       // query buffer (Q_DIM,)

    float *att;    // attention scores (N_HEADS, SEQ_LEN)
    half *logits;  // output logits (VOCAB_SIZE,)

    // KV cache
    half* key_cache;   // (N_LAYERS, SEQ_LEN, KV_DIM)
    half* value_cache; // (N_LAYERS, SEQ_LEN, KV_DIM)
    
    float* d_logits_fp32; // FP32 logits buffer
    
    // Cached dequantized embeddings (for tied weights)
    half* dequant_embeddings;  // (VOCAB_SIZE, DIM) in FP16
    bool embeddings_cached;
} RunState;

typedef struct {
    TransformerWeights weights;
    RunState state;
    cublasHandle_t cublas_handle;
    float* h_logits; // Host-side logits
} Transformer;

// ================================================================
// Memory Allocation
// ================================================================

void malloc_run_state(RunState* s) {
    cudaMalloc(&s->x, DIM * sizeof(half));
    cudaMalloc(&s->xb, DIM * sizeof(half));
    cudaMalloc(&s->xb2, DIM * sizeof(half));
    cudaMalloc(&s->hb, HIDDEN_DIM * sizeof(half));
    cudaMalloc(&s->hb2, HIDDEN_DIM * sizeof(half));
    cudaMalloc(&s->q, Q_DIM * sizeof(half));
    
    cudaMalloc(&s->att, (size_t)N_HEADS * SEQ_LEN * sizeof(float));
    cudaMalloc(&s->logits, VOCAB_SIZE * sizeof(half));
    cudaMalloc(&s->key_cache, (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(half));
    cudaMalloc(&s->value_cache, (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(half));
    cudaMalloc(&s->d_logits_fp32, VOCAB_SIZE * sizeof(float));
    
    // Initialize cached embeddings as null
    s->dequant_embeddings = nullptr;
    s->embeddings_cached = false;
}

void build_transformer(Transformer* t, const char* checkpoint_path) {
    layerskip_loader::load_layerskip_weights(checkpoint_path, t->weights);
    malloc_run_state(&t->state);
    cudaMallocHost((void**)&t->h_logits, VOCAB_SIZE * sizeof(float));
    cublasCreate(&t->cublas_handle);
    
    // Cache dequantized embeddings if they're quantized
    if (t->weights.embed_type != layerskip_loader::GGML_TYPE_F16) {
        std::cout << "Caching dequantized embeddings (one-time cost)..." << std::endl;
        
        size_t embed_size = (size_t)VOCAB_SIZE * DIM;
        cudaMalloc(&t->state.dequant_embeddings, embed_size * sizeof(half));
        
        // Dequantize entire embedding matrix once
        dequantize_row(t->weights.token_embedding_table, 
                      t->state.dequant_embeddings,
                      embed_size,
                      t->weights.embed_type);
        
        t->state.embeddings_cached = true;
        
        float size_mb = (embed_size * sizeof(half)) / (1024.0f * 1024.0f);
        std::cout << "Cached " << size_mb << " MB of dequantized embeddings" << std::endl;
    } else {
        // Already FP16, no need to cache
        t->state.dequant_embeddings = (half*)t->weights.token_embedding_table;
        t->state.embeddings_cached = false;
    }
}

void free_transformer(Transformer* t) {
    cudaFree(t->state.x);
    cudaFree(t->state.xb);
    cudaFree(t->state.xb2);
    cudaFree(t->state.hb);
    cudaFree(t->state.hb2);
    cudaFree(t->state.q);
    cudaFree(t->state.att);
    cudaFree(t->state.logits);
    cudaFree(t->state.key_cache);
    cudaFree(t->state.value_cache);
    cudaFree(t->state.d_logits_fp32);
    
    // Free cached embeddings if we allocated them
    if (t->state.embeddings_cached && t->state.dequant_embeddings) {
        cudaFree(t->state.dequant_embeddings);
    }
    
    cudaFreeHost(t->h_logits);
    cublasDestroy(t->cublas_handle);
}

// ================================================================
// RMSNorm Kernel
// ================================================================

template <int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
rms_norm_kernel(
    half* __restrict__ Y,
    const half* __restrict__ X,
    const float* __restrict__ weight,
    size_t D)
{
    const int t_idx = threadIdx.x;
    const int vec_iters = D / 2;

    const half2* row_in = reinterpret_cast<const half2*>(X);
    half2* row_out = reinterpret_cast<half2*>(Y);

    float lsum = 0.0f;

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        half2 v_h2 = __ldg(&row_in[idx]);
        float2 v_fp32 = __half22float2(v_h2);
        lsum = __fmaf_rn(v_fp32.x, v_fp32.x, lsum);
        lsum = __fmaf_rn(v_fp32.y, v_fp32.y, lsum);
    }

    using BlockReduce = cub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    __shared__ float mul_val;
    if (t_idx == 0) {
        float val = __fmaf_rn(block_sum, INV_DIM, EPS);
        mul_val = rsqrtf(val);
    }
    __syncthreads();

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        half2 v_in_h2 = __ldg(&row_in[idx]);
        float2 v_in_fp32 = __half22float2(v_in_h2);
        
        float w1 = weight[idx * 2];
        float w2 = weight[idx * 2 + 1];

        v_in_fp32.x = (v_in_fp32.x * mul_val) * w1;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * w2;

        row_out[idx] = __float22half2_rn(v_in_fp32);
    }
}

void rmsnorm_gpu(half* o, const half* x, const float* weight, int dim) {
    rms_norm_kernel<THREADS_PER_BLOCK><<<1, THREADS_PER_BLOCK>>>(o, x, weight, dim);
}

// ================================================================
// RoPE (Rotary Position Embedding)
// ================================================================

__global__ void rope_kernel(half* q, half* k, int pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q_DIM / 2) return;

    int head_dim_idx = (i * 2) % HEAD_DIM;
    float freq = 1.0f / powf(ROPE_THETA, (float)head_dim_idx / (float)HEAD_DIM);
    float val = (float)pos * freq;
    float fcr, fci;
    sincosf(val, &fci, &fcr);

    // Rotate Q
    half2 q_val_h2 = reinterpret_cast<half2*>(q)[i];
    float2 q_val_fp32 = __half22float2(q_val_h2);
    float q0 = q_val_fp32.x * fcr - q_val_fp32.y * fci;
    float q1 = q_val_fp32.x * fci + q_val_fp32.y * fcr;
    reinterpret_cast<half2*>(q)[i] = __float22half2_rn(make_float2(q0, q1));

    if (i < KV_DIM / 2) {
        // Rotate K
        half2 k_val_h2 = reinterpret_cast<half2*>(k)[i];
        float2 k_val_fp32 = __half22float2(k_val_h2);
        float k0 = k_val_fp32.x * fcr - k_val_fp32.y * fci;
        float k1 = k_val_fp32.x * fci + k_val_fp32.y * fcr;
        reinterpret_cast<half2*>(k)[i] = __float22half2_rn(make_float2(k0, k1));
    }
}

void rope_gpu(half* q, half* k, int pos) {
    PROFILE_KERNEL("rope_kernel");
    int num_pairs = Q_DIM / 2;
    int grid_size = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rope_kernel<<<grid_size, THREADS_PER_BLOCK>>>(q, k, pos);
}

// ================================================================
// Softmax
// ================================================================

__global__ void softmax_kernel(float* att, int pos) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int len = pos + 1;

    float* scores = att + (size_t)h * SEQ_LEN;
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + block_size;

    // Find max
    float thread_max = -INFINITY;
    for (int i = tid; i < len; i += block_size) {
        if (scores[i] > thread_max) thread_max = scores[i];
    }
    s_max[tid] = thread_max;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && s_max[tid + s] > s_max[tid]) {
            s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }
    float max_val = s_max[0];

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < len; i += block_size) {
        scores[i] = expf(scores[i] - max_val);
        thread_sum += scores[i];
    }
    s_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float sum = s_sum[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < len; i += block_size) {
        scores[i] *= inv_sum;
    }
}

void softmax_gpu(float* att, int pos) {
    PROFILE_KERNEL("softmax_kernel");
    int len = pos + 1;
    int block_size = min(1024, len);
    size_t shared_mem = 2 * block_size * sizeof(float);
    softmax_kernel<<<N_HEADS, block_size, shared_mem>>>(att, pos);
}

// ================================================================
// Attention Kernels
// ================================================================

__global__ void attention_qk_kernel(
    float* att,
    const half* q,
    const half* k_cache,
    int pos)
{
    int h = blockIdx.x; 
    int t = threadIdx.x;
    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    if (t <= pos) {
        const half* q_head = q + h * HEAD_DIM;
        int kv_head_idx = h / kv_mul;
        const half* k_vec = k_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            half2 q_pair = reinterpret_cast<const half2*>(q_head)[i];
            half2 k_pair = reinterpret_cast<const half2*>(k_vec)[i];
            float2 q_vals = __half22float2(q_pair);
            float2 k_vals = __half22float2(k_pair);
            score = __fmaf_rn(q_vals.x, k_vals.x, score);
            score = __fmaf_rn(q_vals.y, k_vals.y, score);
        }

        score /= sqrtf((float)HEAD_DIM);
        att[(size_t)h * SEQ_LEN + t] = score;
    }
}

__global__ void attention_v_kernel(
    half* out,
    const float* att,
    const half* v_cache,
    int pos)
{
    int h = blockIdx.x;
    int i = threadIdx.x;
    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    half* out_head = out + (size_t)h * HEAD_DIM;
    const float* att_head = att + (size_t)h * SEQ_LEN;
    int kv_head_idx = h / kv_mul;

    float weighted_sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        const half* v_vec = v_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;
        weighted_sum = __fmaf_rn(att_head[t], __half2float(v_vec[i]), weighted_sum);
    }
    out_head[i] = __float2half_rn(weighted_sum);
}

void attention_gpu(RunState* s, int l, int pos) {
    PROFILE_KERNEL("attention_qk_kernel");
    half* layer_key_cache = s->key_cache + (size_t)l * SEQ_LEN * KV_DIM;
    half* layer_value_cache = s->value_cache + (size_t)l * SEQ_LEN * KV_DIM;

    int qk_threads = min(1024, pos + 1);
    attention_qk_kernel<<<N_HEADS, qk_threads>>>(s->att, s->q, layer_key_cache, pos);
    
    PROFILE_STOP_KERNEL("attention_qk_kernel");
    PROFILE_START_KERNEL("softmax_kernel");
    softmax_gpu(s->att, pos);
    
    PROFILE_STOP_KERNEL("softmax_kernel");
    PROFILE_START_KERNEL("attention_v_kernel");
    attention_v_kernel<<<N_HEADS, HEAD_DIM>>>(s->q, s->att, layer_value_cache, pos);
    PROFILE_STOP_KERNEL("attention_v_kernel");
}

// ================================================================
// SwiGLU Activation
// ================================================================

__global__ void swiglu_kernel(half* hb, const half* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = __half2float(hb[i]);
        float gate = __half2float(hb2[i]);
        float silu = val / (1.0f + expf(-val));
        hb[i] = __float2half_rn(silu * gate);
    }
}

void swiglu_gpu(half* hb, const half* hb2, int size) {
    PROFILE_KERNEL("swiglu_kernel");
    int grid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    swiglu_kernel<<<grid, THREADS_PER_BLOCK>>>(hb, hb2, size);
}

// ================================================================
// Matrix Multiplication with Dequantization
// ================================================================

// Direct FP16 matrix multiplication (for cached embeddings)
void matmul_fp16(
    cublasHandle_t handle,
    half* y,
    const half* W,
    const half* x,
    int m, int n,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    PROFILE_KERNEL("matmul_fp16");
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 m, 1, n,
                 &alpha,
                 W, CUDA_R_16F, n,
                 x, CUDA_R_16F, n,
                 &beta,
                 y, CUDA_R_16F, m,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Forward declaration of dequantization function
void dequant_matmul(
    cublasHandle_t handle,
    half* y,
    const void* W_quant,
    layerskip_loader::GGMLType quant_type,
    const half* x,
    int m, int n,
    float alpha,
    float beta);

void matmul_quantized(
    cublasHandle_t handle,
    half* y,
    const void* W_quant,
    layerskip_loader::GGMLType type,
    const half* x,
    int m, int n,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    dequant_matmul(handle, y, W_quant, type, x, m, n, alpha, beta);
}

// ================================================================
// Forward Pass (Stub)
// ================================================================

__global__ void convert_fp16_to_fp32(half* fp16_in, float* fp32_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fp32_out[i] = __half2float(fp16_in[i]);
}

float* forward(Transformer* transformer, int token, int pos) {
    RunState* s = &transformer->state;
    TransformerWeights* w = &transformer->weights;
    cublasHandle_t handle = transformer->cublas_handle;

    // Token embedding - use cached dequantized embeddings
    half* token_embed = s->dequant_embeddings + (size_t)token * DIM;
    cudaMemcpy(s->x, token_embed, DIM * sizeof(half), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < N_LAYERS; l++) {
        const auto& layer = w->layers[l];

        // Attention block
        rmsnorm_gpu(s->xb, s->x, layer.attention.attn_norm_weight, DIM);

        half* k_cache_pos = s->key_cache + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;
        half* v_cache_pos = s->value_cache + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;

        // QKV projections
        matmul_quantized(handle, s->q, layer.attention.q_proj_weight, 
                        layer.attention.q_type, s->xb, Q_DIM, DIM);
        matmul_quantized(handle, k_cache_pos, layer.attention.k_proj_weight,
                        layer.attention.k_type, s->xb, KV_DIM, DIM);
        matmul_quantized(handle, v_cache_pos, layer.attention.v_proj_weight,
                        layer.attention.v_type, s->xb, KV_DIM, DIM);

        // RoPE
        rope_gpu(s->q, k_cache_pos, pos);

        // Attention
        attention_gpu(s, l, pos);

        // Output projection with residual
        matmul_quantized(handle, s->x, layer.attention.o_proj_weight,
                        layer.attention.o_type, s->q, DIM, Q_DIM, 1.0f, 1.0f);

        // FFN block
        rmsnorm_gpu(s->xb, s->x, layer.ffn.ffn_norm_weight, DIM);

        matmul_quantized(handle, s->hb, layer.ffn.gate_proj_weight,
                        layer.ffn.gate_type, s->xb, HIDDEN_DIM, DIM);
        matmul_quantized(handle, s->hb2, layer.ffn.up_proj_weight,
                        layer.ffn.up_type, s->xb, HIDDEN_DIM, DIM);

        swiglu_gpu(s->hb, s->hb2, HIDDEN_DIM);

        matmul_quantized(handle, s->x, layer.ffn.down_proj_weight,
                        layer.ffn.down_type, s->hb, DIM, HIDDEN_DIM, 1.0f, 1.0f);
    }

    // Final norm and classifier
    rmsnorm_gpu(s->x, s->x, w->output_norm_weight, DIM);
    
    // Use cached embeddings for output projection (tied weights)
    // This is now a simple FP16 matmul instead of dequantizing every time
    matmul_fp16(handle, s->logits, s->dequant_embeddings, s->x, VOCAB_SIZE, DIM);

    // Convert to FP32 for sampling
    int grid = (VOCAB_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    convert_fp16_to_fp32<<<grid, THREADS_PER_BLOCK>>>(s->logits, s->d_logits_fp32, VOCAB_SIZE);

    cudaMemcpy(transformer->h_logits, s->d_logits_fp32, 
               VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    return transformer->h_logits;
}