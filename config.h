// config.h - LayerSkip-Llama3.2-1B Configuration

#pragma once

#define MAX_LINE_WIDTH 80

constexpr int SEQ_LEN = 4096;              // Practical sequence length for inference
constexpr int PROMPT_BUFFER_SIZE = 16384;  // Buffer for prompt tokenization
constexpr int VOCAB_SIZE = 128256;         // Llama 3.2 vocabulary size

constexpr int DIM = 2048;                  // Hidden dimension / embedding dimension
constexpr int HIDDEN_DIM = 8192;           // FFN intermediate dimension
constexpr int N_LAYERS = 16;               // Number of transformer layers
constexpr int N_HEADS = 32;                // Number of attention heads
constexpr int N_KV_HEADS = 8;              // Number of key-value heads (GQA)
constexpr int HEAD_DIM = 64;               // Dimension per attention head

constexpr int Q_DIM = N_HEADS * HEAD_DIM;      // 32 * 64 = 2048
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;  // 8 * 64 = 512

constexpr float INV_HEAD_DIM = 1.0f / HEAD_DIM;  // 1/64 for attention scaling
constexpr float INV_DIM = 1.0f / DIM;            // 1/2048 for RMSNorm

constexpr float ROPE_THETA = 500000.0f;  // Llama 3.2 uses extended RoPE base

constexpr float EPS = 1e-5f;  

constexpr int MAX_CONTEXT_LENGTH = 131072;  

static_assert(Q_DIM == N_HEADS * HEAD_DIM, 
              "Q_DIM must equal N_HEADS * HEAD_DIM");
static_assert(KV_DIM == N_KV_HEADS * HEAD_DIM, 
              "KV_DIM must equal N_KV_HEADS * HEAD_DIM");
static_assert(N_HEADS % N_KV_HEADS == 0, 
              "N_HEADS must be divisible by N_KV_HEADS for GQA");
static_assert(DIM == Q_DIM, 
              "DIM should equal Q_DIM for this architecture");