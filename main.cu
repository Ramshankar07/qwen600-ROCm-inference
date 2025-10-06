// main.cu - LayerSkip inference with profiling

#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "config.h"
#include "layerskip_loader.h"
#include "gguf_dequant.cuh"
#include "layerskip_model.cuh"
#include "profiling.cuh"
#include "tokenizer.h"

// Simple argmax sampling
int sample_argmax(float* logits, int vocab_size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Temperature sampling
int sample_temperature(float* logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) {
        return sample_argmax(logits, vocab_size);
    }
    
    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    // Softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    
    // Sample
    float r = ((float)rand() / RAND_MAX) * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum > r) {
            return i;
        }
    }
    return vocab_size - 1;
}

// Generate tokens with profiling
void generate(Transformer* transformer, int* tokens, int num_prompt_tokens, int max_new_tokens, float temperature = 0.8f) {
    std::cout << "\nGenerating " << max_new_tokens << " tokens..." << std::endl;
    std::cout << "Starting with token: " << tokens[0] << std::endl;
    
    CPUTimer gen_timer;
    CPUTimer token_timer;
    gen_timer.start();
    
    for (int pos = 0; pos < num_prompt_tokens + max_new_tokens; pos++) {
        token_timer.start();
        
        // Forward pass
        float* logits = forward(transformer, tokens[pos], pos);
        
        double token_time = token_timer.stop();
        
        // Record timing
        if (pos == 0) {
            ProfilingContext::instance().record_ttft(token_time);
            std::cout << "TTFT: " << token_time << " ms" << std::endl;
        } else if (pos >= num_prompt_tokens) {
            ProfilingContext::instance().record_tpot(token_time);
        }
        
        // Sample next token (if not in prompt)
        if (pos >= num_prompt_tokens - 1) {
            int next_token = sample_temperature(logits, VOCAB_SIZE, temperature);
            if (pos + 1 < num_prompt_tokens + max_new_tokens) {
                tokens[pos + 1] = next_token;
            }
            std::cout << next_token << " " << std::flush;
            
            // Stop if we hit EOS
            if (next_token == 128009) { // Llama 3.2 EOS
                std::cout << "\n[EOS]" << std::endl;
                break;
            }
        }
    }
    
    double total_time = gen_timer.stop();
    ProfilingContext::instance().record_total_time(total_time);
    
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Use the specified model path directly
    const char* model_path = "D:\\qwen-rocm\\qwen600\\models\\facebook.layerskip-llama3.2-1B.Q2_K.gguf";
    const char* tokenizer_path = "D:\\qwen-rocm\\qwen600\\tokenizer.bin";
    
    std::cout << "=== LayerSkip-Llama3.2-1B Inference ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Tokenizer: " << tokenizer_path << std::endl;
    
    try {
        // Build transformer
        Transformer transformer;
        build_transformer(&transformer, model_path);
        
        std::cout << "\nModel loaded successfully!" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Layers: " << N_LAYERS << std::endl;
        std::cout << "  Hidden size: " << DIM << std::endl;
        std::cout << "  FFN size: " << HIDDEN_DIM << std::endl;
        std::cout << "  Heads: " << N_HEADS << std::endl;
        std::cout << "  KV heads: " << N_KV_HEADS << std::endl;
        std::cout << "  Vocab size: " << VOCAB_SIZE << std::endl;
        
        // Load tokenizer (stub for now)
        void* tokenizer = load_tokenizer(tokenizer_path);
        
        // Simple test: generate from BOS token with temperature sampling
        std::vector<int> tokens(100);
        tokens[0] = 128000; // Llama 3.2 BOS token
        
        
        std::cout << "\nRunning inference test with temperature=0.8..." << std::endl;
        
        // Reset profiling context
        ProfilingContext::instance().reset();
        
        generate(&transformer, tokens.data(), 1, 50, 0.8f);
        
        // Print profiling summary
        ProfilingContext::instance().print_summary();
        
        // Cleanup
        free_transformer(&transformer);
        
        std::cout << "\nInference test complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}