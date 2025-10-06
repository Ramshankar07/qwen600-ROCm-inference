// tokenizer.h - Minimal header for compilation

#pragma once

#include <vector>
#include <string>

// Minimal tokenizer interface
class Tokenizer {
public:
    Tokenizer(const char* tokenizer_path);
    
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    
    int bos_token_id() const;
    int eos_token_id() const;
    int vocab_size() const;
};

// C-style interface for CUDA code
#ifdef __cplusplus
extern "C" {
#endif

void* load_tokenizer(const char* path);
void free_tokenizer(void* tokenizer);
int* encode_text(void* tokenizer, const char* text, int* num_tokens);
char* decode_tokens(void* tokenizer, const int* tokens, int num_tokens);

#ifdef __cplusplus
}
#endif