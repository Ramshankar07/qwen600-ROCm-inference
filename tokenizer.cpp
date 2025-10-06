// tokenizer.cpp - Llama 3.2 tokenizer implementation

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <cstring>

class Tokenizer {
private:
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    int max_token_length;
    int vocab_size;
    int bos_token_id;
    int eos_token_id;

public:
    Tokenizer(const char* tokenizer_path) {
        std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
        
        std::ifstream file(tokenizer_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open tokenizer file");
        }
        
        // Read header
        file.read(reinterpret_cast<char*>(&max_token_length), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&bos_token_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&eos_token_id), sizeof(uint32_t));
        
        std::cout << "Tokenizer info:" << std::endl;
        std::cout << "  Max token length: " << max_token_length << std::endl;
        std::cout << "  Vocab size: " << vocab_size << std::endl;
        std::cout << "  BOS token ID: " << bos_token_id << std::endl;
        std::cout << "  EOS token ID: " << eos_token_id << std::endl;
        
        // Read tokens
        for (int token_id = 0; token_id < vocab_size; token_id++) {
            float score;
            uint32_t token_length;
            
            file.read(reinterpret_cast<char*>(&score), sizeof(float));
            file.read(reinterpret_cast<char*>(&token_length), sizeof(uint32_t));
            
            std::string token(token_length, '\0');
            file.read(&token[0], token_length);
            
            id_to_token[token_id] = token;
            token_to_id[token] = token_id;
        }
        
        file.close();
        std::cout << "Tokenizer loaded successfully!" << std::endl;
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        
        // Simple word-based encoding for now
        // In a real implementation, you'd use BPE or similar
        std::string current_word;
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!current_word.empty()) {
                    if (token_to_id.find(current_word) != token_to_id.end()) {
                        tokens.push_back(token_to_id[current_word]);
                    } else {
                        // Use space character as fallback
                        tokens.push_back(token_to_id[" "]);
                    }
                    current_word.clear();
                }
                if (c == ' ') {
                    tokens.push_back(token_to_id[" "]);
                }
            } else {
                current_word += c;
            }
        }
        
        if (!current_word.empty()) {
            if (token_to_id.find(current_word) != token_to_id.end()) {
                tokens.push_back(token_to_id[current_word]);
            } else {
                tokens.push_back(token_to_id[" "]);
            }
        }
        
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        
        for (int token_id : tokens) {
            if (id_to_token.find(token_id) != id_to_token.end()) {
                result += id_to_token[token_id];
            } else {
                result += "<unk>";
            }
        }
        
        return result;
    }
    
    int get_bos_token_id() const { return bos_token_id; }
    int get_eos_token_id() const { return eos_token_id; }
    int get_vocab_size() const { return vocab_size; }
};

// Global tokenizer instance
static Tokenizer* g_tokenizer = nullptr;

// C-style interface for main.cu
extern "C" {
    void* load_tokenizer(const char* path) {
        try {
            g_tokenizer = new Tokenizer(path);
            return g_tokenizer;
        } catch (const std::exception& e) {
            std::cerr << "Error loading tokenizer: " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    void free_tokenizer(void* tokenizer) {
        delete static_cast<Tokenizer*>(tokenizer);
        g_tokenizer = nullptr;
    }
    
    int* encode_text(void* tokenizer, const char* text, int* num_tokens) {
        if (!tokenizer) return nullptr;
        
        Tokenizer* tok = static_cast<Tokenizer*>(tokenizer);
        std::vector<int> tokens = tok->encode(std::string(text));
        
        *num_tokens = tokens.size();
        int* result = new int[tokens.size()];
        std::copy(tokens.begin(), tokens.end(), result);
        
        return result;
    }
    
    char* decode_tokens(void* tokenizer, const int* tokens, int num_tokens) {
        if (!tokenizer) return nullptr;
        
        Tokenizer* tok = static_cast<Tokenizer*>(tokenizer);
        std::vector<int> token_vec(tokens, tokens + num_tokens);
        std::string result = tok->decode(token_vec);
        
        char* result_cstr = new char[result.length() + 1];
        std::strcpy(result_cstr, result.c_str());
        
        return result_cstr;
    }
}