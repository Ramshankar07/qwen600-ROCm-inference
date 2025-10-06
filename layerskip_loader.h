// layerskip_loader.h - Full GGUF parser with Windows support

#pragma once

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <map>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Cross-platform file handling
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

#include "config.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

namespace layerskip_loader
{

// GGUF constants
constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;
constexpr int QK_K = 256; // K-quant block size

// GGML quantization types
enum GGMLType {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
};

// Get size in bytes for quantization type (single definition)
inline size_t get_type_size(GGMLType type, size_t n_elements) {
    switch (type) {
        case GGML_TYPE_F32:  return n_elements * 4;
        case GGML_TYPE_F16:  return n_elements * 2;
        case GGML_TYPE_Q4_0: return (n_elements / 32) * 18;
        case GGML_TYPE_Q4_1: return (n_elements / 32) * 20;
        case GGML_TYPE_Q5_0: return (n_elements / 32) * 22;
        case GGML_TYPE_Q5_1: return (n_elements / 32) * 24;
        case GGML_TYPE_Q8_0: return (n_elements / 32) * 36;
        case GGML_TYPE_Q2_K: return (n_elements / QK_K) * (QK_K / 16 + QK_K / 4 + 2 + 2);
        case GGML_TYPE_Q3_K: return (n_elements / QK_K) * (QK_K / 8 + QK_K / 4 + 12 + 2);
        case GGML_TYPE_Q4_K: return (n_elements / QK_K) * (2 + 2 + 12 + QK_K / 2);
        case GGML_TYPE_Q5_K: return (n_elements / QK_K) * (2 + 2 + 12 + QK_K / 8 + QK_K / 2);
        case GGML_TYPE_Q6_K: return (n_elements / QK_K) * (QK_K / 2 + QK_K / 4 + QK_K / 16 + 2);
        default: return n_elements * 2; // fallback to FP16 size
    }
}

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dims;
    GGMLType type;
    uint64_t offset;
    size_t size_bytes;
};

struct AttentionWeights {
    void* q_proj_weight;
    void* k_proj_weight;
    void* v_proj_weight;
    void* o_proj_weight;
    float* attn_norm_weight;
    
    GGMLType q_type;
    GGMLType k_type;
    GGMLType v_type;
    GGMLType o_type;
};

struct FFNWeights {
    void* gate_proj_weight;
    void* up_proj_weight;
    void* down_proj_weight;
    float* ffn_norm_weight;
    
    GGMLType gate_type;
    GGMLType up_type;
    GGMLType down_type;
};

struct TransformerBlockWeights {
    AttentionWeights attention;
    FFNWeights ffn;
};

struct LayerSkipWeights {
    void* token_embedding_table;
    GGMLType embed_type;
    
    TransformerBlockWeights layers[N_LAYERS];
    
    float* output_norm_weight;
    void* output_head_weight;
    GGMLType output_type;
    
    void* _gpu_mem_block = nullptr;
    std::map<std::string, TensorInfo> tensor_map;
    
    ~LayerSkipWeights() {
        if (_gpu_mem_block) { 
            CUDA_CHECK(cudaFree(_gpu_mem_block)); 
        }
    }
};

// Memory-based reading helpers
template<typename T>
T read_value_mem(const char*& ptr) {
    T value;
    memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

std::string read_string_mem(const char*& ptr) {
    uint64_t len = read_value_mem<uint64_t>(ptr);
    std::string result(ptr, len);
    ptr += len;
    return result;
}

void skip_value(const char*& ptr, uint32_t value_type) {
    switch (value_type) {
        case 4: case 5: ptr += 4; break;  // UINT32, INT32
        case 6: ptr += 4; break;            // FLOAT32
        case 7: ptr += 1; break;            // BOOL
        case 8: {                           // STRING
            uint64_t str_len = read_value_mem<uint64_t>(ptr);
            ptr += str_len;
            break;
        }
        case 9: {                           // ARRAY
            uint32_t arr_type = read_value_mem<uint32_t>(ptr);
            uint64_t arr_len = read_value_mem<uint64_t>(ptr);
            for (uint64_t j = 0; j < arr_len; ++j) {
                skip_value(ptr, arr_type);
            }
            break;
        }
        case 10: ptr += 8; break;           // UINT64
        default: ptr += 8; break;
    }
}

void parse_gguf_metadata(
    const char* file_data,
    std::map<std::string, TensorInfo>& tensors,
    size_t& data_offset)
{
    const char* ptr = file_data;
    
    // Read header
    uint32_t magic = read_value_mem<uint32_t>(ptr);
    uint32_t version = read_value_mem<uint32_t>(ptr);
    uint64_t tensor_count = read_value_mem<uint64_t>(ptr);
    uint64_t kv_count = read_value_mem<uint64_t>(ptr);
    
    if (magic != GGUF_MAGIC) {
        throw std::runtime_error("Invalid GGUF magic number");
    }
    
    std::cout << "GGUF version: " << version << std::endl;
    std::cout << "Tensor count: " << tensor_count << std::endl;
    std::cout << "Metadata entries: " << kv_count << std::endl;
    
    // Skip key-value metadata
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = read_string_mem(ptr);
        uint32_t value_type = read_value_mem<uint32_t>(ptr);
        skip_value(ptr, value_type);
    }
    
    // Parse tensor metadata
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info;
        info.name = read_string_mem(ptr);
        info.n_dims = read_value_mem<uint32_t>(ptr);
        
        info.dims.resize(info.n_dims);
        for (uint32_t d = 0; d < info.n_dims; ++d) {
            info.dims[d] = read_value_mem<uint64_t>(ptr);
        }
        
        info.type = static_cast<GGMLType>(read_value_mem<uint32_t>(ptr));
        info.offset = read_value_mem<uint64_t>(ptr);
        
        // Calculate actual size
        uint64_t n_elements = 1;
        for (auto dim : info.dims) {
            n_elements *= dim;
        }
        info.size_bytes = get_type_size(info.type, n_elements);
        
        tensors[info.name] = info;
    }
    
    // Align to 32 bytes for data section
    size_t header_size = ptr - file_data;
    data_offset = (header_size + 31) & ~31ULL;
    
    std::cout << "Parsed " << tensors.size() << " tensors" << std::endl;
    std::cout << "Data starts at offset: " << data_offset << " bytes" << std::endl;
    
    // Debug: Print first few tensor names
    std::cout << "\nSample tensor names:" << std::endl;
    int count = 0;
    for (const auto& [name, info] : tensors) {
        std::cout << "  " << name << std::endl;
        if (++count >= 10) break;
    }
}

void load_layerskip_weights(
    const std::string& filepath,
    LayerSkipWeights& weights)
{
    std::cout << "Loading LayerSkip model from: " << filepath << std::endl;
    
    // Open and map file (platform-specific)
    char* mapped_file = nullptr;
    size_t file_size = 0;
    
#ifdef _WIN32
    HANDLE hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        throw std::runtime_error("Failed to get file size");
    }
    file_size = static_cast<size_t>(fileSize.QuadPart);
    
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        CloseHandle(hFile);
        throw std::runtime_error("Failed to create file mapping");
    }
    
    mapped_file = static_cast<char*>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
    if (!mapped_file) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        throw std::runtime_error("Failed to map file");
    }
    
    CloseHandle(hMapping);
    CloseHandle(hFile);
#else
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1) {
        close(fd);
        throw std::runtime_error("Failed to get file stats");
    }
    file_size = file_stat.st_size;
    
    mapped_file = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_file == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file");
    }
    close(fd);
#endif
    
    std::cout << "File size: " << (file_size / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Parse GGUF metadata
    size_t data_offset;
    parse_gguf_metadata(mapped_file, weights.tensor_map, data_offset);
    
    char* data_ptr = mapped_file + data_offset;
    
    // Calculate total GPU memory needed
    size_t total_bytes = 0;
    for (const auto& [name, info] : weights.tensor_map) {
        total_bytes += info.size_bytes;
    }
    
    std::cout << "Allocating " << (total_bytes / 1024.0 / 1024.0) << " MB on GPU" << std::endl;
    
    // Allocate single GPU memory block
    CUDA_CHECK(cudaMalloc(&weights._gpu_mem_block, total_bytes));
    char* gpu_ptr = (char*)weights._gpu_mem_block;
    
    // Copy all tensors to GPU
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    size_t current_offset = 0;
    for (auto& [name, info] : weights.tensor_map) {
        void* gpu_dest = gpu_ptr + current_offset;
        char* host_src = data_ptr + info.offset;
        
        CUDA_CHECK(cudaMemcpyAsync(gpu_dest, host_src, info.size_bytes,
                                    cudaMemcpyHostToDevice, stream));
        
        info.offset = current_offset; // Update to GPU offset
        current_offset += info.size_bytes;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    // Cleanup mapped file
#ifdef _WIN32
    UnmapViewOfFile(mapped_file);
#else
    munmap(mapped_file, file_size);
#endif
    
    std::cout << "All tensors copied to GPU" << std::endl;
    
    // Assign weight pointers
    auto get_tensor_ptr = [&](const std::string& name) -> void* {
        auto it = weights.tensor_map.find(name);
        if (it == weights.tensor_map.end()) {
            std::cerr << "ERROR: Required tensor not found: " << name << std::endl;
            throw std::runtime_error("Tensor not found: " + name);
        }
        return gpu_ptr + it->second.offset;
    };
    
    auto get_tensor_ptr_safe = [&](const std::string& name) -> void* {
        auto it = weights.tensor_map.find(name);
        return (it != weights.tensor_map.end()) ? (gpu_ptr + it->second.offset) : nullptr;
    };
    
    // Token embeddings
    weights.token_embedding_table = get_tensor_ptr("token_embd.weight");
    weights.embed_type = weights.tensor_map["token_embd.weight"].type;
    
    std::cout << "Token embedding type: " << (int)weights.embed_type << " (";
    switch(weights.embed_type) {
        case GGML_TYPE_F16: std::cout << "F16"; break;
        case GGML_TYPE_F32: std::cout << "F32"; break;
        case GGML_TYPE_Q2_K: std::cout << "Q2_K"; break;
        case GGML_TYPE_Q3_K: std::cout << "Q3_K"; break;
        case GGML_TYPE_Q4_K: std::cout << "Q4_K"; break;
        case GGML_TYPE_Q5_K: std::cout << "Q5_K"; break;
        case GGML_TYPE_Q6_K: std::cout << "Q6_K"; break;
        default: std::cout << "UNKNOWN"; break;
    }
    std::cout << ")" << std::endl;
    
    // Per-layer weights
    for (int i = 0; i < N_LAYERS; ++i) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        auto& layer = weights.layers[i];
        
        // Attention - exact names from GGUF
        layer.attention.q_proj_weight = get_tensor_ptr(prefix + "attn_q.weight");
        layer.attention.k_proj_weight = get_tensor_ptr(prefix + "attn_k.weight");
        layer.attention.v_proj_weight = get_tensor_ptr(prefix + "attn_v.weight");
        layer.attention.o_proj_weight = get_tensor_ptr(prefix + "attn_output.weight");
        layer.attention.attn_norm_weight = (float*)get_tensor_ptr(prefix + "attn_norm.weight");
        
        layer.attention.q_type = weights.tensor_map[prefix + "attn_q.weight"].type;
        layer.attention.k_type = weights.tensor_map[prefix + "attn_k.weight"].type;
        layer.attention.v_type = weights.tensor_map[prefix + "attn_v.weight"].type;
        layer.attention.o_type = weights.tensor_map[prefix + "attn_output.weight"].type;
        
        // FFN - exact names from GGUF
        layer.ffn.gate_proj_weight = get_tensor_ptr(prefix + "ffn_gate.weight");
        layer.ffn.up_proj_weight = get_tensor_ptr(prefix + "ffn_up.weight");
        layer.ffn.down_proj_weight = get_tensor_ptr(prefix + "ffn_down.weight");
        layer.ffn.ffn_norm_weight = (float*)get_tensor_ptr(prefix + "ffn_norm.weight");
        
        layer.ffn.gate_type = weights.tensor_map[prefix + "ffn_gate.weight"].type;
        layer.ffn.up_type = weights.tensor_map[prefix + "ffn_up.weight"].type;
        layer.ffn.down_type = weights.tensor_map[prefix + "ffn_down.weight"].type;
    }
    
    // Output layer - exact names from GGUF
    weights.output_norm_weight = (float*)get_tensor_ptr("output_norm.weight");
    
    // No separate output head in this model - uses tied embeddings
    weights.output_head_weight = weights.token_embedding_table;
    weights.output_type = weights.embed_type;
    
    std::cout << "Successfully loaded LayerSkip-Llama3.2-1B!" << std::endl;
    std::cout << "Note: Using tied embeddings (no separate output head)" << std::endl;
}

} // namespace layerskip_loader