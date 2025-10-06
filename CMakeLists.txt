cmake_minimum_required(VERSION 3.20)

project(LAYERSKIP_LLAMA32_1B)

# Enable CUDA language support
enable_language(CUDA)
enable_language(CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures:
# 75=Turing (RTX 20xx, T4)
# 80=Ampere (A100)
# 86=Ampere (RTX 30xx, A10, A30, A40)
# 89=Ada Lovelace (RTX 40xx, L4, L40)
# 90=Hopper (H100, H200)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "========================================")

# Source files
set(CUDA_SOURCES
    main.cu
)

set(CPP_SOURCES
    tokenizer.cpp
)

# Main executable
add_executable(
    layerskip_llama32
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Explicitly set language for each source
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CXX)

# Link libraries
target_link_libraries(
    layerskip_llama32
    PRIVATE
    CUDA::cublas
    CUDA::cudart
)

# Profiling option
option(PROFILING "Enable CUDA profiling flags" OFF)

# CUDA compile options
target_compile_options(layerskip_llama32 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        --extra-device-vectorization
    >
)

# Add profiling flags if enabled
if(PROFILING)
    message(STATUS "Enabling CUDA profiling flags")
    target_compile_options(layerskip_llama32 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --ptxas-options=-v
            --ptxas-options=-lineinfo
            --ptxas-options=-warn-lmem-usage
            --ptxas-options=-warn-spills
        >
    )
endif()

# Optimization flags
set_target_properties(layerskip_llama32 PROPERTIES
    COMPILE_FLAGS "-O3"
)

# Add include directories
target_include_directories(layerskip_llama32 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Disable warnings for both C++ and CUDA
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(layerskip_llama32 PRIVATE -w)
endif()

if (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(layerskip_llama32 PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-w>
    )
endif()

# Optional: Add debug build type with debug symbols
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(layerskip_llama32 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -G          # Device debug symbols
            -lineinfo   # Line number info
        >
    )
    message(STATUS "Debug build enabled with CUDA debug symbols")
endif()

# Optional: Print final configuration
message(STATUS "========================================")
message(STATUS "Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "========================================")

# Installation rules (optional)
install(TARGETS layerskip_llama32
    RUNTIME DESTINATION bin
)