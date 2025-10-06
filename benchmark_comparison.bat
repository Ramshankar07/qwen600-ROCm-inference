@echo off
REM Benchmark Comparison: LayerSkip vs llama.cpp
REM This script compares performance between custom LayerSkip and reference llama.cpp

echo ============================================================
echo LayerSkip vs llama.cpp Performance Comparison
echo ============================================================

REM Check if we're in the right directory
if not exist "main.cu" (
    echo Error: Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "models\facebook.layerskip-llama3.2-1B.Q2_K.gguf" (
    echo Error: Model file not found: models\facebook.layerskip-llama3.2-1B.Q2_K.gguf
    pause
    exit /b 1
)

REM Check if tokenizer exists
if not exist "tokenizer.bin" (
    echo Error: Tokenizer file not found: tokenizer.bin
    pause
    exit /b 1
)

echo.
echo Running LayerSkip benchmark...
echo ============================================================

REM Build LayerSkip with profiling
cd build
cmake .. -DPROFILING=ON
cmake --build . --config Release
if errorlevel 1 (
    echo LayerSkip build failed!
    pause
    exit /b 1
)
cd ..

REM Run LayerSkip benchmark
echo Running LayerSkip inference...
build\Release\layerskip_llama32.exe > layerskip_benchmark.txt 2>&1
if errorlevel 1 (
    echo LayerSkip benchmark failed!
    pause
    exit /b 1
)

echo LayerSkip benchmark completed!

REM Check if llama.cpp is available
echo.
echo Checking llama.cpp availability...
llama-cli --version >nul 2>&1
if errorlevel 1 (
    echo Warning: llama.cpp not found in PATH
    echo Please ensure llama.cpp is installed and restart your shell
    echo Skipping llama.cpp benchmark...
    goto :compare_results
)

echo.
echo Running llama.cpp benchmark...
echo ============================================================

REM Run llama.cpp benchmark
echo Running llama.cpp inference...
llama-cli -m "models\facebook.layerskip-llama3.2-1B.Q2_K.gguf" -n 50 -t 4 --temp 0.8 --no-display-prompt > llamacpp_benchmark.txt 2>&1
if errorlevel 1 (
    echo llama.cpp benchmark failed!
    echo Check llamacpp_benchmark.txt for details
) else (
    echo llama.cpp benchmark completed!
)

:compare_results
echo.
echo ============================================================
echo PERFORMANCE COMPARISON RESULTS
echo ============================================================

echo.
echo LayerSkip Results:
echo ------------------
if exist "layerskip_benchmark.txt" (
    findstr /C:"Time to First Token" layerskip_benchmark.txt
    findstr /C:"Average:" layerskip_benchmark.txt
    findstr /C:"Throughput:" layerskip_benchmark.txt
    findstr /C:"Total Generation Time:" layerskip_benchmark.txt
    findstr /C:"Tokens Generated:" layerskip_benchmark.txt
) else (
    echo No LayerSkip results found
)

echo.
echo llama.cpp Results:
echo ------------------
if exist "llamacpp_benchmark.txt" (
    findstr /C:"llama_print_timings" llamacpp_benchmark.txt
    findstr /C:"eval time" llamacpp_benchmark.txt
    findstr /C:"eval tokens" llamacpp_benchmark.txt
    findstr /C:"tokens per second" llamacpp_benchmark.txt
) else (
    echo No llama.cpp results found
)

echo.
echo ============================================================
echo DETAILED ANALYSIS
echo ============================================================

echo.
echo LayerSkip Kernel Performance Breakdown:
if exist "layerskip_benchmark.txt" (
    findstr /C:"Kernel Performance Breakdown" -A 20 layerskip_benchmark.txt
) else (
    echo No kernel breakdown available
)

echo.
echo ============================================================
echo NEXT STEPS
echo ============================================================

echo.
echo Generated files:
if exist "layerskip_benchmark.txt" (
    echo   [OK] layerskip_benchmark.txt
) else (
    echo   [MISSING] layerskip_benchmark.txt
)

if exist "llamacpp_benchmark.txt" (
    echo   [OK] llamacpp_benchmark.txt
) else (
    echo   [MISSING] llamacpp_benchmark.txt
)

echo.
echo Analysis recommendations:
echo   1. Compare throughput (tokens/sec) between implementations
echo   2. Analyze LayerSkip kernel performance breakdown
echo   3. Check memory usage patterns
echo   4. Use NVIDIA Nsight tools for detailed kernel analysis
echo   5. Focus optimization on matmul_fp16 kernel (dominant bottleneck)

echo.
echo Benchmark comparison completed!
pause
