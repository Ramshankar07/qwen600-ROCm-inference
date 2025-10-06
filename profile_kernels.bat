@echo off
REM CUDA Kernel Profiling Script for LayerSkip-Llama3.2-1B
REM This script runs various NVIDIA profiling tools

echo ============================================================
echo CUDA Kernel Profiling for LayerSkip-Llama3.2-1B
echo ============================================================

REM Check if we're in the right directory
if not exist "main.cu" (
    echo Error: Please run this script from the project root directory
    pause
    exit /b 1
)

REM Build with profiling flags
echo.
echo Building with profiling flags...
cd build
cmake --build . --config Release --target clean
cmake --build . --config Release -- /p:CUDA_VERBOSE_PTXAS=1
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)
cd ..

echo Build successful with profiling flags!

REM Run basic profiling
echo.
echo Running basic profiling...
build\Release\layerskip_llama32.exe > basic_profile.txt 2>&1
if errorlevel 1 (
    echo Basic profiling failed!
    pause
    exit /b 1
)

echo Basic profiling completed!

REM Check for NVIDIA profiling tools
echo.
echo Checking NVIDIA profiling tools...

REM Try nvprof (legacy)
nvprof --version >nul 2>&1
if errorlevel 1 (
    echo nvprof not available
) else (
    echo Running nvprof profiling...
    nvprof --print-gpu-trace --print-gpu-summary --log-file nvprof_output.log --csv build\Release\layerskip_llama32.exe
    if errorlevel 1 (
        echo nvprof profiling failed
    ) else (
        echo nvprof profiling completed!
    )
)

REM Try NVIDIA Nsight Compute
ncu --version >nul 2>&1
if errorlevel 1 (
    echo NVIDIA Nsight Compute not available
) else (
    echo Running NVIDIA Nsight Compute profiling...
    ncu --set full --target-processes all --kernel-regex ".*" --launch-skip 0 --launch-count 1 --export ncu_profile.ncu-rep build\Release\layerskip_llama32.exe
    if errorlevel 1 (
        echo NVIDIA Nsight Compute profiling failed
    ) else (
        echo NVIDIA Nsight Compute profiling completed!
    )
)

REM Try NVIDIA Nsight Systems
nsys --version >nul 2>&1
if errorlevel 1 (
    echo NVIDIA Nsight Systems not available
) else (
    echo Running NVIDIA Nsight Systems profiling...
    nsys profile --trace=cuda,nvtx --output=nsys_profile.nsys-rep --force-overwrite=true build\Release\layerskip_llama32.exe
    if errorlevel 1 (
        echo NVIDIA Nsight Systems profiling failed
    ) else (
        echo NVIDIA Nsight Systems profiling completed!
    )
)

REM Display results
echo.
echo ============================================================
echo PROFILING RESULTS
echo ============================================================

echo.
echo Generated files:
if exist "basic_profile.txt" (
    echo   [OK] basic_profile.txt
) else (
    echo   [MISSING] basic_profile.txt
)

if exist "nvprof_output.log" (
    echo   [OK] nvprof_output.log
) else (
    echo   [MISSING] nvprof_output.log
)

if exist "ncu_profile.ncu-rep" (
    echo   [OK] ncu_profile.ncu-rep
) else (
    echo   [MISSING] ncu_profile.ncu-rep
)

if exist "nsys_profile.nsys-rep" (
    echo   [OK] nsys_profile.nsys-rep
) else (
    echo   [MISSING] nsys_profile.nsys-rep
)

echo.
echo Next steps:
echo   1. Open ncu_profile.ncu-rep in NVIDIA Nsight Compute for detailed kernel analysis
echo   2. Open nsys_profile.nsys-rep in NVIDIA Nsight Systems for timeline analysis  
echo   3. Analyze nvprof_output.log for API trace and GPU metrics
echo   4. Review basic_profile.txt for application-level performance metrics

echo.
echo Profiling completed!
pause
