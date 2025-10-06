#!/usr/bin/env python3
"""
CUDA Kernel Profiling Script for LayerSkip-Llama3.2-1B
This script runs the inference with various profiling tools and analyzes kernel performance.
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path

def run_command(cmd, cwd=None, timeout=300):
    """Run a command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, 
                               text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

def check_nvcc_profiling_tools():
    """Check if NVIDIA profiling tools are available."""
    tools = {
        'nvcc': 'CUDA Compiler',
        'nsight': 'NVIDIA Nsight Systems',
        'ncu': 'NVIDIA Nsight Compute',
        'nvprof': 'NVIDIA Profiler (legacy)'
    }
    
    available = {}
    for tool, desc in tools.items():
        ret, _, _ = run_command(f"{tool} --version", timeout=10)
        available[tool] = ret == 0
    
    print("🔍 NVIDIA Profiling Tools Status:")
    for tool, desc in tools.items():
        status = "✅ Available" if available[tool] else "❌ Not Available"
        print(f"  {tool:12} ({desc:25}) - {status}")
    
    return available

def build_with_profiling():
    """Build the project with profiling flags."""
    print("\n🔨 Building with profiling flags...")
    
    # Clean build first
    run_command("cmake --build . --config Release --target clean", cwd="build")
    
    # Build with profiling flags
    build_cmd = """
    cmake --build . --config Release -- 
    -DCMAKE_CUDA_FLAGS="--generate-line-info --ptxas-options=-v --ptxas-options=-lineinfo"
    """
    
    ret, stdout, stderr = run_command(build_cmd, cwd="build")
    
    if ret == 0:
        print("✅ Build successful with profiling flags")
        return True
    else:
        print("❌ Build failed:")
        print(stderr)
        return False

def run_basic_profiling():
    """Run basic profiling with the built executable."""
    print("\n📊 Running basic profiling...")
    
    # Run the executable and capture output
    ret, stdout, stderr = run_command("..\\build\\Release\\layerskip_llama32.exe", 
                                     cwd=".", timeout=60)
    
    if ret == 0:
        print("✅ Basic profiling completed")
        
        # Extract performance metrics
        lines = stdout.split('\n')
        metrics = {}
        
        for line in lines:
            if "TTFT:" in line:
                metrics['ttft'] = float(line.split(':')[1].strip().split()[0])
            elif "Average:" in line and "TPOT" in stdout:
                metrics['avg_tpot'] = float(line.split(':')[1].strip().split()[0])
            elif "Throughput:" in line:
                metrics['throughput'] = float(line.split(':')[1].strip().split()[0])
        
        return metrics, stdout
    else:
        print("❌ Basic profiling failed:")
        print(stderr)
        return None, stderr

def run_nvprof_profiling():
    """Run profiling with nvprof (legacy NVIDIA profiler)."""
    print("\n🔬 Running nvprof profiling...")
    
    # nvprof command with detailed metrics
    nvprof_cmd = """
    nvprof --print-gpu-trace --print-api-trace --print-gpu-summary 
    --log-file nvprof_output.log --csv 
    ..\\build\\Release\\layerskip_llama32.exe
    """
    
    ret, stdout, stderr = run_command(nvprof_cmd, timeout=120)
    
    if ret == 0:
        print("✅ nvprof profiling completed")
        
        # Parse nvprof output
        if os.path.exists("nvprof_output.log"):
            with open("nvprof_output.log", 'r') as f:
                nvprof_data = f.read()
            return nvprof_data
    else:
        print("❌ nvprof profiling failed:")
        print(stderr)
    
    return None

def run_nsight_compute_profiling():
    """Run profiling with NVIDIA Nsight Compute."""
    print("\n🔬 Running Nsight Compute profiling...")
    
    # ncu command for detailed kernel analysis
    ncu_cmd = """
    ncu --set full --target-processes all --kernel-regex ".*" 
    --launch-skip 0 --launch-count 1 --call-stack 
    --export ncu_profile.ncu-rep 
    ..\\build\\Release\\layerskip_llama32.exe
    """
    
    ret, stdout, stderr = run_command(ncu_cmd, timeout=180)
    
    if ret == 0:
        print("✅ Nsight Compute profiling completed")
        return True
    else:
        print("❌ Nsight Compute profiling failed:")
        print(stderr)
        return False

def run_nsight_systems_profiling():
    """Run profiling with NVIDIA Nsight Systems."""
    print("\n🔬 Running Nsight Systems profiling...")
    
    # nsys command for timeline analysis
    nsys_cmd = """
    nsys profile --trace=cuda,nvtx --output=nsys_profile.nsys-rep 
    --force-overwrite=true 
    ..\\build\\Release\\layerskip_llama32.exe
    """
    
    ret, stdout, stderr = run_command(nsys_cmd, timeout=180)
    
    if ret == 0:
        print("✅ Nsight Systems profiling completed")
        return True
    else:
        print("❌ Nsight Systems profiling failed:")
        print(stderr)
        return False

def analyze_kernel_performance():
    """Analyze kernel performance from the profiling data."""
    print("\n📈 Kernel Performance Analysis:")
    print("=" * 60)
    
    # This would parse the profiling output and provide insights
    # For now, we'll provide a template for analysis
    
    analysis = {
        "kernel_breakdown": {
            "attention_qk_kernel": "QK attention computation - typically highest compute",
            "attention_v_kernel": "Value attention computation - memory intensive", 
            "softmax_kernel": "Softmax normalization - shared memory intensive",
            "rope_kernel": "Rotary position embedding - trigonometric operations",
            "swiglu_kernel": "SwiGLU activation - element-wise operations",
            "matmul_quantized": "Quantized matrix multiplication - main compute bottleneck",
            "matmul_fp16": "FP16 matrix multiplication - for cached embeddings"
        },
        "optimization_suggestions": [
            "Profile matmul_quantized kernels - likely the main bottleneck",
            "Check attention kernel occupancy and shared memory usage",
            "Analyze memory bandwidth utilization",
            "Consider kernel fusion opportunities",
            "Profile dequantization overhead"
        ]
    }
    
    for kernel, description in analysis["kernel_breakdown"].items():
        print(f"🔹 {kernel:20} - {description}")
    
    print("\n💡 Optimization Suggestions:")
    for suggestion in analysis["optimization_suggestions"]:
        print(f"   • {suggestion}")

def generate_profiling_report(metrics, stdout):
    """Generate a comprehensive profiling report."""
    print("\n📋 PROFILING REPORT")
    print("=" * 80)
    
    if metrics:
        print(f"📊 Performance Metrics:")
        print(f"   Time to First Token (TTFT): {metrics.get('ttft', 'N/A')} ms")
        print(f"   Average Time Per Token:     {metrics.get('avg_tpot', 'N/A')} ms")
        print(f"   Throughput:                {metrics.get('throughput', 'N/A')} tokens/sec")
    
    print(f"\n📁 Generated Files:")
    files = [
        "nvprof_output.log",
        "ncu_profile.ncu-rep", 
        "nsys_profile.nsys-rep"
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"   ✅ {file:20} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file:20} (not generated)")
    
    print(f"\n🔧 Next Steps:")
    print(f"   1. Open ncu_profile.ncu-rep in NVIDIA Nsight Compute for detailed kernel analysis")
    print(f"   2. Open nsys_profile.nsys-rep in NVIDIA Nsight Systems for timeline analysis")
    print(f"   3. Analyze nvprof_output.log for API trace and GPU metrics")
    print(f"   4. Use the insights to optimize kernel performance")

def main():
    """Main profiling workflow."""
    print("🚀 CUDA Kernel Profiling for LayerSkip-Llama3.2-1B")
    print("=" * 60)
    
    # Check available tools
    tools = check_nvcc_profiling_tools()
    
    # Build with profiling flags
    if not build_with_profiling():
        print("❌ Cannot proceed without successful build")
        return 1
    
    # Run basic profiling
    metrics, stdout = run_basic_profiling()
    if metrics is None:
        print("❌ Cannot proceed without basic profiling")
        return 1
    
    # Run advanced profiling tools
    if tools.get('nvprof', False):
        nvprof_data = run_nvprof_profiling()
    
    if tools.get('ncu', False):
        run_nsight_compute_profiling()
    
    if tools.get('nsight', False):
        run_nsight_systems_profiling()
    
    # Analyze results
    analyze_kernel_performance()
    
    # Generate report
    generate_profiling_report(metrics, stdout)
    
    print("\n✅ Profiling completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
