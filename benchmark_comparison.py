#!/usr/bin/env python3
"""
Benchmarking Script: LayerSkip vs llama.cpp
Compare performance between custom LayerSkip implementation and reference llama.cpp
"""

import subprocess
import os
import sys
import time
import json
import csv
from pathlib import Path
import argparse

class BenchmarkRunner:
    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.results = {}
        
    def run_layerskip_benchmark(self, num_tokens=50, temperature=0.8):
        """Run benchmark on custom LayerSkip implementation."""
        print("🔬 Running LayerSkip benchmark...")
        
        # Build with profiling enabled
        build_cmd = "cd build && cmake .. -DPROFILING=ON && cmake --build . --config Release"
        ret, stdout, stderr = self.run_command(build_cmd)
        
        if ret != 0:
            print(f"❌ Build failed: {stderr}")
            return None
            
        # Run the benchmark
        run_cmd = f"build\\Release\\layerskip_llama32.exe"
        ret, stdout, stderr = self.run_command(run_cmd, timeout=120)
        
        if ret != 0:
            print(f"❌ LayerSkip benchmark failed: {stderr}")
            return None
            
        # Parse results
        metrics = self.parse_layerskip_output(stdout)
        metrics['implementation'] = 'LayerSkip'
        metrics['model_path'] = self.model_path
        
        return metrics
    
    def run_llamacpp_benchmark(self, num_tokens=50, temperature=0.8):
        """Run benchmark on llama.cpp reference implementation."""
        print("🔬 Running llama.cpp benchmark...")
        
        # Check if llama.cpp is available
        ret, _, _ = self.run_command("llama-cli --version", timeout=10)
        if ret != 0:
            print("❌ llama.cpp not found. Please ensure it's installed and in PATH.")
            return None
            
        # Run llama.cpp benchmark
        cmd = f"""llama-cli -m "{self.model_path}" -n {num_tokens} -t 4 --temp {temperature} --no-display-prompt"""
        
        ret, stdout, stderr = self.run_command(cmd, timeout=120)
        
        if ret != 0:
            print(f"❌ llama.cpp benchmark failed: {stderr}")
            return None
            
        # Parse results
        metrics = self.parse_llamacpp_output(stdout)
        metrics['implementation'] = 'llama.cpp'
        metrics['model_path'] = self.model_path
        
        return metrics
    
    def parse_layerskip_output(self, output):
        """Parse LayerSkip benchmark output."""
        lines = output.split('\n')
        metrics = {}
        
        for line in lines:
            if "Time to First Token (TTFT):" in line:
                metrics['ttft_ms'] = float(line.split(':')[1].strip().split()[0])
            elif "Average:" in line and "TPOT" in output:
                metrics['avg_tpot_ms'] = float(line.split(':')[1].strip().split()[0])
            elif "Throughput:" in line:
                metrics['throughput_tokens_per_sec'] = float(line.split(':')[1].strip().split()[0])
            elif "Total Generation Time:" in line:
                metrics['total_time_ms'] = float(line.split(':')[1].strip().split()[0])
            elif "Tokens Generated:" in line:
                metrics['tokens_generated'] = int(line.split(':')[1].strip().split()[0])
        
        return metrics
    
    def parse_llamacpp_output(self, output):
        """Parse llama.cpp benchmark output."""
        lines = output.split('\n')
        metrics = {}
        
        # Look for timing information in llama.cpp output
        for line in lines:
            if "llama_print_timings:" in line:
                # Parse timing information
                if "eval time" in line:
                    metrics['total_time_ms'] = float(line.split()[2]) * 1000  # Convert to ms
                elif "eval tokens" in line:
                    metrics['tokens_generated'] = int(line.split()[2])
                elif "tokens per second" in line:
                    metrics['throughput_tokens_per_sec'] = float(line.split()[3])
        
        # Calculate derived metrics
        if 'total_time_ms' in metrics and 'tokens_generated' in metrics:
            metrics['avg_tpot_ms'] = metrics['total_time_ms'] / metrics['tokens_generated']
            metrics['ttft_ms'] = metrics['avg_tpot_ms']  # Approximation for llama.cpp
        
        return metrics
    
    def run_command(self, cmd, timeout=60):
        """Run a command and return the result."""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, 
                                 text=True, timeout=timeout)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
    
    def compare_results(self, layerskip_results, llamacpp_results):
        """Compare benchmark results between implementations."""
        print("\n📊 PERFORMANCE COMPARISON")
        print("=" * 80)
        
        if not layerskip_results or not llamacpp_results:
            print("❌ Cannot compare - one or both benchmarks failed")
            return
        
        # Create comparison table
        metrics = [
            ('Time to First Token (TTFT)', 'ttft_ms', 'ms'),
            ('Average Time Per Token (TPOT)', 'avg_tpot_ms', 'ms'),
            ('Throughput', 'throughput_tokens_per_sec', 'tokens/sec'),
            ('Total Generation Time', 'total_time_ms', 'ms'),
            ('Tokens Generated', 'tokens_generated', 'tokens')
        ]
        
        print(f"{'Metric':<30} {'LayerSkip':<15} {'llama.cpp':<15} {'Speedup':<15}")
        print("-" * 80)
        
        for metric_name, key, unit in metrics:
            if key in layerskip_results and key in llamacpp_results:
                layerskip_val = layerskip_results[key]
                llamacpp_val = llamacpp_results[key]
                
                if key in ['throughput_tokens_per_sec']:
                    # Higher is better
                    speedup = layerskip_val / llamacpp_val if llamacpp_val > 0 else 0
                    speedup_str = f"{speedup:.2f}x"
                else:
                    # Lower is better
                    speedup = llamacpp_val / layerskip_val if layerskip_val > 0 else 0
                    speedup_str = f"{speedup:.2f}x"
                
                print(f"{metric_name:<30} {layerskip_val:<15.3f} {llamacpp_val:<15.3f} {speedup_str:<15}")
        
        # Performance analysis
        print("\n💡 PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if 'throughput_tokens_per_sec' in layerskip_results and 'throughput_tokens_per_sec' in llamacpp_results:
            throughput_ratio = layerskip_results['throughput_tokens_per_sec'] / llamacpp_results['throughput_tokens_per_sec']
            
            if throughput_ratio > 1.1:
                print("✅ LayerSkip is faster than llama.cpp")
            elif throughput_ratio < 0.9:
                print("⚠️  LayerSkip is slower than llama.cpp")
            else:
                print("⚖️  LayerSkip and llama.cpp have similar performance")
        
        # Memory usage comparison (if available)
        print("\n🔍 ADDITIONAL INSIGHTS")
        print("-" * 40)
        print("• LayerSkip uses custom CUDA kernels with detailed profiling")
        print("• llama.cpp uses optimized reference implementation")
        print("• Compare kernel-level performance in LayerSkip profiling output")
        print("• Consider memory usage and GPU utilization differences")
    
    def save_results(self, results, filename="benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to {filename}")
    
    def generate_report(self, results):
        """Generate a comprehensive benchmark report."""
        report = f"""
# LayerSkip vs llama.cpp Benchmark Report

## Test Configuration
- Model: {self.model_path}
- Tokenizer: {self.tokenizer_path}
- Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Results Summary

### LayerSkip Implementation
"""
        
        if 'layerskip' in results:
            layerskip = results['layerskip']
            report += f"""
- Time to First Token: {layerskip.get('ttft_ms', 'N/A')} ms
- Average Time Per Token: {layerskip.get('avg_tpot_ms', 'N/A')} ms  
- Throughput: {layerskip.get('throughput_tokens_per_sec', 'N/A')} tokens/sec
- Total Time: {layerskip.get('total_time_ms', 'N/A')} ms
- Tokens Generated: {layerskip.get('tokens_generated', 'N/A')}
"""
        
        report += """
### llama.cpp Reference Implementation
"""
        
        if 'llamacpp' in results:
            llamacpp = results['llamacpp']
            report += f"""
- Time to First Token: {llamacpp.get('ttft_ms', 'N/A')} ms
- Average Time Per Token: {llamacpp.get('avg_tpot_ms', 'N/A')} ms
- Throughput: {llamacpp.get('throughput_tokens_per_sec', 'N/A')} tokens/sec
- Total Time: {llamacpp.get('total_time_ms', 'N/A')} ms
- Tokens Generated: {llamacpp.get('tokens_generated', 'N/A')}
"""
        
        report += """
## Analysis

### Performance Comparison
The benchmark compares the custom LayerSkip CUDA implementation against the reference llama.cpp implementation.

### Key Metrics
- **TTFT (Time to First Token)**: Latency for generating the first token
- **TPOT (Time Per Output Token)**: Average time for each subsequent token
- **Throughput**: Tokens generated per second
- **Total Time**: Complete generation time

### Optimization Opportunities
Based on the profiling data, focus on:
1. Matrix multiplication kernels (matmul_fp16 dominates execution time)
2. Memory bandwidth utilization
3. Kernel fusion opportunities
4. Quantization overhead analysis

## Next Steps
1. Analyze detailed kernel profiling output
2. Use NVIDIA Nsight tools for deeper analysis
3. Consider kernel optimizations based on profiling data
4. Compare memory usage patterns
"""
        
        with open("benchmark_report.md", 'w') as f:
            f.write(report)
        
        print("📄 Benchmark report saved to benchmark_report.md")

def main():
    parser = argparse.ArgumentParser(description="Benchmark LayerSkip vs llama.cpp")
    parser.add_argument("--model", default="models/facebook.layerskip-llama3.2-1B.Q2_K.gguf",
                       help="Path to GGUF model file")
    parser.add_argument("--tokenizer", default="tokenizer.bin",
                       help="Path to tokenizer file")
    parser.add_argument("--tokens", type=int, default=50,
                       help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--skip-layerskip", action="store_true",
                       help="Skip LayerSkip benchmark")
    parser.add_argument("--skip-llamacpp", action="store_true",
                       help="Skip llama.cpp benchmark")
    
    args = parser.parse_args()
    
    print("🚀 LayerSkip vs llama.cpp Benchmark")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    if not os.path.exists(args.tokenizer):
        print(f"❌ Tokenizer file not found: {args.tokenizer}")
        return 1
    
    runner = BenchmarkRunner(args.model, args.tokenizer)
    results = {}
    
    # Run benchmarks
    if not args.skip_layerskip:
        layerskip_results = runner.run_layerskip_benchmark(args.tokens, args.temperature)
        if layerskip_results:
            results['layerskip'] = layerskip_results
    
    if not args.skip_llamacpp:
        llamacpp_results = runner.run_llamacpp_benchmark(args.tokens, args.temperature)
        if llamacpp_results:
            results['llamacpp'] = llamacpp_results
    
    # Compare results
    if 'layerskip' in results and 'llamacpp' in results:
        runner.compare_results(results['layerskip'], results['llamacpp'])
    
    # Save results
    runner.save_results(results)
    runner.generate_report(results)
    
    print("\n✅ Benchmark completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
