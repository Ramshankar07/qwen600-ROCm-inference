// profiling.cuh - Profiling utilities for LayerSkip inference

#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

// CUDA event-based GPU timer
class CUDATimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CUDATimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event);
    }
    
    float stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }
};

// CPU timer using high-resolution clock
class CPUTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

// Kernel profiling statistics
struct KernelStats {
    std::string name;
    std::vector<float> times;
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;
    int call_count = 0;
    
    void add_time(float time_ms) {
        times.push_back(time_ms);
        total_time += time_ms;
        min_time = std::min(min_time, time_ms);
        max_time = std::max(max_time, time_ms);
        call_count++;
    }
    
    float avg_time() const {
        return call_count > 0 ? total_time / call_count : 0.0f;
    }
    
    float median_time() const {
        if (times.empty()) return 0.0f;
        std::vector<float> sorted_times = times;
        std::sort(sorted_times.begin(), sorted_times.end());
        size_t n = sorted_times.size();
        return n % 2 == 0 ? (sorted_times[n/2-1] + sorted_times[n/2]) / 2.0f : sorted_times[n/2];
    }
};

// Global profiling context
class ProfilingContext {
private:
    std::map<std::string, KernelStats> kernel_stats;
    CUDATimer cuda_timer;
    CPUTimer cpu_timer;
    
    // Generation metrics
    double ttft_ms = 0.0; // Time to first token
    std::vector<double> tpot_ms; // Time per output token
    double total_gen_time_ms = 0.0;
    int tokens_generated = 0;
    
public:
    static ProfilingContext& instance() {
        static ProfilingContext ctx;
        return ctx;
    }
    
    void start_kernel(const std::string& name) {
        cuda_timer.start();
    }
    
    void stop_kernel(const std::string& name) {
        float time_ms = cuda_timer.stop();
        kernel_stats[name].add_time(time_ms);
    }
    
    void record_ttft(double time_ms) {
        ttft_ms = time_ms;
    }
    
    void record_tpot(double time_ms) {
        tpot_ms.push_back(time_ms);
        tokens_generated++;
    }
    
    void record_total_time(double time_ms) {
        total_gen_time_ms = time_ms;
    }
    
    void print_summary() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "PERFORMANCE SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Generation metrics
        std::cout << "\n📊 Generation Metrics:" << std::endl;
        std::cout << "  Time to First Token (TTFT): " << ttft_ms << " ms" << std::endl;
        
        if (!tpot_ms.empty()) {
            double avg_tpot = std::accumulate(tpot_ms.begin(), tpot_ms.end(), 0.0) / tpot_ms.size();
            std::vector<double> sorted_tpot = tpot_ms;
            std::sort(sorted_tpot.begin(), sorted_tpot.end());
            double median_tpot = sorted_tpot[sorted_tpot.size() / 2];
            
            std::cout << "  Time Per Output Token (TPOT):" << std::endl;
            std::cout << "    Average: " << avg_tpot << " ms" << std::endl;
            std::cout << "    Median:  " << median_tpot << " ms" << std::endl;
            std::cout << "    Min:     " << *std::min_element(tpot_ms.begin(), tpot_ms.end()) << " ms" << std::endl;
            std::cout << "    Max:     " << *std::max_element(tpot_ms.begin(), tpot_ms.end()) << " ms" << std::endl;
        }
        
        std::cout << "  Total Generation Time: " << total_gen_time_ms << " ms" << std::endl;
        std::cout << "  Tokens Generated: " << tokens_generated << std::endl;
        
        if (total_gen_time_ms > 0) {
            double tokens_per_sec = (tokens_generated * 1000.0) / total_gen_time_ms;
            std::cout << "  Throughput: " << tokens_per_sec << " tokens/sec" << std::endl;
        }
        
        // Kernel breakdown
        if (!kernel_stats.empty()) {
            std::cout << "\n⚡ Kernel Performance Breakdown:" << std::endl;
            std::cout << std::string(80, '-') << std::endl;
            
            // Calculate total kernel time
            float total_kernel_time = 0.0f;
            for (const auto& [name, stats] : kernel_stats) {
                total_kernel_time += stats.total_time;
            }
            
            // Sort kernels by total time
            std::vector<std::pair<std::string, KernelStats>> sorted_kernels(
                kernel_stats.begin(), kernel_stats.end());
            std::sort(sorted_kernels.begin(), sorted_kernels.end(),
                     [](const auto& a, const auto& b) {
                         return a.second.total_time > b.second.total_time;
                     });
            
            printf("%-25s %8s %8s %8s %8s %8s %8s\n",
                   "Kernel", "Calls", "Total", "Avg", "Median", "Min", "Max");
            printf("%-25s %8s %8s %8s %8s %8s %8s\n",
                   "", "", "(ms)", "(ms)", "(ms)", "(ms)", "(ms)");
            std::cout << std::string(80, '-') << std::endl;
            
            for (const auto& [name, stats] : sorted_kernels) {
                float percent = (stats.total_time / total_kernel_time) * 100.0f;
                printf("%-25s %8d %7.2f %7.3f %7.3f %7.3f %7.3f  [%.1f%%]\n",
                       name.c_str(),
                       stats.call_count,
                       stats.total_time,
                       stats.avg_time(),
                       stats.median_time(),
                       stats.min_time,
                       stats.max_time,
                       percent);
            }
            
            std::cout << std::string(80, '-') << std::endl;
            printf("%-25s %8s %7.2f\n", "TOTAL", "", total_kernel_time);
        }
        
        std::cout << std::string(80, '=') << std::endl;
    }
    
    void reset() {
        kernel_stats.clear();
        tpot_ms.clear();
        ttft_ms = 0.0;
        total_gen_time_ms = 0.0;
        tokens_generated = 0;
    }
};

// RAII wrapper for automatic kernel timing
class ScopedKernelTimer {
private:
    std::string kernel_name;
    
public:
    ScopedKernelTimer(const std::string& name) : kernel_name(name) {
        ProfilingContext::instance().start_kernel(kernel_name);
    }
    
    ~ScopedKernelTimer() {
        ProfilingContext::instance().stop_kernel(kernel_name);
    }
};

// Macros for easy profiling
#define PROFILE_KERNEL(name) ScopedKernelTimer __timer__(name)
#define PROFILE_START_KERNEL(name) ProfilingContext::instance().start_kernel(name)
#define PROFILE_STOP_KERNEL(name) ProfilingContext::instance().stop_kernel(name)