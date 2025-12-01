#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// ===================== CUDA KERNELS =====================

// Kernel 1: Partial sums for global average
__global__ void sumKernel(const double* temps, double* partialSums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    double local = 0.0;
    for (int i = idx; i < N; i += totalThreads) {
        local += temps[i];
    }
    if (idx < totalThreads) {
        partialSums[idx] = local;
    }
}

// Kernel 2: Moving average (7-day window)
__global__ void movingAvgKernel(const double* temps, double* movAvg, int N, int window) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || i < window - 1) return;

    double sum = 0.0;
    for (int j = i - (window - 1); j <= i; j++) {
        sum += temps[j];
    }
    movAvg[i] = sum / window;
}

// Kernel 3: Anomaly detection
__global__ void anomalyKernel(const double* temps, const double* movAvg,
                              int N, int window, double threshold,
                              int* partialCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    int localCount = 0;
    for (int i = idx; i < N; i += totalThreads) {
        if (i < window - 1) continue;
        double diff = temps[i] - movAvg[i];
        if (fabs(diff) > threshold) {
            localCount++;
        }
    }
    if (idx < totalThreads) {
        partialCounts[idx] = localCount;
    }
}

// ===================== HOST (CPU) CODE =====================

int main() {
    // ---------- 1. Load data from cleaned_temps.txt ----------
    ifstream file("cleaned_temps.txt");
    vector<double> h_temps;
    double v;
    while (file >> v) h_temps.push_back(v);
    file.close();

    int N = (int)h_temps.size();
    if (N == 0) {
        cout << "Error: cleaned_temps.txt is empty.\n";
        return 1;
    }

    const int WINDOW = 7;
    const double THRESHOLD = 5.0;


    cout << "Loaded Temperature Values : " << N << "\n\n";

    // ---------- 2. Allocate GPU memory ----------
    double* d_temps;
    double* d_movAvg;
    double* d_partialSums;
    int* d_partialCounts;

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    int totalThreads = blockSize * gridSize;

    cudaMalloc(&d_temps, N * sizeof(double));
    cudaMalloc(&d_movAvg, N * sizeof(double));
    cudaMalloc(&d_partialSums, totalThreads * sizeof(double));
    cudaMalloc(&d_partialCounts, totalThreads * sizeof(int));

    // Copy temperatures to device
    cudaMemcpy(d_temps, h_temps.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // For collecting results
    vector<double> h_partialSums(totalThreads);
    vector<int> h_partialCounts(totalThreads);
    vector<double> h_movAvg(N, 0.0);

    // CUDA timing (per operation)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sumKernel<<<gridSize, blockSize>>>(d_temps, d_partialSums, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_avg_ms = 0.0f;
    cudaEventElapsedTime(&time_avg_ms, start, stop); // milliseconds

    cudaMemcpy(h_partialSums.data(), d_partialSums,
               totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

    double totalSum = 0.0;
    for (int i = 0; i < totalThreads; i++)
        totalSum += h_partialSums[i];

    double avg = totalSum / N;
    double time_avg_us = time_avg_ms * 1000.0; // convert to microseconds

    cout << "GLOBAL AVERAGE: \n";
    cout << "Computed Average Temperature : " << avg << " °C\n";
    cout << "Execution Time               : " << time_avg_us << " µs\n\n";

    cudaEventRecord(start);

    movingAvgKernel<<<gridSize, blockSize>>>(d_temps, d_movAvg, N, WINDOW);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_mavg_ms = 0.0f;
    cudaEventElapsedTime(&time_mavg_ms, start, stop);
    double time_mavg_us = time_mavg_ms * 1000.0;

    // Copy moving average back to host so we can print a sample & use for debug
    cudaMemcpy(h_movAvg.data(), d_movAvg, N * sizeof(double), cudaMemcpyDeviceToHost);

    cout << "7-DAY MOVING AVERAGE\n";
    cout << "Window Size                  : " << WINDOW << " days\n";
    if (N > 100)/
        cout << "Sample Moving Avg [index 100]: " << h_movAvg[100] << " °C\n";
    else
        cout << "Sample Moving Avg [index 100]: N/A (N <= 100)\n";
    cout << "Execution Time               : " << time_mavg_us << " µs\n\n";

    cudaEventRecord(start);

    anomalyKernel<<<gridSize, blockSize>>>(d_temps, d_movAvg, N, WINDOW, THRESHOLD, d_partialCounts);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_anom_ms = 0.0f;
    cudaEventElapsedTime(&time_anom_ms, start, stop);
    double time_anom_us = time_anom_ms * 1000.0;

    cudaMemcpy(h_partialCounts.data(), d_partialCounts,
               totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

    int totalAnomalies = 0;
    for (int i = 0; i < totalThreads; i++)
        totalAnomalies += h_partialCounts[i];

    cout << "ANOMALY DETECTION: \n";
    cout << "Threshold                    : " << THRESHOLD << " °C\n";
    cout << "Total Anomalies Found        : " << totalAnomalies << "\n";
    cout << "Execution Time               : " << time_anom_us << " µs\n\n";


    // Cleanup
    cudaFree(d_temps);
    cudaFree(d_movAvg);
    cudaFree(d_partialSums);
    cudaFree(d_partialCounts);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
