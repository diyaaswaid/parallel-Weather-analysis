#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// ---- CUDA ERROR CHECK MACRO ----
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;             \
            return 1;                                                           \
        }                                                                       \
    } while (0)


// ===================== CUDA KERNELS =====================

// Kernel 1: Partial sums for global average
__global__ void sumKernel(const double* temps, double* partialSums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    double local = 0.0;
    // grid-stride loop: each thread sums a subset of elements
    for (int i = idx; i < N; i += totalThreads) {
        local += temps[i];
    }

    if (idx < totalThreads) {
        partialSums[idx] = local;
    }
}

// Kernel 2: Moving average (7-day window)
__global__ void movingAvgKernel(const double* temps, double* movAvg,
                                int N, int window) {
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
    if (!file.is_open()) {
        cerr << "Error: could not open cleaned_temps.txt\n";
        return 1;
    }

    vector<double> h_temps;
    double v;
    while (file >> v) {
        h_temps.push_back(v);
    }
    file.close();

    int N = static_cast<int>(h_temps.size());
    if (N == 0) {
        cout << "Error: cleaned_temps.txt is empty.\n";
        return 1;
    }

    const int WINDOW = 7;
    const double THRESHOLD = 5.0;

    cout << "Loaded Temperature Values : " << N << "\n\n";

    // ---------- 2. Allocate GPU memory ----------
    double* d_temps = nullptr;
    double* d_movAvg = nullptr;
    double* d_partialSums = nullptr;
    int*    d_partialCounts = nullptr;

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    int totalThreads = blockSize * gridSize;

    CUDA_CHECK(cudaMalloc(&d_temps,        N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_movAvg,       N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partialSums,  totalThreads * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partialCounts,totalThreads * sizeof(int)));

    // Copy temperatures to device
    CUDA_CHECK(cudaMemcpy(d_temps, h_temps.data(),
                          N * sizeof(double), cudaMemcpyHostToDevice));

    // Host buffers for results
    vector<double> h_partialSums(totalThreads, 0.0);
    vector<int>    h_partialCounts(totalThreads, 0);
    vector<double> h_movAvg(N, 0.0);

    // CUDA timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------- 3. GLOBAL AVERAGE ----------
    CUDA_CHECK(cudaEventRecord(start));

    sumKernel<<<gridSize, blockSize>>>(d_temps, d_partialSums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_avg_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_avg_ms, start, stop)); // ms

    CUDA_CHECK(cudaMemcpy(h_partialSums.data(), d_partialSums,
                          totalThreads * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double totalSum = 0.0;
    for (int i = 0; i < totalThreads; i++) {
        totalSum += h_partialSums[i];
    }

    double avg = totalSum / static_cast<double>(N);
    double time_avg_us = time_avg_ms * 1000.0; // convert to µs

    cout << "GLOBAL AVERAGE:\n";
    cout << "Computed Average Temperature : " << avg << " °C\n";
    cout << "Execution Time               : " << time_avg_us << " µs\n\n";

    // ---------- 4. 7-DAY MOVING AVERAGE ----------
    CUDA_CHECK(cudaEventRecord(start));

    movingAvgKernel<<<gridSize, blockSize>>>(d_temps, d_movAvg, N, WINDOW);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_mavg_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_mavg_ms, start, stop));
    double time_mavg_us = time_mavg_ms * 1000.0;

    CUDA_CHECK(cudaMemcpy(h_movAvg.data(), d_movAvg,
                          N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cout << "7-DAY MOVING AVERAGE\n";
    cout << "Window Size                  : " << WINDOW << " days\n";
    if (N > 100)
        cout << "Sample Moving Avg [index 100]: " << h_movAvg[100] << " °C\n";
    else
        cout << "Sample Moving Avg [index 100]: N/A (N <= 100)\n";
    cout << "Execution Time               : " << time_mavg_us << " µs\n\n";

    // ---------- 5. ANOMALY DETECTION ----------
    CUDA_CHECK(cudaEventRecord(start));

    anomalyKernel<<<gridSize, blockSize>>>(
        d_temps, d_movAvg, N, WINDOW, THRESHOLD, d_partialCounts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_anom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_anom_ms, start, stop));
    double time_anom_us = time_anom_ms * 1000.0;

    CUDA_CHECK(cudaMemcpy(h_partialCounts.data(), d_partialCounts,
                          totalThreads * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int totalAnomalies = 0;
    for (int i = 0; i < totalThreads; i++) {
        totalAnomalies += h_partialCounts[i];
    }

    cout << "ANOMALY DETECTION:\n";
    cout << "Threshold                    : " << THRESHOLD << " °C\n";
    cout << "Total Anomalies Found        : " << totalAnomalies << "\n";
    cout << "Execution Time               : " << time_anom_us << " µs\n\n";

    // ---------- 6. Cleanup ----------
    CUDA_CHECK(cudaFree(d_temps));
    CUDA_CHECK(cudaFree(d_movAvg));
    CUDA_CHECK(cudaFree(d_partialSums));
    CUDA_CHECK(cudaFree(d_partialCounts));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
