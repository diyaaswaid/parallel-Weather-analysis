#include <iostream>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

vector<double> temps;
vector<double> partial_sum;
vector<double> movAvg;
vector<int> partial_anomalies;

int N;
int NUM_THREADS;
const int WINDOW = 7;


void* thread_sum(void* arg) {
    long tid = (long)arg;
    int chunk = N / NUM_THREADS;
    int start = tid * chunk;
    int end = (tid == NUM_THREADS - 1) ? N : start + chunk;

    double sum = 0.0;
    for (int i = start; i < end; i++)
        sum += temps[i];

    partial_sum[tid] = sum;
    pthread_exit(NULL);
}


void* thread_mavg(void* arg) {
    long tid = (long)arg;
    int chunk = N / NUM_THREADS;
    int start = tid * chunk;
    int end = (tid == NUM_THREADS - 1) ? N : start + chunk;

    if (start < WINDOW - 1)
        start = WINDOW - 1;

    for (int i = start; i < end; i++) {
        double sum = 0.0;
        for (int j = i - (WINDOW - 1); j <= i; j++) {
            sum += temps[j];
        }
        movAvg[i] = sum / WINDOW;
    }

    pthread_exit(NULL);
}


void* thread_anomaly(void* arg) {
    long tid = (long)arg;
    int chunk = N / NUM_THREADS;
    int start = tid * chunk;
    int end = (tid == NUM_THREADS - 1) ? N : start + chunk;

    if (start < WINDOW - 1)
        start = WINDOW - 1;

    int count = 0;
    for (int i = start; i < end; i++) {
        double diff = temps[i] - movAvg[i];
        if (fabs(diff) > 5.0)
            count++;
    }

    partial_anomalies[tid] = count;
    pthread_exit(NULL);
}

int main() {

    ifstream file("cleaned_temps.txt");
    double v;
    while (file >> v) temps.push_back(v);
    file.close();

    N = temps.size();
    if (N == 0) {
        cout << "Error: cleaned_temps.txt is empty.\n";
        return 1;
    }

    // Get thread count from environment variable
    char* env = getenv("THREADS");
    if (env == NULL) {
        NUM_THREADS = 4; // Default
    } else {
        NUM_THREADS = atoi(env);
        if (NUM_THREADS <= 0)
            NUM_THREADS = 4;
    }

    partial_sum.resize(NUM_THREADS);
    partial_anomalies.resize(NUM_THREADS);
    movAvg.resize(N, 0.0);

    pthread_t threads[NUM_THREADS];

    cout << "Loaded Temperature Values: " << N << "\n";
    cout << "Threads Used             : " << NUM_THREADS << "\n\n";


    auto start_avg = high_resolution_clock::now();

    for (long t = 0; t < NUM_THREADS; t++)
        pthread_create(&threads[t], NULL, thread_sum, (void*)t);

    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    double total_sum = 0;
    for (int t = 0; t < NUM_THREADS; t++)
        total_sum += partial_sum[t];

    double avg = total_sum / N;

    auto end_avg = high_resolution_clock::now();
    double time_avg_us = duration_cast<microseconds>(end_avg - start_avg).count();
    cout << "GLOBAL AVERAGE\n";
    cout << "Computed Average Temperature : " << avg << " °C\n";
    cout << "Execution Time               : " << time_avg_us << " µs\n\n";

    auto start_mavg = high_resolution_clock::now();

    for (long t = 0; t < NUM_THREADS; t++)
        pthread_create(&threads[t], NULL, thread_mavg, (void*)t);

    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    auto end_mavg = high_resolution_clock::now();
    double time_mavg_us = duration_cast<microseconds>(end_mavg - start_mavg).count();

    cout << "7-DAY MOVING AVERAGE: \n";
    cout << "Window Size                  : 7 days\n";
    cout << "Sample Moving Avg [index 100]: " << movAvg[100] << " °C\n";
    cout << "Execution Time               : " << time_mavg_us << " µs\n\n";

    auto start_anom = high_resolution_clock::now();

    for (long t = 0; t < NUM_THREADS; t++)
        pthread_create(&threads[t], NULL, thread_anomaly, (void*)t);

    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    int total_anomalies = 0;
    for (int t = 0; t < NUM_THREADS; t++)
        total_anomalies += partial_anomalies[t];

    auto end_anom = high_resolution_clock::now();
    double time_anom_us = duration_cast<microseconds>(end_anom - start_anom).count();

    cout << "ANOMALY DETECTION: \n";
    cout << "Threshold                    : 5.0 °C\n";
    cout << "Total Anomalies Found        : " << total_anomalies << "\n";
    cout << "Execution Time               : " << time_anom_us << " µs\n\n";


    return 0;
}
