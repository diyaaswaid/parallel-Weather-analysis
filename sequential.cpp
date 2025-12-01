#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main() {

    // =========================================================
    // 1. LOAD TEMPERATURE DATA
    // =========================================================
    ifstream file("cleaned_temps.txt");
    vector<double> temps;
    double value;

    while (file >> value) {
        temps.push_back(value);
    }
    file.close();

    int N = temps.size();
    if (N == 0) {
        cout << "Error: cleaned_temps.txt is empty.\n";
        return 1;
    }

    cout << "Loaded Temperature Values: " << N << "\n\n";
    auto start_avg = high_resolution_clock::now();

    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += temps[i];
    }
    double avg = sum / N;

    auto end_avg = high_resolution_clock::now();
    double time_avg_us = duration_cast<microseconds>(end_avg - start_avg).count();

    cout << "GLOBAL AVERAGE: \n";
    cout << "Computed Average Temperature : " << avg << " °C\n";
    cout << "Execution Time               : " << time_avg_us << " µs\n\n";

    const int WINDOW = 7;
    vector<double> movAvg(N, 0.0);

    auto start_mavg = high_resolution_clock::now();

    for (int i = WINDOW - 1; i < N; i++) {
        double windowSum = 0.0;
        for (int j = i - (WINDOW - 1); j <= i; j++) {
            windowSum += temps[j];
        }
        movAvg[i] = windowSum / WINDOW;
    }

    auto end_mavg = high_resolution_clock::now();
    double time_mavg_us = duration_cast<microseconds>(end_mavg - start_mavg).count();

    cout << "7-DAY MOVING AVERAGE: \n";
    cout << "Window Size                  : 7 days\n";
    cout << "Sample Moving Avg [index 100]: " << movAvg[100] << " °C\n";
    cout << "Execution Time               : " << time_mavg_us << " µs\n\n";

    const double threshold = 5.0;
    int anomalyCount = 0;

    auto start_anom = high_resolution_clock::now();

    for (int i = WINDOW - 1; i < N; i++) {
        double diff = temps[i] - movAvg[i];
        if (fabs(diff) > threshold) {
            anomalyCount++;
        }
    }

    auto end_anom = high_resolution_clock::now();
    double time_anom_us = duration_cast<microseconds>(end_anom - start_anom).count();

    cout << "ANOMALY DETECTION: \n";
    cout << "Threshold                    : " << threshold << " °C\n";
    cout << "Total Anomalies Found        : " << anomalyCount << "\n";
    cout << "Execution Time               : " << time_anom_us << " µs\n\n";


    return 0;
}
