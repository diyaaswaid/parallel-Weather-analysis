#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<double> temps;
    int N = 0;

    if (rank == 0) {
        ifstream file("cleaned_temps.txt");
        double v;
        while (file >> v)
            temps.push_back(v);
        file.close();
        N = temps.size();

        if (N == 0) {
            cout << "Error: cleaned_temps.txt is empty.\n";
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast size to all ranks
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local buffer
    vector<double> local_data;
    local_data.resize(N);

    // Broadcast whole dataset to all ranks (simple, correct)
    if (rank == 0) {
        MPI_Bcast(temps.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(local_data.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        temps = local_data;
    }

    // Compute chunk boundaries
    int chunk = N / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? N : start + chunk;

    auto start_avg = high_resolution_clock::now();

    double local_sum = 0.0;
    for (int i = start; i < end; i++) {
        local_sum += temps[i];
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double avg = 0.0;
    if (rank == 0) avg = global_sum / N;

    auto end_avg = high_resolution_clock::now();
    double time_avg_us = duration_cast<microseconds>(end_avg - start_avg).count();

    const int WINDOW = 7;
    vector<double> movAvg(N, 0.0);

    auto start_mavg = high_resolution_clock::now();

    for (int i = max(WINDOW - 1, start); i < end; i++) {
        double sum = 0.0;
        for (int j = i - (WINDOW - 1); j <= i; j++)
            sum += temps[j];
        movAvg[i] = sum / WINDOW;
    }

    auto end_mavg = high_resolution_clock::now();
    double time_mavg_us = duration_cast<microseconds>(end_mavg - start_mavg).count();

    auto start_anom = high_resolution_clock::now();

    int local_anomalies = 0;

    for (int i = max(WINDOW - 1, start); i < end; i++) {
        double diff = temps[i] - movAvg[i];
        if (fabs(diff) > 5.0)
            local_anomalies++;
    }

    int global_anomalies = 0;
    MPI_Reduce(&local_anomalies, &global_anomalies, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto end_anom = high_resolution_clock::now();
    double time_anom_us = duration_cast<microseconds>(end_anom - start_anom).count();
    if (rank == 0) {        
        cout << "Loaded Temperature Values : " << N << "\n";
        cout << "MPI Processes Used        : " << size << "\n\n";

        cout << "GLOBAL AVERAGE: \n";
        cout << "Computed Average Temperature : " << avg << " °C\n";
        cout << "Execution Time               : " << time_avg_us << " µs\n\n";

        cout << "7-DAY MOVING AVERAGE: \n";
        cout << "Window Size                  : 7 days\n";
        cout << "Sample Moving Avg [index 100]: " << movAvg[100] << " °C\n";
        cout << "Execution Time               : " << time_mavg_us << " µs\n\n";

        cout << "ANOMALY DETECTION: \n";
        cout << "Threshold                    : 5.0 °C\n";
        cout << "Total Anomalies Found        : " << global_anomalies << "\n";
        cout << "Execution Time               : " << time_anom_us << " µs\n\n";

    }

    MPI_Finalize();
    return 0;
}
