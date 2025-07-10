#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <thread>
#include <cuda_runtime.h>

typedef unsigned long long ull;
const int MAX_N = 64;
const int MAX_GAP = MAX_N * MAX_N + 1;

__constant__ int bigP_const[3];
__constant__ int c_const[3][MAX_N + 1];
__constant__ int d_const[3][MAX_N + 1][MAX_N + 1];

struct State {
    int a[MAX_N + 1];
    int b_arr[MAX_N + 1];
    int gapter[MAX_N + 2];
};

std::vector<State> collect_partial_states(int grid_size, int partition_depth, int hc[3][MAX_N + 1], int hd[3][MAX_N + 1][MAX_N + 1]) {
    int dim = 2;
    int p[2] = {grid_size, grid_size};
    int n = p[0] * p[1];

    std::vector<State> states;
    states.reserve(100000);

    int *a = new int[n + 1];
    int *b_arr = new int[n + 1];
    int *gapter = new int[n + 2];
    int *count = new int[n + 1];
    int *gap = new int[MAX_GAP];

    memset(a, 0, (n + 1) * sizeof(int));
    memset(b_arr, 0, (n + 1) * sizeof(int));
    memset(gapter, 0, (n + 2) * sizeof(int));
    memset(count, 0, (n + 1) * sizeof(int));
    memset(gap, 0, MAX_GAP * sizeof(int));

    b_arr[0] = 1; // Sentinel

    int flag = 1;
    int res = 0;
    int mod = 0;
    int g = 0;
    int l_leaf = 1;
    gapter[0] = 0;

    while (l_leaf > 0) {
        if (!flag || l_leaf <= 1 || b_arr[0] == 1) {
            if (l_leaf > n) {
                // Should not reach here
            } else if (l_leaf > partition_depth) {
                // Collect the partial state and force backtrack without computing gaps
                State s;
                memcpy(s.a, a, (n + 1) * sizeof(int));
                memcpy(s.b_arr, b_arr, (n + 1) * sizeof(int));
                memcpy(s.gapter, gapter, (n + 2) * sizeof(int));
                states.push_back(s);
                // No need to set g; it's already set to trigger pop in the while loop
            } else {
                int dd = 0;
                int gg = gapter[l_leaf - 1];
                g = gg;

                for (int i = 1; i <= dim; i++) {
                    if (hd[i][l_leaf][l_leaf] == l_leaf) {
                        dd++;
                    } else {
                        int m = hd[i][l_leaf][l_leaf];
                        while (m != l_leaf) {
                            if (mod == 0 || l_leaf != mod || m % mod == res) {
                                gap[gg] = m;
                                if (count[m]++ == 0) gg++;
                            }
                            m = hd[i][l_leaf][b_arr[m]];
                        }
                    }
                }

                if (dd == dim) {
                    for (int m = 0; m < l_leaf; m++) {
                        gap[gg++] = m;
                    }
                }

                for (int j = g; j < gg; j++) {
                    gap[g] = gap[j];
                    if (count[gap[j]] == dim - dd) {
                        g++;
                    }
                    count[gap[j]] = 0;
                }
            }
        }

        while (l_leaf > 0 && g == gapter[l_leaf - 1]) {
            l_leaf--;
            b_arr[a[l_leaf]] = b_arr[l_leaf];
            a[b_arr[l_leaf]] = a[l_leaf];
        }

        if (l_leaf > 0) {
            a[l_leaf] = gap[--g];
            b_arr[l_leaf] = b_arr[a[l_leaf]];
            b_arr[a[l_leaf]] = l_leaf;
            a[b_arr[l_leaf]] = l_leaf;
            gapter[l_leaf] = g;
            l_leaf++;
        }
    }

    delete [] a;
    delete [] b_arr;
    delete [] gapter;
    delete [] count;
    delete [] gap;

    return states;
}

__global__ void compute_counts(int grid_size, int partition_depth, int num_states, int *all_a, int *all_b_arr, int *all_gapter, ull *result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_states) return;

    int n = grid_size * grid_size;
    int dim = 2;
    int p[2] = {grid_size, grid_size};

    int a[MAX_N + 1];
    int b_arr[MAX_N + 1];
    int gapter[MAX_N + 2];
    int count[MAX_N + 1] = {0};
    int gap[MAX_GAP] = {0};

    for (int i = 0; i <= n; i++) {
        a[i] = all_a[id * (n + 1) + i];
        b_arr[i] = all_b_arr[id * (n + 1) + i];
    }
    for (int i = 0; i <= n + 1; i++) {
        gapter[i] = all_gapter[id * (n + 2) + i];
    }

    int flag = 1;
    int res = 0;
    int mod = 0;
    int g = 0;
    int l_leaf = partition_depth + 1;
    ull myCount = 0;

    while (l_leaf > 0) {
        if (!flag || l_leaf <= 1 || b_arr[0] == 1) {
            if (l_leaf > n) {
                myCount += n;
            } else {
                int dd = 0;
                int gg = gapter[l_leaf - 1];
                g = gg;

                for (int i = 1; i <= dim; i++) {
                    if (d_const[i][l_leaf][l_leaf] == l_leaf) {
                        dd++;
                    } else {
                        int m = d_const[i][l_leaf][l_leaf];
                        while (m != l_leaf) {
                            if (mod == 0 || l_leaf != mod || m % mod == res) {
                                gap[gg] = m;
                                if (count[m]++ == 0) gg++;
                            }
                            m = d_const[i][l_leaf][b_arr[m]];
                        }
                    }
                }

                if (dd == dim) {
                    for (int m = 0; m < l_leaf; m++) {
                        gap[gg++] = m;
                    }
                }

                for (int j = g; j < gg; j++) {
                    gap[g] = gap[j];
                    if (count[gap[j]] == dim - dd) {
                        g++;
                    }
                    count[gap[j]] = 0;
                }
            }
        }

        while (l_leaf > 0 && g == gapter[l_leaf - 1]) {
            l_leaf--;
            b_arr[a[l_leaf]] = b_arr[l_leaf];
            a[b_arr[l_leaf]] = a[l_leaf];
        }

        if (l_leaf > 0) {
            a[l_leaf] = gap[--g ];
            b_arr[l_leaf] = b_arr[a[l_leaf]];
            b_arr[a[l_leaf]] = l_leaf;
            a[b_arr[l_leaf]] = l_leaf;
            gapter[l_leaf] = g;
            l_leaf++;
        }
    }

    atomicAdd(result, myCount);
}

void process_gpu(int device_id, int grid_size, int partition_depth, int start_idx, int end_idx, const std::vector<State>& states, ull& partial_result) {
    cudaSetDevice(device_id);

    int num_local_states = end_idx - start_idx;
    if (num_local_states <= 0) {
        partial_result = 0;
        return;
    }

    int n = grid_size * grid_size;

    int *d_all_a, *d_all_b_arr, *d_all_gapter;
    ull *d_result;
    cudaMalloc(&d_all_a, num_local_states * (n + 1) * sizeof(int));
    cudaMalloc(&d_all_b_arr, num_local_states * (n + 1) * sizeof(int));
    cudaMalloc(&d_all_gapter, num_local_states * (n + 2) * sizeof(int));
    cudaMalloc(&d_result, sizeof(ull));

    int *h_all_a = new int[num_local_states * (n + 1)];
    int *h_all_b_arr = new int[num_local_states * (n + 1)];
    int *h_all_gapter = new int[num_local_states * (n + 2)];

    for (int s = 0; s < num_local_states; s++) {
        const State& state = states[start_idx + s];
        memcpy(h_all_a + s * (n + 1), state.a, (n + 1) * sizeof(int));
        memcpy(h_all_b_arr + s * (n + 1), state.b_arr, (n + 1) * sizeof(int));
        memcpy(h_all_gapter + s * (n + 2), state.gapter, (n + 2) * sizeof(int));
    }

    cudaMemcpy(d_all_a, h_all_a, num_local_states * (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_b_arr, h_all_b_arr, num_local_states * (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_gapter, h_all_gapter, num_local_states * (n + 2) * sizeof(int), cudaMemcpyHostToDevice);

    ull h_result = 0;
    cudaMemcpy(d_result, &h_result, sizeof(ull), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_blocks = (num_local_states + block_size - 1) / block_size;
    compute_counts<<<grid_blocks, block_size>>>(grid_size, partition_depth, num_local_states, d_all_a, d_all_b_arr, d_all_gapter, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(ull), cudaMemcpyDeviceToHost);
    partial_result = h_result;

    delete[] h_all_a;
    delete[] h_all_b_arr;
    delete[] h_all_gapter;
    cudaFree(d_all_a);
    cudaFree(d_all_b_arr);
    cudaFree(d_all_gapter);
    cudaFree(d_result);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: ./gpu_folding grid_size partition_depth\n");
        printf("Example: ./gpu_folding 8 20\n");
        return 1;
    }

    int grid_size = atoi(argv[1]);
    int partition_depth = atoi(argv[2]);
    int n = grid_size * grid_size;
    if (n > MAX_N) {
        printf("Grid too large, max 8 (64 stamps)\n");
        return 1;
    }

    int dim = 2;
    int p[2] = {grid_size, grid_size};

    int hbigP[3];
    hbigP[0] = 1;
    for (int i = 1; i <= dim; i++) {
        hbigP[i] = hbigP[i - 1] * p[i - 1];
    }

    int hc[3][MAX_N + 1];
    for (int i = 1; i <= dim; i++) {
        for (int m = 1; m <= n; m++) {
            hc[i][m] = (m - 1) / hbigP[i - 1] - ((m - 1) / hbigP[i]) * p[i - 1] + 1;
        }
    }

    int hd[3][MAX_N + 1][MAX_N + 1];
    for (int i = 1; i <= dim; i++) {
        for (int l = 1; l <= n; l++) {
            for (int m = 1; m <= l; m++) {
                int delta = hc[i][l] - hc[i][m];
                if ((delta & 1) == 0) {
                    if (hc[i][m] == 1) {
                        hd[i][l][m] = m;
                    } else {
                        hd[i][l][m] = m - hbigP[i - 1];
                    }
                } else {
                    if (hc[i][m] == p[i - 1] || m + hbigP[i - 1] > l) {
                        hd[i][l][m] = m;
                    } else {
                        hd[i][l][m] = m + hbigP[i - 1];
                    }
                }
            }
        }
    }

    auto states = collect_partial_states(grid_size, partition_depth, hc, hd);
    int num_states = states.size();
    printf("Number of partial states at depth %d: %d\n", partition_depth, num_states);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("Detected %d GPUs\n", num_gpus);

    for (int dev = 0; dev < num_gpus; ++dev) {
        cudaSetDevice(dev);
        cudaMemcpyToSymbol(bigP_const, hbigP, sizeof(hbigP));
        cudaMemcpyToSymbol(c_const, hc, sizeof(hc));
        cudaMemcpyToSymbol(d_const, hd, sizeof(hd));
    }

    std::vector<ull> partial_results(num_gpus, 0);
    std::vector<std::thread> threads;

    int states_per_gpu = num_states / num_gpus;
    int remainder = num_states % num_gpus;

    int current_start = 0;
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        int local_num = states_per_gpu + (gpu < remainder ? 1 : 0);
        int end_idx = current_start + local_num;

        threads.emplace_back([gpu, grid_size, partition_depth, current_start, end_idx, &states, &partial_results]() {
            ull local_result;
            process_gpu(gpu, grid_size, partition_depth, current_start, end_idx, states, local_result);
            partial_results[gpu] = local_result;
        });

        current_start = end_idx;
    }

    for (auto& t : threads) {
        t.join();
    }

    ull total_result = 0;
    for (auto res : partial_results) {
        total_result += res;
    }

    printf("Total number of ways: %llu\n", total_result);

    return 0;
}