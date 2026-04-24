#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace {

#ifndef MAP_FOLDING_MAX_GRID
#define MAP_FOLDING_MAX_GRID 9
#endif

constexpr int MAX_GRID = MAP_FOLDING_MAX_GRID;
constexpr int MAX_N = MAX_GRID * MAX_GRID;
constexpr int MAX_GAP = MAX_N * MAX_N + 1;

using Count = unsigned __int128;
using U64 = unsigned long long;

__constant__ uint16_t d_const[2][MAX_N + 1][MAX_N + 1];

struct Problem {
    int grid = 0;
    int n = 0;
    uint16_t d[2][MAX_N + 1][MAX_N + 1]{};
};

struct State {
    uint16_t a[MAX_N + 1]{};
    uint16_t b[MAX_N + 1]{};
    uint16_t gapter[MAX_N + 1]{};
};

struct WorkBatch {
    std::vector<State> states;
    std::size_t first_index = 0;
};

class BatchQueue {
public:
    explicit BatchQueue(std::size_t max_batches) : max_batches_(std::max<std::size_t>(1, max_batches)) {}

    void push(WorkBatch&& batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&]() { return queue_.size() < max_batches_; });
        queue_.push_back(std::move(batch));
        not_empty_.notify_one();
    }

    bool pop(WorkBatch& batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        batch = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
    }

private:
    std::size_t max_batches_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<WorkBatch> queue_;
    bool closed_ = false;
};

struct Options {
    int grid = 0;
    int depth = 0;
    int batch_size = 100000;
    int threads_per_block = 128;
    int blocks_per_sm = 8;
    int max_gpus = 0;
    int max_queued_batches = 8;
    std::size_t max_states = 0;
};

void check_cuda(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << file << ":" << line << ": " << call << " failed: "
                  << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

double seconds_since(std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

std::string to_decimal(Count value) {
    if (value == 0) {
        return "0";
    }

    std::string out;
    while (value > 0) {
        int digit = static_cast<int>(value % 10);
        out.push_back(static_cast<char>('0' + digit));
        value /= 10;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

Problem make_problem(int grid) {
    Problem problem;
    problem.grid = grid;
    problem.n = grid * grid;

    int bigP[3] = {1, grid, grid * grid};
    int c[2][MAX_N + 1]{};

    for (int dim = 0; dim < 2; ++dim) {
        for (int m = 1; m <= problem.n; ++m) {
            c[dim][m] = (m - 1) / bigP[dim] - ((m - 1) / bigP[dim + 1]) * grid + 1;
        }
    }

    for (int dim = 0; dim < 2; ++dim) {
        for (int l = 1; l <= problem.n; ++l) {
            for (int m = 1; m <= l; ++m) {
                int delta = c[dim][l] - c[dim][m];
                int next = m;
                if ((delta & 1) == 0) {
                    if (c[dim][m] != 1) {
                        next = m - bigP[dim];
                    }
                } else if (c[dim][m] != grid && m + bigP[dim] <= l) {
                    next = m + bigP[dim];
                }
                problem.d[dim][l][m] = static_cast<uint16_t>(next);
            }
        }
    }

    return problem;
}

inline void compute_gaps_host(
    const Problem& problem,
    const uint16_t* b,
    uint8_t* seen,
    uint16_t* gap,
    int l_leaf,
    int& g
) {
    int dd = 0;
    int gg = g;

    uint16_t m = problem.d[0][l_leaf][l_leaf];
    if (m == l_leaf) {
        ++dd;
    } else {
        while (m != l_leaf) {
            gap[gg] = m;
            if (seen[m]++ == 0) {
                ++gg;
            }
            m = problem.d[0][l_leaf][b[m]];
        }
    }

    m = problem.d[1][l_leaf][l_leaf];
    if (m == l_leaf) {
        ++dd;
    } else {
        while (m != l_leaf) {
            gap[gg] = m;
            if (seen[m]++ == 0) {
                ++gg;
            }
            m = problem.d[1][l_leaf][b[m]];
        }
    }

    if (dd == 2) {
        for (int m2 = 0; m2 < l_leaf; ++m2) {
            gap[gg++] = static_cast<uint16_t>(m2);
        }
    }

    int out = g;
    const uint8_t needed = static_cast<uint8_t>(2 - dd);
    for (int j = g; j < gg; ++j) {
        uint16_t candidate = gap[j];
        if (seen[candidate] == needed) {
            gap[out++] = candidate;
        }
        seen[candidate] = 0;
    }
    g = out;
}

std::vector<State> collect_states(const Problem& problem, int depth) {
    State state;
    uint8_t seen[MAX_N + 1]{};
    uint16_t gap[MAX_GAP];
    std::vector<State> states;

    int l_leaf = 1;
    int g = 0;

    for (;;) {
        if (l_leaf <= 1 || state.b[0] == 1) {
            if (l_leaf > depth) {
                states.push_back(state);
                g = state.gapter[l_leaf - 1];
            } else {
                g = state.gapter[l_leaf - 1];
                compute_gaps_host(problem, state.b, seen, gap, l_leaf, g);
            }
        }

        while (g == state.gapter[l_leaf - 1]) {
            --l_leaf;
            if (l_leaf == 0) {
                return states;
            }
            state.b[state.a[l_leaf]] = state.b[l_leaf];
            state.a[state.b[l_leaf]] = state.a[l_leaf];
        }

        state.a[l_leaf] = gap[--g];
        state.b[l_leaf] = state.b[state.a[l_leaf]];
        state.b[state.a[l_leaf]] = static_cast<uint16_t>(l_leaf);
        state.a[state.b[l_leaf]] = static_cast<uint16_t>(l_leaf);
        state.gapter[l_leaf] = static_cast<uint16_t>(g);
        ++l_leaf;
    }
}

std::size_t stream_state_batches(
    const Problem& problem,
    int depth,
    int batch_size,
    std::size_t max_states,
    BatchQueue& queue,
    std::atomic<std::size_t>& generated
) {
    State state;
    uint8_t seen[MAX_N + 1]{};
    uint16_t gap[MAX_GAP];
    WorkBatch batch;
    batch.states.reserve(batch_size);
    std::size_t total = 0;

    int l_leaf = 1;
    int g = 0;

    auto flush = [&]() {
        if (batch.states.empty()) {
            return;
        }
        batch.first_index = total - batch.states.size();
        generated += batch.states.size();
        queue.push(std::move(batch));
        batch = WorkBatch{};
        batch.states.reserve(batch_size);
    };

    for (;;) {
        if (l_leaf <= 1 || state.b[0] == 1) {
            if (l_leaf > depth) {
                batch.states.push_back(state);
                ++total;
                if (static_cast<int>(batch.states.size()) == batch_size) {
                    flush();
                }
                if (max_states != 0 && total >= max_states) {
                    flush();
                    queue.close();
                    return total;
                }
                g = state.gapter[l_leaf - 1];
            } else {
                g = state.gapter[l_leaf - 1];
                compute_gaps_host(problem, state.b, seen, gap, l_leaf, g);
            }
        }

        while (g == state.gapter[l_leaf - 1]) {
            --l_leaf;
            if (l_leaf == 0) {
                flush();
                queue.close();
                return total;
            }
            state.b[state.a[l_leaf]] = state.b[l_leaf];
            state.a[state.b[l_leaf]] = state.a[l_leaf];
        }

        state.a[l_leaf] = gap[--g];
        state.b[l_leaf] = state.b[state.a[l_leaf]];
        state.b[state.a[l_leaf]] = static_cast<uint16_t>(l_leaf);
        state.a[state.b[l_leaf]] = static_cast<uint16_t>(l_leaf);
        state.gapter[l_leaf] = static_cast<uint16_t>(g);
        ++l_leaf;
    }
}

__device__ inline void compute_gaps_device(
    const uint16_t* b,
    uint8_t* seen,
    uint16_t* gap,
    int l_leaf,
    int& g
) {
    int dd = 0;
    int gg = g;

    uint16_t m = d_const[0][l_leaf][l_leaf];
    if (m == l_leaf) {
        ++dd;
    } else {
        while (m != l_leaf) {
            gap[gg] = m;
            if (seen[m]++ == 0) {
                ++gg;
            }
            m = d_const[0][l_leaf][b[m]];
        }
    }

    m = d_const[1][l_leaf][l_leaf];
    if (m == l_leaf) {
        ++dd;
    } else {
        while (m != l_leaf) {
            gap[gg] = m;
            if (seen[m]++ == 0) {
                ++gg;
            }
            m = d_const[1][l_leaf][b[m]];
        }
    }

    if (dd == 2) {
        for (int m2 = 0; m2 < l_leaf; ++m2) {
            gap[gg++] = static_cast<uint16_t>(m2);
        }
    }

    int out = g;
    const uint8_t needed = static_cast<uint8_t>(2 - dd);
    for (int j = g; j < gg; ++j) {
        uint16_t candidate = gap[j];
        if (seen[candidate] == needed) {
            gap[out++] = candidate;
        }
        seen[candidate] = 0;
    }
    g = out;
}

__device__ U64 solve_state_device(
    int n,
    int start_leaf,
    int state_index,
    const uint16_t* all_a,
    const uint16_t* all_b,
    const uint16_t* all_gapter
) {
    uint16_t a[MAX_N + 1];
    uint16_t b[MAX_N + 1];
    uint16_t gapter[MAX_N + 1];
    uint8_t seen[MAX_N + 1] = {};
    uint16_t gap[MAX_GAP];

    int stride = n + 1;
    int offset = state_index * stride;
    for (int i = 0; i <= n; ++i) {
        a[i] = all_a[offset + i];
        b[i] = all_b[offset + i];
        gapter[i] = all_gapter[offset + i];
    }

    int l_leaf = start_leaf;
    int g = gapter[start_leaf - 1];
    U64 total = 0;

    for (;;) {
        if (l_leaf <= 1 || b[0] == 1) {
            if (l_leaf > n) {
                total += static_cast<U64>(n);
            } else {
                g = gapter[l_leaf - 1];
                compute_gaps_device(b, seen, gap, l_leaf, g);
            }
        }

        while (g == gapter[l_leaf - 1]) {
            if (l_leaf == start_leaf) {
                return total;
            }
            --l_leaf;
            b[a[l_leaf]] = b[l_leaf];
            a[b[l_leaf]] = a[l_leaf];
        }

        a[l_leaf] = gap[--g];
        b[l_leaf] = b[a[l_leaf]];
        b[a[l_leaf]] = static_cast<uint16_t>(l_leaf);
        a[b[l_leaf]] = static_cast<uint16_t>(l_leaf);
        gapter[l_leaf] = static_cast<uint16_t>(g);
        ++l_leaf;
    }
}

__global__ void solve_kernel(
    int n,
    int start_leaf,
    int num_states,
    const uint16_t* all_a,
    const uint16_t* all_b,
    const uint16_t* all_gapter,
    U64* results,
    unsigned int* next_index
) {
    for (;;) {
        unsigned int state_index = atomicAdd(next_index, 1U);
        if (state_index >= static_cast<unsigned int>(num_states)) {
            return;
        }
        results[state_index] = solve_state_device(
            n, start_leaf, static_cast<int>(state_index), all_a, all_b, all_gapter);
    }
}

void pack_states(
    const std::vector<State>& states,
    int n,
    std::vector<uint16_t>& all_a,
    std::vector<uint16_t>& all_b,
    std::vector<uint16_t>& all_gapter
) {
    int count = static_cast<int>(states.size());
    int stride = n + 1;
    all_a.resize(static_cast<std::size_t>(count) * stride);
    all_b.resize(static_cast<std::size_t>(count) * stride);
    all_gapter.resize(static_cast<std::size_t>(count) * stride);

    for (int i = 0; i < count; ++i) {
        const State& state = states[i];
        std::memcpy(all_a.data() + static_cast<std::size_t>(i) * stride, state.a, stride * sizeof(uint16_t));
        std::memcpy(all_b.data() + static_cast<std::size_t>(i) * stride, state.b, stride * sizeof(uint16_t));
        std::memcpy(all_gapter.data() + static_cast<std::size_t>(i) * stride, state.gapter, stride * sizeof(uint16_t));
    }
}

Count run_gpu_batches(
    int device,
    const Problem& problem,
    const Options& options,
    BatchQueue& queue,
    std::atomic<std::size_t>& completed
) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpyToSymbol(d_const, problem.d, sizeof(problem.d)));

    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int blocks = std::max(1, sms * options.blocks_per_sm);
    int stride = problem.n + 1;
    int max_batch = options.batch_size;

    uint16_t* d_a = nullptr;
    uint16_t* d_b = nullptr;
    uint16_t* d_gapter = nullptr;
    U64* d_results = nullptr;
    unsigned int* d_next = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, static_cast<std::size_t>(max_batch) * stride * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_b, static_cast<std::size_t>(max_batch) * stride * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_gapter, static_cast<std::size_t>(max_batch) * stride * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_results, static_cast<std::size_t>(max_batch) * sizeof(U64)));
    CUDA_CHECK(cudaMalloc(&d_next, sizeof(unsigned int)));

    std::vector<uint16_t> all_a;
    std::vector<uint16_t> all_b;
    std::vector<uint16_t> all_gapter;
    std::vector<U64> results(max_batch);

    Count total = 0;
    WorkBatch batch;

    while (queue.pop(batch)) {
        int count = static_cast<int>(batch.states.size());
        pack_states(batch.states, problem.n, all_a, all_b, all_gapter);

        std::size_t bytes = static_cast<std::size_t>(count) * stride * sizeof(uint16_t);
        CUDA_CHECK(cudaMemcpy(d_a, all_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, all_b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gapter, all_gapter.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_next, 0, sizeof(unsigned int)));

        solve_kernel<<<blocks, options.threads_per_block>>>(
            problem.n, options.depth + 1, count, d_a, d_b, d_gapter, d_results, d_next);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(results.data(), d_results, count * sizeof(U64), cudaMemcpyDeviceToHost));
        for (int i = 0; i < count; ++i) {
            total += static_cast<Count>(results[i]);
        }
        completed += static_cast<std::size_t>(count);
        batch.states.clear();
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_gapter));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_next));

    return total;
}

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0') {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        std::exit(2);
    }
    return static_cast<int>(value);
}

std::size_t parse_size(const char* text, const char* name) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (*text == '\0' || *end != '\0') {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        std::exit(2);
    }
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    if (argc < 3 || argc > 9) {
        std::cerr
            << "Usage: " << argv[0] << " grid depth [batch_size] [threads_per_block] [blocks_per_sm] [max_gpus] [max_queued_batches] [max_states]\n"
            << "Example: " << argv[0] << " 6 18 5000 128 8 4 8 0\n";
        std::exit(1);
    }

    Options options;
    options.grid = parse_int(argv[1], "grid");
    options.depth = parse_int(argv[2], "depth");
    if (argc >= 4) {
        options.batch_size = parse_int(argv[3], "batch_size");
    }
    if (argc >= 5) {
        options.threads_per_block = parse_int(argv[4], "threads_per_block");
    }
    if (argc >= 6) {
        options.blocks_per_sm = parse_int(argv[5], "blocks_per_sm");
    }
    if (argc >= 7) {
        options.max_gpus = parse_int(argv[6], "max_gpus");
    }
    if (argc >= 8) {
        options.max_queued_batches = parse_int(argv[7], "max_queued_batches");
    }
    if (argc >= 9) {
        options.max_states = parse_size(argv[8], "max_states");
    }

    if (options.grid < 1 || options.grid > MAX_GRID) {
        std::cerr << "grid must be between 1 and " << MAX_GRID << "\n";
        std::exit(1);
    }
    int n = options.grid * options.grid;
    if (options.depth < 1 || options.depth > n) {
        std::cerr << "depth must be between 1 and " << n << "\n";
        std::exit(1);
    }
    if (options.batch_size < 1 || options.threads_per_block < 1 ||
        options.blocks_per_sm < 1 || options.max_queued_batches < 1) {
        std::cerr << "batch_size, threads_per_block, blocks_per_sm, and max_queued_batches must be positive\n";
        std::exit(1);
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    Options options = parse_options(argc, argv);
    auto total_start = std::chrono::steady_clock::now();

    Problem problem = make_problem(options.grid);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }
    if (options.max_gpus > 0) {
        device_count = std::min(device_count, options.max_gpus);
    }

    std::cerr << "gpus=" << device_count
              << " batch_size=" << options.batch_size
              << " threads_per_block=" << options.threads_per_block
              << " blocks_per_sm=" << options.blocks_per_sm
              << " max_queued_batches=" << options.max_queued_batches
              << " max_states=" << options.max_states << "\n";

    BatchQueue queue(options.max_queued_batches);
    std::atomic<std::size_t> generated{0};
    std::atomic<std::size_t> completed{0};
    std::vector<Count> partial(device_count, 0);
    std::vector<std::thread> workers;
    workers.reserve(device_count);
    std::mutex progress_mutex;
    std::condition_variable progress_cv;
    bool progress_done = false;

    auto solve_start = std::chrono::steady_clock::now();
    for (int device = 0; device < device_count; ++device) {
        workers.emplace_back([&, device]() {
            partial[device] = run_gpu_batches(device, problem, options, queue, completed);
        });
    }

    std::thread progress([&]() {
        std::unique_lock<std::mutex> lock(progress_mutex);
        while (!progress_done) {
            if (progress_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return progress_done; })) {
                break;
            }
            std::size_t done = completed.load(std::memory_order_relaxed);
            std::size_t made = generated.load(std::memory_order_relaxed);
            double elapsed = seconds_since(solve_start);
            double rate = elapsed > 0.0 ? done / elapsed : 0.0;
            std::cerr << "progress completed=" << done
                      << " generated=" << made
                      << " states rate=" << rate << "/s elapsed_s=" << elapsed << "\n";
        }
    });

    auto collect_start = std::chrono::steady_clock::now();
    std::size_t total_states = stream_state_batches(
        problem, options.depth, options.batch_size, options.max_states, queue, generated);
    double collect_s = seconds_since(collect_start);
    std::cerr << "grid=" << options.grid
              << " depth=" << options.depth
              << " states=" << total_states
              << " collect_s=" << collect_s << "\n";

    for (auto& worker : workers) {
        worker.join();
    }
    completed = total_states;
    {
        std::lock_guard<std::mutex> lock(progress_mutex);
        progress_done = true;
    }
    progress_cv.notify_one();
    progress.join();

    Count answer = 0;
    for (Count value : partial) {
        answer += value;
    }

    double solve_s = seconds_since(solve_start);
    double total_s = seconds_since(total_start);
    std::cout << to_decimal(answer) << "\n";
    std::cerr << "solve_s=" << solve_s << " total_s=" << total_s << "\n";
    return 0;
}
