#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int MAX_GRID = 9;
constexpr int MAX_N = MAX_GRID * MAX_GRID;
constexpr int MAX_GAP = MAX_N * MAX_N + 1;

using Count = unsigned __int128;

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

inline void compute_gaps(
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

Count solve_from_state(const Problem& problem, const State& state, int start_leaf) {
    uint16_t a[MAX_N + 1];
    uint16_t b[MAX_N + 1];
    uint16_t gapter[MAX_N + 1];
    uint8_t seen[MAX_N + 1]{};
    uint16_t gap[MAX_GAP];

    std::memcpy(a, state.a, sizeof(a));
    std::memcpy(b, state.b, sizeof(b));
    std::memcpy(gapter, state.gapter, sizeof(gapter));

    int l_leaf = start_leaf;
    int g = gapter[start_leaf - 1];
    Count total = 0;

    for (;;) {
        if (l_leaf <= 1 || b[0] == 1) {
            if (l_leaf > problem.n) {
                total += static_cast<Count>(problem.n);
            } else {
                g = gapter[l_leaf - 1];
                compute_gaps(problem, b, seen, gap, l_leaf, g);
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
                compute_gaps(problem, state.b, seen, gap, l_leaf, g);
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

Count count_states(const Problem& problem, int depth) {
    State state;
    uint8_t seen[MAX_N + 1]{};
    uint16_t gap[MAX_GAP];
    Count states = 0;

    int l_leaf = 1;
    int g = 0;

    for (;;) {
        if (l_leaf <= 1 || state.b[0] == 1) {
            if (l_leaf > depth) {
                ++states;
                g = state.gapter[l_leaf - 1];
            } else {
                g = state.gapter[l_leaf - 1];
                compute_gaps(problem, state.b, seen, gap, l_leaf, g);
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

Count solve_parallel(const Problem& problem, int depth, int thread_count, std::size_t& state_count) {
    auto states = collect_states(problem, depth);
    state_count = states.size();
    if (states.empty()) {
        return 0;
    }

    thread_count = std::max(1, std::min<int>(thread_count, static_cast<int>(states.size())));
    std::atomic<std::size_t> next{0};
    std::vector<Count> partial(thread_count, 0);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (int tid = 0; tid < thread_count; ++tid) {
        threads.emplace_back([&, tid]() {
            Count local = 0;
            for (;;) {
                std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= states.size()) {
                    break;
                }
                local += solve_from_state(problem, states[index], depth + 1);
            }
            partial[tid] = local;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    Count total = 0;
    for (Count value : partial) {
        total += value;
    }
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

void usage(const char* exe) {
    std::cerr
        << "Usage: " << exe << " grid [partition_depth] [threads]\n"
        << "Example: " << exe << " 6 18 8\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        usage(argv[0]);
        return 1;
    }

    int grid = parse_int(argv[1], "grid");
    if (grid < 1 || grid > MAX_GRID) {
        std::cerr << "grid must be between 1 and " << MAX_GRID << "\n";
        return 1;
    }

    int n = grid * grid;
    int depth = (argc >= 3) ? parse_int(argv[2], "partition_depth") : 0;
    if (depth < 0 || depth > n) {
        std::cerr << "partition_depth must be between 0 and " << n << "\n";
        return 1;
    }

    unsigned hardware_threads = std::thread::hardware_concurrency();
    int threads = (hardware_threads == 0) ? 1 : static_cast<int>(hardware_threads);
    if (argc >= 4) {
        threads = parse_int(argv[3], "threads");
    }
    if (threads < 0) {
        std::cerr << "threads must be non-negative\n";
        return 1;
    }
    bool collect_only = std::getenv("MF_COLLECT_ONLY") != nullptr;

    auto total_start = std::chrono::steady_clock::now();
    Problem problem = make_problem(grid);

    Count answer = 0;
    std::size_t state_count = 1;
    double collect_seconds = 0.0;
    double solve_seconds = 0.0;

    if (depth == 0) {
        if (collect_only) {
            std::cout << "1\n";
            std::cerr << "grid=" << grid
                      << " depth=0 states=1 collect_only=1 total_s="
                      << seconds_since(total_start) << "\n";
            return 0;
        }
        State initial;
        auto solve_start = std::chrono::steady_clock::now();
        answer = solve_from_state(problem, initial, 1);
        solve_seconds = seconds_since(solve_start);
    } else {
        if (collect_only || threads == 0) {
            auto count_start = std::chrono::steady_clock::now();
            Count counted_states = count_states(problem, depth);
            collect_seconds = seconds_since(count_start);
            std::cout << to_decimal(counted_states) << "\n";
            std::cerr << "grid=" << grid
                      << " depth=" << depth
                      << " states=" << to_decimal(counted_states)
                      << " collect_s=" << collect_seconds
                      << " collect_only=1 total_s=" << seconds_since(total_start) << "\n";
            return 0;
        }

        auto collect_start = std::chrono::steady_clock::now();
        auto states = collect_states(problem, depth);
        collect_seconds = seconds_since(collect_start);
        state_count = states.size();

        int thread_count = std::max(1, std::min<int>(threads, static_cast<int>(states.size())));
        std::atomic<std::size_t> next{0};
        std::vector<Count> partial(thread_count, 0);
        std::vector<std::thread> workers;
        workers.reserve(thread_count);

        auto solve_start = std::chrono::steady_clock::now();
        for (int tid = 0; tid < thread_count; ++tid) {
            workers.emplace_back([&, tid]() {
                Count local = 0;
                for (;;) {
                    std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= states.size()) {
                        break;
                    }
                    local += solve_from_state(problem, states[index], depth + 1);
                }
                partial[tid] = local;
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        solve_seconds = seconds_since(solve_start);

        for (Count value : partial) {
            answer += value;
        }
    }

    double total_seconds = seconds_since(total_start);
    std::cout << to_decimal(answer) << "\n";
    std::cerr << "grid=" << grid
              << " depth=" << depth
              << " states=" << state_count
              << " threads=" << (depth == 0 ? 1 : std::min<int>(threads, static_cast<int>(state_count)))
              << " collect_s=" << collect_seconds
              << " solve_s=" << solve_seconds
              << " total_s=" << total_seconds << "\n";
    return 0;
}
