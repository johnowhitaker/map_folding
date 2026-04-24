// Symmetry-reduced square map folding solver.
//
// This is an independent C++ port of the algorithm posted by gsitcia on
// Code Golf Stack Exchange:
// https://codegolf.stackexchange.com/questions/276709/count-the-possible-folds-of-an-n%C2%B2-grid
//
// The key difference from mf.c is the square-specific symmetry reduction:
// enumerate the center 3x3 or 4x4 cycle, quotient those cycles by the D4
// symmetries of the square and stack reversal, then count only representatives
// with their orbit sizes.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int MAX_N = 64;
using Count = unsigned __int128;
using Cycle = std::array<uint8_t, MAX_N>;
using Edges = std::array<std::array<uint8_t, MAX_N>, 4>;

struct Case {
    Cycle cycle{};
    uint64_t multiplier = 1;
};

class CaseQueue {
public:
    explicit CaseQueue(std::size_t max_size) : max_size_(std::max<std::size_t>(1, max_size)) {}

    void push(Case&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&]() { return queue_.size() < max_size_; });
        queue_.push_back(std::move(item));
        not_empty_.notify_one();
    }

    bool pop(Case& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
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
    std::size_t max_size_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<Case> queue_;
    bool closed_ = false;
};

struct CenterKey {
    std::array<uint8_t, 16> next{};

    bool operator==(const CenterKey& other) const {
        return next == other.next;
    }
};

struct CenterKeyHash {
    std::size_t operator()(const CenterKey& key) const {
        uint64_t h = 1469598103934665603ULL;
        for (uint8_t value : key.next) {
            h ^= value;
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
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

std::size_t parse_size(const char* text, const char* name);

std::vector<std::vector<int>> build_map(int width, int height) {
    std::vector<std::vector<int>> map(width, std::vector<int>(height, width * height));
    int x = 0;
    int y = 0;
    int dx = 1;
    int dy = 0;

    for (int i = width * height - 1; i >= 0; --i) {
        map[x][y] = i;
        int x1 = x + dx;
        int y1 = y + dy;
        if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height || map[x1][y1] < width * height) {
            int t = dy;
            dy = dx;
            dx = -t;
            x += dx;
            y += dy;
        } else {
            x = x1;
            y = y1;
        }
    }
    return map;
}

Edges make_edges(int width, int height) {
    int n = width * height;
    Edges edges{};
    for (auto& edge : edges) {
        edge.fill(static_cast<uint8_t>(n));
    }

    auto map = build_map(width, height);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int idx = map[x][y];
            if (x > 0) {
                auto& edge = edges[x % 2];
                int idx1 = map[x - 1][y];
                edge[idx] = static_cast<uint8_t>(idx1);
                edge[idx1] = static_cast<uint8_t>(idx);
            }
            if (y > 0) {
                auto& edge = edges[2 + (y % 2)];
                int idx1 = map[x][y - 1];
                edge[idx] = static_cast<uint8_t>(idx1);
                edge[idx1] = static_cast<uint8_t>(idx);
            }
        }
    }
    return edges;
}

uint64_t low_bits(int count) {
    return count == 64 ? ~0ULL : ((1ULL << count) - 1ULL);
}

uint64_t get_good(const Cycle& cycle, int i, const Edges& edges) {
    uint64_t good = low_bits(i);
    for (const auto& edge : edges) {
        int other = edge[i];
        if (other >= i) {
            continue;
        }

        uint64_t allowed = 1ULL << other;
        int j = cycle[other];
        while (j != other) {
            int j1 = edge[j];
            if (j1 < i) {
                j = j1;
            }
            allowed |= 1ULL << j;
            j = cycle[j];
        }

        good &= allowed;
        if (good == 0) {
            return 0;
        }
    }
    return good;
}

template <typename Fn>
void iter_good(uint64_t good, Cycle& cycle, int i, Fn&& fn) {
    while (good != 0) {
        int j = __builtin_ctzll(good);
        good &= good - 1;
        uint8_t next = cycle[j];
        cycle[i] = next;
        cycle[j] = static_cast<uint8_t>(i);
        fn(cycle);
        cycle[j] = next;
    }
}

uint64_t recurse_count(Cycle& cycle, int i, const Edges& edges, int n) {
    uint64_t good = get_good(cycle, i, edges);
    if (good == 0) {
        return 0;
    }
    if (i == n - 1) {
        return static_cast<uint64_t>(__builtin_popcountll(good));
    }

    uint64_t out = 0;
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        out += recurse_count(next_cycle, i + 1, edges, n);
    });
    return out;
}

uint64_t estimate_work(Cycle& cycle, int i, const Edges& edges, int stop_i) {
    if (i >= stop_i) {
        return 1;
    }

    uint64_t good = get_good(cycle, i, edges);
    if (good == 0) {
        return 0;
    }

    uint64_t out = 0;
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        out += estimate_work(next_cycle, i + 1, edges, stop_i);
    });
    return out;
}

void recurse_generate(Cycle& cycle, int i, const Edges& edges, int n, std::vector<Cycle>& out) {
    if (i == n) {
        out.push_back(cycle);
        return;
    }

    uint64_t good = get_good(cycle, i, edges);
    if (good == 0) {
        return;
    }
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        recurse_generate(next_cycle, i + 1, edges, n, out);
    });
}

void score_and_sort_cases(std::vector<Case>& cases, int start_i, const Edges& edges, int n2) {
    const char* depth_text = std::getenv("MF_COST_DEPTH");
    int depth = depth_text == nullptr ? 0 : std::atoi(depth_text);
    if (depth <= 0 || cases.empty()) {
        return;
    }

    int stop_i = std::min(n2, start_i + depth);
    std::vector<std::pair<uint64_t, Case>> scored;
    scored.reserve(cases.size());
    uint64_t min_score = UINT64_MAX;
    uint64_t max_score = 0;
    Count total_score = 0;

    auto start = std::chrono::steady_clock::now();
    for (const Case& candidate : cases) {
        Cycle cycle = candidate.cycle;
        uint64_t score = estimate_work(cycle, start_i, edges, stop_i);
        min_score = std::min(min_score, score);
        max_score = std::max(max_score, score);
        total_score += static_cast<Count>(score);
        scored.push_back({score, candidate});
    }

    std::cerr << "cost_depth=" << depth
              << " min_score=" << min_score
              << " max_score=" << max_score
              << " total_score=" << to_decimal(total_score)
              << " score_s=" << seconds_since(start) << "\n";

    const char* sort_text = std::getenv("MF_SORT_CASES");
    if (sort_text == nullptr || std::string(sort_text).empty() || std::string(sort_text) == "0") {
        return;
    }

    bool ascending = std::string(sort_text) == "asc";
    std::sort(scored.begin(), scored.end(), [&](const auto& lhs, const auto& rhs) {
        if (lhs.first == rhs.first) {
            return lhs.second.multiplier < rhs.second.multiplier;
        }
        return ascending ? lhs.first < rhs.first : lhs.first > rhs.first;
    });

    for (std::size_t i = 0; i < cases.size(); ++i) {
        cases[i] = scored[i].second;
    }
    std::cerr << "sorted_cases=" << (ascending ? "asc" : "desc") << "\n";
}

std::vector<std::array<uint8_t, 16>> generate_permutations(int c) {
    auto map = build_map(c, c);
    std::vector<std::pair<int, int>> reverse_map(c * c);
    for (int x = 0; x < c; ++x) {
        for (int y = 0; y < c; ++y) {
            reverse_map[map[x][y]] = {x, y};
        }
    }

    int d = c - 1;
    std::vector<std::array<uint8_t, 16>> out;
    out.reserve(8);

    auto make_perm = [&](auto transform) {
        std::array<uint8_t, 16> perm{};
        perm.fill(255);
        for (int i = 0; i < c * c; ++i) {
            auto [x, y] = reverse_map[i];
            auto [tx, ty] = transform(x, y);
            perm[i] = static_cast<uint8_t>(map[tx][ty]);
        }
        out.push_back(perm);
    };

    make_perm([&](int x, int y) { return std::pair<int, int>{x, y}; });
    make_perm([&](int x, int y) { return std::pair<int, int>{y, x}; });
    make_perm([&](int x, int y) { return std::pair<int, int>{x, d - y}; });
    make_perm([&](int x, int y) { return std::pair<int, int>{y, d - x}; });
    make_perm([&](int x, int y) { return std::pair<int, int>{d - x, y}; });
    make_perm([&](int x, int y) { return std::pair<int, int>{d - y, x}; });
    make_perm([&](int x, int y) { return std::pair<int, int>{d - x, d - y}; });
    make_perm([&](int x, int y) { return std::pair<int, int>{d - y, d - x}; });

    return out;
}

CenterKey make_key(const Cycle& cycle, int c2) {
    CenterKey key;
    key.next.fill(255);
    for (int i = 0; i < c2; ++i) {
        key.next[i] = cycle[i];
    }
    return key;
}

std::unordered_set<CenterKey, CenterKeyHash> orbit_keys(
    const Cycle& cycle,
    int c,
    const std::vector<std::array<uint8_t, 16>>& permutations
) {
    int c2 = c * c;
    std::unordered_set<CenterKey, CenterKeyHash> out;

    for (const auto& perm : permutations) {
        Cycle mapped = cycle;
        for (int i = 0; i < c2; ++i) {
            int j = cycle[i];
            mapped[perm[i]] = perm[j];
        }

        Cycle reversed = mapped;
        int i = 0;
        do {
            reversed[mapped[i]] = static_cast<uint8_t>(i);
            i = mapped[i];
        } while (i != 0);

        out.insert(make_key(mapped, c2));
        out.insert(make_key(reversed, c2));
    }

    return out;
}

std::vector<Case> deduplicate_cases(const std::vector<Cycle>& cases, int c) {
    auto permutations = generate_permutations(c);
    std::unordered_set<CenterKey, CenterKeyHash> seen;
    std::vector<Case> out;
    int c2 = c * c;

    for (const Cycle& cycle : cases) {
        CenterKey key = make_key(cycle, c2);
        if (seen.find(key) != seen.end()) {
            continue;
        }

        auto orbit = orbit_keys(cycle, c, permutations);
        out.push_back({cycle, static_cast<uint64_t>(orbit.size())});
        seen.insert(orbit.begin(), orbit.end());
    }

    return out;
}

Count solve_cases_parallel(
    std::vector<Case>& cases,
    int start_i,
    const Edges& edges,
    int n2,
    int thread_count,
    std::size_t max_cases,
    std::size_t start_case
) {
    if (start_case >= cases.size()) {
        return 0;
    }
    std::size_t end_case = cases.size();
    if (max_cases != 0) {
        end_case = std::min(end_case, start_case + max_cases);
    }
    std::size_t case_count = end_case - start_case;
    if (case_count == 0) {
        return 0;
    }

    thread_count = std::max(1, std::min<int>(thread_count, static_cast<int>(case_count)));
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> completed{0};
    std::vector<Count> partial(thread_count, 0);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    auto start = std::chrono::steady_clock::now();
    std::mutex progress_mutex;
    std::condition_variable progress_cv;
    bool done = false;
    std::thread progress([&]() {
        std::unique_lock<std::mutex> lock(progress_mutex);
        while (!done) {
            if (progress_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; })) {
                break;
            }
            auto c = completed.load(std::memory_order_relaxed);
            double elapsed = seconds_since(start);
            std::cerr << "progress=" << c << "/" << case_count
                      << " rate=" << (elapsed > 0.0 ? c / elapsed : 0.0)
                      << "/s elapsed_s=" << elapsed << "\n";
        }
    });

    for (int tid = 0; tid < thread_count; ++tid) {
        workers.emplace_back([&, tid]() {
            Count local = 0;
            for (;;) {
                std::size_t offset = next.fetch_add(1, std::memory_order_relaxed);
                if (offset >= case_count) {
                    break;
                }
                std::size_t index = start_case + offset;
                Cycle cycle = cases[index].cycle;
                uint64_t count = recurse_count(cycle, start_i, edges, n2);
                local += static_cast<Count>(cases[index].multiplier) * static_cast<Count>(count);
                completed.fetch_add(1, std::memory_order_relaxed);
            }
            partial[tid] = local;
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(progress_mutex);
        done = true;
    }
    progress_cv.notify_one();
    progress.join();

    Count total = 0;
    for (Count value : partial) {
        total += value;
    }
    return total;
}

bool recurse_stream_cases(
    Cycle& cycle,
    int i,
    int stop_i,
    const Edges& edges,
    uint64_t multiplier,
    std::size_t start_case,
    std::size_t end_case,
    std::size_t& case_index,
    CaseQueue& queue,
    std::atomic<std::size_t>& selected
) {
    if (i == stop_i) {
        if (case_index >= start_case && case_index < end_case) {
            queue.push(Case{cycle, multiplier});
            selected.fetch_add(1, std::memory_order_relaxed);
        }
        ++case_index;
        return case_index >= end_case;
    }

    uint64_t good = get_good(cycle, i, edges);
    while (good != 0) {
        int j = __builtin_ctzll(good);
        good &= good - 1;
        uint8_t next = cycle[j];
        cycle[i] = next;
        cycle[j] = static_cast<uint8_t>(i);
        bool stop = recurse_stream_cases(
            cycle, i + 1, stop_i, edges, multiplier, start_case, end_case, case_index, queue, selected);
        cycle[j] = next;
        if (stop) {
            return true;
        }
    }
    return false;
}

Count solve_cases_streaming(
    const std::vector<Case>& seed_cases,
    int seed_i,
    int expanded_i,
    const Edges& edges,
    int n2,
    int thread_count,
    std::size_t max_cases,
    std::size_t start_case
) {
    std::size_t queue_size = 4096;
    if (const char* text = std::getenv("MF_QUEUE_SIZE")) {
        queue_size = std::max<std::size_t>(1, parse_size(text, "MF_QUEUE_SIZE"));
    }
    std::size_t end_case = max_cases == 0
        ? std::numeric_limits<std::size_t>::max()
        : start_case + max_cases;

    CaseQueue queue(queue_size);
    std::atomic<std::size_t> selected{0};
    std::atomic<std::size_t> completed{0};
    std::vector<Count> partial(thread_count, 0);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    auto start = std::chrono::steady_clock::now();
    std::atomic<bool> producer_done{false};
    std::mutex progress_mutex;
    std::condition_variable progress_cv;
    bool done = false;

    for (int tid = 0; tid < thread_count; ++tid) {
        workers.emplace_back([&, tid]() {
            Count local = 0;
            Case item;
            while (queue.pop(item)) {
                Cycle cycle = item.cycle;
                uint64_t count = recurse_count(cycle, expanded_i, edges, n2);
                local += static_cast<Count>(item.multiplier) * static_cast<Count>(count);
                completed.fetch_add(1, std::memory_order_relaxed);
            }
            partial[tid] = local;
        });
    }

    std::thread progress([&]() {
        std::unique_lock<std::mutex> lock(progress_mutex);
        while (!done) {
            if (progress_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; })) {
                break;
            }
            std::size_t s = selected.load(std::memory_order_relaxed);
            std::size_t c = completed.load(std::memory_order_relaxed);
            double elapsed = seconds_since(start);
            std::cerr << "stream selected=" << s
                      << " completed=" << c
                      << " producer_done=" << producer_done.load(std::memory_order_relaxed)
                      << " rate=" << (elapsed > 0.0 ? c / elapsed : 0.0)
                      << "/s elapsed_s=" << elapsed << "\n";
        }
    });

    std::size_t case_index = 0;
    for (const Case& seed : seed_cases) {
        Cycle cycle = seed.cycle;
        bool stop = recurse_stream_cases(
            cycle, seed_i, expanded_i, edges, seed.multiplier,
            start_case, end_case, case_index, queue, selected);
        if (stop) {
            break;
        }
    }
    producer_done = true;
    queue.close();

    for (auto& worker : workers) {
        worker.join();
    }

    {
        std::lock_guard<std::mutex> lock(progress_mutex);
        done = true;
    }
    progress_cv.notify_one();
    progress.join();

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

std::size_t parse_size(const char* text, const char* name) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (*text == '\0' || *end != '\0') {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        std::exit(2);
    }
    return static_cast<std::size_t>(value);
}

void usage(const char* exe) {
    std::cerr << "Usage: " << exe << " n [threads] [max_cases] [start_case]\n"
              << "Example: " << exe << " 6 12\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        usage(argv[0]);
        return 1;
    }

    int n = parse_int(argv[1], "n");
    if (n < 1 || n > 8) {
        std::cerr << "n must be between 1 and 8\n";
        return 1;
    }

    unsigned hw = std::thread::hardware_concurrency();
    int threads = hw == 0 ? 1 : static_cast<int>(hw);
    if (argc >= 3) {
        threads = parse_int(argv[2], "threads");
    }
    if (threads < 1) {
        std::cerr << "threads must be positive\n";
        return 1;
    }
    std::size_t max_cases = 0;
    if (argc >= 4) {
        max_cases = parse_size(argv[3], "max_cases");
    }
    std::size_t start_case = 0;
    if (argc >= 5) {
        start_case = parse_size(argv[4], "start_case");
    }

    auto total_start = std::chrono::steady_clock::now();
    if (n <= 1) {
        std::cout << "1\n";
        return 0;
    }
    if (n == 2) {
        std::cout << "8\n";
        return 0;
    }

    int n2 = n * n;
    Edges edges = make_edges(n, n);
    Cycle cycle{};
    cycle.fill(0);
    cycle[0] = 1;
    cycle[1] = 2;
    cycle[2] = 0;

    Count answer = 0;
    if (n <= 5) {
        uint64_t count = recurse_count(cycle, 3, edges, n2);
        answer = static_cast<Count>(2) * static_cast<Count>(n2) * static_cast<Count>(count);
        std::cerr << "raw_count=" << count << "\n";
    } else {
        int c = (n % 2 == 0) ? 4 : 3;
        int c2 = c * c;
        std::vector<Cycle> center_cases;
        recurse_generate(cycle, 3, edges, c2, center_cases);
        std::cerr << "center_cases=" << center_cases.size() << "\n";

        std::vector<Case> cases = deduplicate_cases(center_cases, c);
        std::cerr << "reduced_cases=" << cases.size() << "\n";

        bool stream_cases = std::getenv("MF_STREAM_CASES") != nullptr;
        int seed_i = c2;
        int solve_i = c2;
        if (n >= 7) {
            int expanded_n = 24;
            const char* expanded_text = std::getenv("MF_EXPANDED_N");
            if (expanded_text != nullptr) {
                expanded_n = std::atoi(expanded_text);
            }
            if (expanded_n < c2 || expanded_n > n2) {
                std::cerr << "MF_EXPANDED_N must be between " << c2 << " and " << n2 << "\n";
                return 1;
            }
            if (stream_cases) {
                solve_i = expanded_n;
                std::cerr << "stream_seed_cases=" << cases.size()
                          << " stream_from=" << seed_i
                          << " stream_to=" << solve_i << "\n";
            } else {
                std::vector<Case> expanded;
                for (const Case& center_case : cases) {
                    std::vector<Cycle> generated;
                    Cycle local = center_case.cycle;
                    recurse_generate(local, c2, edges, expanded_n, generated);
                    for (Cycle& generated_cycle : generated) {
                        expanded.push_back({generated_cycle, center_case.multiplier});
                    }
                }
                cases = std::move(expanded);
                c2 = expanded_n;
                seed_i = c2;
                solve_i = c2;
                std::cerr << "expanded_cases=" << cases.size() << "\n";
            }
        }

        if (!stream_cases) {
            score_and_sort_cases(cases, solve_i, edges, n2);
        }
        if (std::getenv("MF_SETUP_ONLY") != nullptr) {
            std::cout << cases.size() << "\n";
            std::cerr << "n=" << n
                      << " threads=" << threads
                      << " max_cases=" << max_cases
                      << " start_case=" << start_case
                      << " setup_only=1"
                      << " total_s=" << seconds_since(total_start) << "\n";
            return 0;
        }
        Count count = stream_cases
            ? solve_cases_streaming(cases, seed_i, solve_i, edges, n2, threads, max_cases, start_case)
            : solve_cases_parallel(cases, solve_i, edges, n2, threads, max_cases, start_case);
        answer = count * static_cast<Count>(n2);
    }

    std::cout << to_decimal(answer) << "\n";
    std::cerr << "n=" << n
              << " threads=" << threads
              << " max_cases=" << max_cases
              << " start_case=" << start_case
              << " total_s=" << seconds_since(total_start) << "\n";
    return 0;
}
