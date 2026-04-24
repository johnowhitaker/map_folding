// Experimental 9x9-capable symmetry-reduced square map folding solver.
//
// This starts from ../sym_fold.cpp, an independent C++ port of the algorithm
// posted by gsitcia on Code Golf Stack Exchange:
// https://codegolf.stackexchange.com/questions/276709/count-the-possible-folds-of-an-n%C2%B2-grid
//
// Keep sym_fold.cpp fixed for 8x8 replication. This file experiments with
// 9x9 by using 128-bit masks and optional 5x5 center reduction.

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
#include <numeric>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int MAX_N = 100;
constexpr int MAX_CENTER = 25;
using Count = unsigned __int128;
using Mask = unsigned __int128;
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
    std::array<uint8_t, MAX_CENTER> next{};

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

std::vector<std::vector<int>> build_spiral_map(int width, int height) {
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

std::vector<std::pair<int, int>> coords_from_map(const std::vector<std::vector<int>>& map) {
    int width = static_cast<int>(map.size());
    int height = static_cast<int>(map[0].size());
    std::vector<std::pair<int, int>> coords(width * height);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            coords[map[x][y]] = {x, y};
        }
    }
    return coords;
}

std::vector<std::vector<int>> build_map(int width, int height) {
    auto base = build_spiral_map(width, height);
    const char* order_text = std::getenv("MF_ORDER");
    if (order_text == nullptr || std::string(order_text).empty() || std::string(order_text) == "spiral") {
        return base;
    }
    if (width != height || (width % 2) == 0) {
        return base;
    }

    int lock_center = 3;
    if (const char* text = std::getenv("MF_LOCK_CENTER_N")) {
        lock_center = std::atoi(text);
    }
    int lock_radius = std::max(0, (lock_center - 1) / 2);
    int center = width / 2;
    auto base_coords = coords_from_map(base);
    std::vector<std::pair<int, int>> ordered;
    ordered.reserve(width * height);
    std::string order = order_text;

    for (int radius = 0; radius <= center; ++radius) {
        std::vector<std::pair<int, int>> ring;
        for (auto coord : base_coords) {
            int r = std::max(std::abs(coord.first - center), std::abs(coord.second - center));
            if (r == radius) {
                ring.push_back(coord);
            }
        }

        if (radius > lock_radius) {
            if (order == "reverse") {
                std::reverse(ring.begin(), ring.end());
            } else if (order == "opposite") {
                std::vector<std::pair<int, int>> interleaved;
                interleaved.reserve(ring.size());
                std::size_t half = ring.size() / 2;
                for (std::size_t i = 0; i < half; ++i) {
                    interleaved.push_back(ring[i]);
                    if (i + half < ring.size()) {
                        interleaved.push_back(ring[i + half]);
                    }
                }
                if ((ring.size() % 2) != 0) {
                    interleaved.push_back(ring.back());
                }
                ring = std::move(interleaved);
            } else if (order == "axes_first") {
                std::stable_sort(ring.begin(), ring.end(), [&](auto lhs, auto rhs) {
                    int la = std::abs(lhs.first - center) + std::abs(lhs.second - center);
                    int ra = std::abs(rhs.first - center) + std::abs(rhs.second - center);
                    return la < ra;
                });
            } else if (order == "corners_first") {
                std::stable_sort(ring.begin(), ring.end(), [&](auto lhs, auto rhs) {
                    int lc = (std::abs(lhs.first - center) == radius && std::abs(lhs.second - center) == radius) ? 0 : 1;
                    int rc = (std::abs(rhs.first - center) == radius && std::abs(rhs.second - center) == radius) ? 0 : 1;
                    return lc < rc;
                });
            }
        }

        ordered.insert(ordered.end(), ring.begin(), ring.end());
    }

    std::vector<std::vector<int>> map(width, std::vector<int>(height, width * height));
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        auto [x, y] = ordered[i];
        map[x][y] = static_cast<int>(i);
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

Mask low_bits(int count) {
    if (count <= 0) {
        return 0;
    }
    if (count >= 128) {
        return ~static_cast<Mask>(0);
    }
    return (static_cast<Mask>(1) << count) - 1;
}

int ctz_mask(Mask value) {
    uint64_t low = static_cast<uint64_t>(value);
    if (low != 0) {
        return __builtin_ctzll(low);
    }
    uint64_t high = static_cast<uint64_t>(value >> 64);
    return 64 + __builtin_ctzll(high);
}

int popcount_mask(Mask value) {
    uint64_t low = static_cast<uint64_t>(value);
    uint64_t high = static_cast<uint64_t>(value >> 64);
    return __builtin_popcountll(low) + __builtin_popcountll(high);
}

Mask get_good(const Cycle& cycle, int i, const Edges& edges) {
    Mask good = low_bits(i);
    for (const auto& edge : edges) {
        int other = edge[i];
        if (other >= i) {
            continue;
        }

        Mask allowed = static_cast<Mask>(1) << other;
        int j = cycle[other];
        while (j != other) {
            int j1 = edge[j];
            if (j1 < i) {
                j = j1;
            }
            allowed |= static_cast<Mask>(1) << j;
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
void iter_good(Mask good, Cycle& cycle, int i, Fn&& fn) {
    while (good != 0) {
        int j = ctz_mask(good);
        good &= good - 1;
        uint8_t next = cycle[j];
        cycle[i] = next;
        cycle[j] = static_cast<uint8_t>(i);
        fn(cycle);
        cycle[j] = next;
    }
}

Count recurse_count(Cycle& cycle, int i, const Edges& edges, int n) {
    Mask good = get_good(cycle, i, edges);
    if (good == 0) {
        return 0;
    }
    if (i == n - 1) {
        return static_cast<Count>(popcount_mask(good));
    }

    Count out = 0;
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        out += recurse_count(next_cycle, i + 1, edges, n);
    });
    return out;
}

Count recurse_count_work(Cycle& cycle, int i, const Edges& edges, int n, Count& nodes) {
    ++nodes;
    Mask good = get_good(cycle, i, edges);
    if (good == 0) {
        return 0;
    }
    if (i == n - 1) {
        return static_cast<Count>(popcount_mask(good));
    }

    Count out = 0;
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        out += recurse_count_work(next_cycle, i + 1, edges, n, nodes);
    });
    return out;
}

uint64_t estimate_work(Cycle& cycle, int i, const Edges& edges, int stop_i) {
    if (i >= stop_i) {
        return 1;
    }

    Mask good = get_good(cycle, i, edges);
    if (good == 0) {
        return 0;
    }

    uint64_t out = 0;
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        out += estimate_work(next_cycle, i + 1, edges, stop_i);
    });
    return out;
}

Count count_generated_cases(Cycle& cycle, int i, int stop_i, const Edges& edges) {
    if (i == stop_i) {
        return 1;
    }

    Mask good = get_good(cycle, i, edges);
    if (good == 0) {
        return 0;
    }

    Count out = 0;
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        out += count_generated_cases(next_cycle, i + 1, stop_i, edges);
    });
    return out;
}

Count percentile(std::vector<Count>& values, double p) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    std::size_t index = static_cast<std::size_t>(p * static_cast<double>(values.size() - 1));
    return values[index];
}

void print_count_summary(const char* label, std::vector<Count> values) {
    if (values.empty()) {
        std::cerr << label << "_count=0\n";
        return;
    }

    Count total = std::accumulate(values.begin(), values.end(), static_cast<Count>(0));
    Count p50 = percentile(values, 0.50);
    Count p90 = percentile(values, 0.90);
    Count p99 = percentile(values, 0.99);
    Count min_value = values.front();
    Count max_value = values.back();
    Count mean = total / static_cast<Count>(values.size());
    std::cerr << label
              << "_count=" << values.size()
              << " min=" << to_decimal(min_value)
              << " p50=" << to_decimal(p50)
              << " p90=" << to_decimal(p90)
              << " p99=" << to_decimal(p99)
              << " max=" << to_decimal(max_value)
              << " mean=" << to_decimal(mean)
              << " total=" << to_decimal(total)
              << "\n";
}

void recurse_generate(Cycle& cycle, int i, const Edges& edges, int n, std::vector<Cycle>& out) {
    if (i == n) {
        out.push_back(cycle);
        return;
    }

    Mask good = get_good(cycle, i, edges);
    if (good == 0) {
        return;
    }
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        recurse_generate(next_cycle, i + 1, edges, n, out);
    });
}

template <typename Fn>
void recurse_visit(Cycle& cycle, int i, const Edges& edges, int n, Fn&& fn) {
    if (i == n) {
        fn(cycle);
        return;
    }

    Mask good = get_good(cycle, i, edges);
    if (good == 0) {
        return;
    }
    iter_good(good, cycle, i, [&](Cycle& next_cycle) {
        recurse_visit(next_cycle, i + 1, edges, n, fn);
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

std::vector<std::array<uint8_t, MAX_CENTER>> generate_permutations(int c) {
    auto map = build_map(c, c);
    std::vector<std::pair<int, int>> reverse_map(c * c);
    for (int x = 0; x < c; ++x) {
        for (int y = 0; y < c; ++y) {
            reverse_map[map[x][y]] = {x, y};
        }
    }

    int d = c - 1;
    std::vector<std::array<uint8_t, MAX_CENTER>> out;
    out.reserve(8);

    auto make_perm = [&](auto transform) {
        std::array<uint8_t, MAX_CENTER> perm{};
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
    const std::vector<std::array<uint8_t, MAX_CENTER>>& permutations
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

std::vector<Case> deduplicate_cases_streamed(
    Cycle& cycle,
    int start_i,
    int c,
    const Edges& edges,
    std::size_t& raw_count
) {
    auto permutations = generate_permutations(c);
    std::unordered_set<CenterKey, CenterKeyHash> seen;
    std::vector<Case> out;
    int c2 = c * c;
    raw_count = 0;

    recurse_visit(cycle, start_i, edges, c2, [&](const Cycle& candidate) {
        ++raw_count;
        CenterKey key = make_key(candidate, c2);
        if (seen.find(key) != seen.end()) {
            return;
        }

        auto orbit = orbit_keys(candidate, c, permutations);
        out.push_back({candidate, static_cast<uint64_t>(orbit.size())});
        seen.insert(orbit.begin(), orbit.end());
    });

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
                Count count = recurse_count(cycle, start_i, edges, n2);
                local += static_cast<Count>(cases[index].multiplier) * count;
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

    Mask good = get_good(cycle, i, edges);
    while (good != 0) {
        int j = ctz_mask(good);
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

template <typename Fn>
bool recurse_profile_cases(
    Cycle& cycle,
    int i,
    int stop_i,
    const Edges& edges,
    uint64_t multiplier,
    std::size_t start_case,
    std::size_t end_case,
    std::size_t& case_index,
    Fn&& fn
) {
    if (i == stop_i) {
        if (case_index >= start_case && case_index < end_case) {
            fn(case_index, multiplier, cycle);
        }
        ++case_index;
        return case_index >= end_case;
    }

    Mask good = get_good(cycle, i, edges);
    while (good != 0) {
        int j = ctz_mask(good);
        good &= good - 1;
        uint8_t next = cycle[j];
        cycle[i] = next;
        cycle[j] = static_cast<uint8_t>(i);
        bool stop = recurse_profile_cases(
            cycle, i + 1, stop_i, edges, multiplier, start_case, end_case, case_index, fn);
        cycle[j] = next;
        if (stop) {
            return true;
        }
    }
    return false;
}

Count profile_cases_work(
    const std::vector<Case>& seed_cases,
    int seed_i,
    int expanded_i,
    const Edges& edges,
    int n2,
    std::size_t max_cases,
    std::size_t start_case
) {
    if (max_cases == 0) {
        std::cerr << "MF_PROFILE_WORK requires max_cases > 0\n";
        std::exit(2);
    }

    std::size_t end_case = start_case + max_cases;
    std::size_t case_index = 0;
    std::vector<Count> node_counts;
    std::vector<Count> leaf_counts;
    std::vector<Count> weighted_leaf_counts;
    node_counts.reserve(max_cases);
    leaf_counts.reserve(max_cases);
    weighted_leaf_counts.reserve(max_cases);
    bool verbose = std::getenv("MF_PROFILE_VERBOSE") != nullptr;
    auto profile_start = std::chrono::steady_clock::now();

    for (const Case& seed : seed_cases) {
        Cycle cycle = seed.cycle;
        bool stop = recurse_profile_cases(
            cycle, seed_i, expanded_i, edges, seed.multiplier,
            start_case, end_case, case_index,
            [&](std::size_t index, uint64_t multiplier, const Cycle& prefix) {
                Cycle local = prefix;
                Count nodes = 0;
                auto case_start = std::chrono::steady_clock::now();
                Count leaves = recurse_count_work(local, expanded_i, edges, n2, nodes);
                double elapsed = seconds_since(case_start);
                node_counts.push_back(nodes);
                leaf_counts.push_back(leaves);
                weighted_leaf_counts.push_back(static_cast<Count>(multiplier) * leaves);
                if (verbose) {
                    std::cerr << "profile_case index=" << index
                              << " multiplier=" << multiplier
                              << " nodes=" << to_decimal(nodes)
                              << " leaves=" << to_decimal(leaves)
                              << " weighted_leaves="
                              << to_decimal(static_cast<Count>(multiplier) * leaves)
                              << " elapsed_s=" << elapsed << "\n";
                }
            });
        if (stop) {
            break;
        }
    }

    print_count_summary("profile_nodes", node_counts);
    print_count_summary("profile_leaves", leaf_counts);
    print_count_summary("profile_weighted_leaves", weighted_leaf_counts);
    std::cerr << "profile_start_case=" << start_case
              << " profile_max_cases=" << max_cases
              << " profile_seen_cases=" << node_counts.size()
              << " profile_total_s=" << seconds_since(profile_start) << "\n";

    return std::accumulate(weighted_leaf_counts.begin(), weighted_leaf_counts.end(), static_cast<Count>(0));
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
                Count count = recurse_count(cycle, expanded_i, edges, n2);
                local += static_cast<Count>(item.multiplier) * count;
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

int choose_center_size(int n) {
    int c = n >= 9 ? 5 : ((n % 2 == 0) ? 4 : 3);
    if (const char* text = std::getenv("MF_CENTER_N")) {
        c = std::atoi(text);
    }
    if (c < 3 || c > n || c * c > MAX_CENTER || (c % 2) != (n % 2)) {
        std::cerr << "MF_CENTER_N must be between 3 and " << n
                  << ", have the same parity as n, and fit MAX_CENTER="
                  << MAX_CENTER << "\n";
        std::exit(2);
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        usage(argv[0]);
        return 1;
    }

    int n = parse_int(argv[1], "n");
    if (n < 1 || n > 9) {
        std::cerr << "n must be between 1 and 9\n";
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
        Count count = recurse_count(cycle, 3, edges, n2);
        answer = static_cast<Count>(2) * static_cast<Count>(n2) * static_cast<Count>(count);
        std::cerr << "raw_count=" << to_decimal(count) << "\n";
    } else {
        int c = choose_center_size(n);
        int c2 = c * c;
        std::vector<Case> cases;
        if (std::getenv("MF_CENTER_STREAM_DEDUP") != nullptr) {
            std::size_t center_count = 0;
            cases = deduplicate_cases_streamed(cycle, 3, c, edges, center_count);
            std::cerr << "center_size=" << c << " center_cases=" << center_count
                      << " center_stream_dedup=1\n";
        } else {
            std::vector<Cycle> center_cases;
            recurse_generate(cycle, 3, edges, c2, center_cases);
            std::cerr << "center_size=" << c << " center_cases=" << center_cases.size() << "\n";
            cases = deduplicate_cases(center_cases, c);
        }
        std::cerr << "reduced_cases=" << cases.size() << "\n";

        if (std::getenv("MF_CENTER_HIST") != nullptr) {
            std::array<std::size_t, 17> hist{};
            for (const Case& item : cases) {
                if (item.multiplier < hist.size()) {
                    ++hist[static_cast<std::size_t>(item.multiplier)];
                }
            }
            std::cerr << "center_multiplier_hist";
            for (std::size_t i = 0; i < hist.size(); ++i) {
                if (hist[i] != 0) {
                    std::cerr << " " << i << ":" << hist[i];
                }
            }
            std::cerr << "\n";
        }

        std::size_t seed_slice_base = 0;
        if (const char* seed_start_text = std::getenv("MF_SEED_START")) {
            std::size_t seed_start = parse_size(seed_start_text, "MF_SEED_START");
            seed_slice_base = seed_start;
            std::size_t seed_count = cases.size();
            if (const char* seed_count_text = std::getenv("MF_SEED_COUNT")) {
                seed_count = parse_size(seed_count_text, "MF_SEED_COUNT");
            }
            std::size_t seed_end = std::min(cases.size(), seed_start + seed_count);
            if (seed_start >= cases.size()) {
                cases.clear();
            } else {
                cases = std::vector<Case>(cases.begin() + static_cast<std::ptrdiff_t>(seed_start),
                                          cases.begin() + static_cast<std::ptrdiff_t>(seed_end));
            }
            std::cerr << "seed_slice_start=" << seed_start
                      << " seed_slice_count=" << cases.size() << "\n";
        }

        bool stream_cases = std::getenv("MF_STREAM_CASES") != nullptr;
        int seed_i = c2;
        int solve_i = c2;
        if (n >= 7) {
            int expanded_n = std::max(24, c2);
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
        if (std::getenv("MF_PROFILE_WORK") != nullptr) {
            if (!stream_cases || solve_i <= seed_i) {
                std::cerr << "MF_PROFILE_WORK requires MF_STREAM_CASES=1 and MF_EXPANDED_N > center depth\n";
                return 2;
            }
            Count count = profile_cases_work(cases, seed_i, solve_i, edges, n2, max_cases, start_case);
            answer = count * static_cast<Count>(n2);
            std::cout << to_decimal(answer) << "\n";
            std::cerr << "n=" << n
                      << " threads=" << threads
                      << " center_n=" << c
                      << " max_cases=" << max_cases
                      << " start_case=" << start_case
                      << " profile_work=1"
                      << " total_s=" << seconds_since(total_start) << "\n";
            return 0;
        }
        if (std::getenv("MF_SETUP_ONLY") != nullptr) {
            if (stream_cases && solve_i > seed_i) {
                Count generated = 0;
                bool print_seed_counts = std::getenv("MF_SEED_COUNTS") != nullptr;
                bool summarize_seed_counts = std::getenv("MF_SEED_COUNTS_SUMMARY") != nullptr;
                std::vector<Count> seed_generated_counts;
                if (summarize_seed_counts) {
                    seed_generated_counts.reserve(cases.size());
                }
                for (std::size_t seed_index = 0; seed_index < cases.size(); ++seed_index) {
                    const Case& seed = cases[seed_index];
                    Cycle local = seed.cycle;
                    Count seed_generated = count_generated_cases(local, seed_i, solve_i, edges);
                    generated += seed_generated;
                    if (summarize_seed_counts) {
                        seed_generated_counts.push_back(seed_generated);
                    }
                    if (print_seed_counts) {
                        std::cerr << "seed_count index=" << (seed_slice_base + seed_index)
                                  << " multiplier=" << seed.multiplier
                                  << " expanded_cases=" << to_decimal(seed_generated)
                                  << "\n";
                    }
                }
                if (summarize_seed_counts) {
                    print_count_summary("seed_expanded", seed_generated_counts);
                }
                std::cout << to_decimal(generated) << "\n";
            } else {
                std::cout << cases.size() << "\n";
            }
            std::cerr << "n=" << n
                      << " threads=" << threads
                      << " center_n=" << c
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
