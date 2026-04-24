// Experimental stack-state solver for square map folding.
//
// This tests an alternative formulation: after cutting the final folded stack
// at a fixed top cell, each parity class of grid edges behaves like a matching
// whose endpoints must be properly nested. Scanning the stack from top to
// bottom, first endpoints push onto one of four stacks and second endpoints
// must pop the matching edge from that same stack.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int MAX_CELLS = 81;
constexpr int PAGES = 4;
constexpr int STACK_WORDS = 1;
using Count = unsigned __int128;

struct Incident {
    uint8_t page = 0;
    uint8_t mate = 0;
    uint8_t edge = 0;
};

struct Vertex {
    std::vector<Incident> edges;
};

struct State {
    uint64_t placed_lo = 0;
    uint64_t placed_hi = 0;
    std::array<std::array<uint64_t, STACK_WORDS>, PAGES> stacks{};
    std::array<uint8_t, PAGES> stack_sizes{};

    bool operator==(const State& other) const {
        return placed_lo == other.placed_lo
            && placed_hi == other.placed_hi
            && stacks == other.stacks
            && stack_sizes == other.stack_sizes;
    }
};

struct StateHash {
    std::size_t operator()(const State& state) const {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t value) {
            for (int i = 0; i < 8; ++i) {
                h ^= static_cast<uint8_t>(value >> (i * 8));
                h *= 1099511628211ULL;
            }
        };
        mix(state.placed_lo);
        mix(state.placed_hi);
        for (const auto& stack : state.stacks) {
            for (uint64_t word : stack) {
                mix(word);
            }
        }
        for (uint8_t size : state.stack_sizes) {
            h ^= size;
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

struct Solver {
    int n = 0;
    int n2 = 0;
    int edge_bits = 0;
    uint64_t edge_mask = 0;
    std::vector<Vertex> vertices;
    std::unordered_map<State, Count, StateHash> memo;
    uint64_t memo_limit = 5000000;
    uint64_t calls = 0;
    uint64_t hits = 0;
    uint64_t legal_moves = 0;

    bool is_placed(const State& state, int v) const {
        if (v < 64) {
            return ((state.placed_lo >> v) & 1ULL) != 0;
        }
        return ((state.placed_hi >> (v - 64)) & 1ULL) != 0;
    }

    void set_placed(State& state, int v) const {
        if (v < 64) {
            state.placed_lo |= 1ULL << v;
        } else {
            state.placed_hi |= 1ULL << (v - 64);
        }
    }

    uint8_t get_stack_value(const State& state, int page, int index) const {
        int bit = index * edge_bits;
        int word = bit / 64;
        int offset = bit % 64;
        uint64_t value = state.stacks[page][word] >> offset;
        if (offset + edge_bits > 64) {
            value |= state.stacks[page][word + 1] << (64 - offset);
        }
        return static_cast<uint8_t>(value & edge_mask);
    }

    void set_stack_value(State& state, int page, int index, uint8_t value) const {
        int bit = index * edge_bits;
        int word = bit / 64;
        int offset = bit % 64;
        uint64_t shifted_mask = edge_mask << offset;
        state.stacks[page][word] &= ~shifted_mask;
        state.stacks[page][word] |= (static_cast<uint64_t>(value) << offset) & shifted_mask;
        if (offset + edge_bits > 64) {
            int spill = offset + edge_bits - 64;
            uint64_t spill_mask = (1ULL << spill) - 1ULL;
            state.stacks[page][word + 1] &= ~spill_mask;
            state.stacks[page][word + 1] |= static_cast<uint64_t>(value) >> (edge_bits - spill);
        }
    }

    uint8_t stack_top(const State& state, int page) const {
        return get_stack_value(state, page, state.stack_sizes[page] - 1);
    }

    void stack_push(State& state, int page, uint8_t edge) const {
        set_stack_value(state, page, state.stack_sizes[page], static_cast<uint8_t>(edge + 1));
        ++state.stack_sizes[page];
    }

    void stack_pop(State& state, int page) const {
        --state.stack_sizes[page];
        set_stack_value(state, page, state.stack_sizes[page], 0);
    }

    bool apply_vertex(State& state, int v) const {
        for (const Incident& inc : vertices[v].edges) {
            if (is_placed(state, inc.mate)) {
                if (state.stack_sizes[inc.page] == 0 || stack_top(state, inc.page) != inc.edge + 1) {
                    return false;
                }
            }
        }

        for (const Incident& inc : vertices[v].edges) {
            if (is_placed(state, inc.mate)) {
                stack_pop(state, inc.page);
            } else {
                stack_push(state, inc.page, inc.edge);
            }
        }
        set_placed(state, v);
        return true;
    }

    Count dfs(const State& state, int placed_count) {
        ++calls;
        if (placed_count == n2) {
            for (uint8_t stack_size : state.stack_sizes) {
                if (stack_size != 0) {
                    return 0;
                }
            }
            return 1;
        }

        auto found = memo.find(state);
        if (found != memo.end()) {
            ++hits;
            return found->second;
        }

        Count total = 0;
        for (int v = 0; v < n2; ++v) {
            if (is_placed(state, v)) {
                continue;
            }
            State next = state;
            if (!apply_vertex(next, v)) {
                continue;
            }
            ++legal_moves;
            total += dfs(next, placed_count + 1);
        }

        if (memo_limit != 0 && memo.size() >= memo_limit) {
            std::cerr << "memo_limit_hit=" << memo_limit
                      << " calls=" << calls
                      << " hits=" << hits
                      << " legal_moves=" << legal_moves << "\n";
            std::exit(3);
        }
        memo.emplace(state, total);
        return total;
    }

    Count solve() {
        State state;
        apply_vertex(state, 0);
        Count top_fixed = dfs(state, 1);
        return top_fixed * static_cast<Count>(n2);
    }
};

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

double seconds_since(std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
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

uint64_t parse_u64(const char* text, const char* name) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (*text == '\0' || *end != '\0') {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        std::exit(2);
    }
    return static_cast<uint64_t>(value);
}

Solver build_solver(int n) {
    Solver solver;
    solver.n = n;
    solver.n2 = n * n;
    solver.vertices.resize(solver.n2);
    std::array<uint8_t, PAGES> edge_counts{};

    auto add_edge = [&](int a, int b, int page) {
        uint8_t edge = edge_counts[page]++;
        solver.vertices[a].edges.push_back({static_cast<uint8_t>(page), static_cast<uint8_t>(b), edge});
        solver.vertices[b].edges.push_back({static_cast<uint8_t>(page), static_cast<uint8_t>(a), edge});
    };

    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            int idx = x * n + y;
            if (x > 0) {
                add_edge(idx, (x - 1) * n + y, x % 2);
            }
            if (y > 0) {
                add_edge(idx, x * n + (y - 1), 2 + (y % 2));
            }
        }
    }

    uint8_t max_edges = 0;
    for (uint8_t count : edge_counts) {
        max_edges = std::max(max_edges, count);
    }
    solver.edge_bits = 1;
    while ((1ULL << solver.edge_bits) <= static_cast<uint64_t>(max_edges)) {
        ++solver.edge_bits;
    }
    solver.edge_mask = (1ULL << solver.edge_bits) - 1ULL;
    if ((max_edges * solver.edge_bits + 63) / 64 > STACK_WORDS) {
        std::cerr << "STACK_WORDS too small for n=" << n << "\n";
        std::exit(2);
    }
    return solver;
}

void usage(const char* exe) {
    std::cerr << "Usage: " << exe << " n\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        usage(argv[0]);
        return 1;
    }
    int n = parse_int(argv[1], "n");
    if (n < 1 || n > 9 || n * n > MAX_CELLS) {
        std::cerr << "n must be between 1 and 9\n";
        return 1;
    }

    auto start = std::chrono::steady_clock::now();
    Solver solver = build_solver(n);
    if (const char* text = std::getenv("MF_MEMO_LIMIT")) {
        solver.memo_limit = parse_u64(text, "MF_MEMO_LIMIT");
    }
    Count answer = solver.solve();
    std::cout << to_decimal(answer) << "\n";
    std::cerr << "n=" << n
              << " calls=" << solver.calls
              << " memo=" << solver.memo.size()
              << " hits=" << solver.hits
              << " legal_moves=" << solver.legal_moves
              << " total_s=" << seconds_since(start) << "\n";
    return 0;
}
