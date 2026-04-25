#include <stdint.h>

#define MAX_N 100
#define MAX_SIDE 9
#define MAX_CASES 256
#define CASE_STRIDE 112

typedef struct {
    uint64_t lo;
    uint64_t hi;
} U128;

static uint8_t case_buffer[MAX_CASES * CASE_STRIDE];
static uint32_t output_words[8];
static uint8_t edges[4][MAX_N];
static uint8_t cycle_work[MAX_N];
static int grid[MAX_SIDE][MAX_SIDE];
static uint64_t node_count;

uint32_t input_ptr(void) {
    return (uint32_t)(uintptr_t)case_buffer;
}

uint32_t output_ptr(void) {
    return (uint32_t)(uintptr_t)output_words;
}

uint32_t case_stride(void) {
    return CASE_STRIDE;
}

uint32_t max_cases(void) {
    return MAX_CASES;
}

static U128 u128_zero(void) {
    U128 value = {0, 0};
    return value;
}

static U128 u128_one(void) {
    U128 value = {1, 0};
    return value;
}

static U128 low_bits(int count) {
    if (count <= 0) {
        return u128_zero();
    }
    if (count >= 128) {
        U128 value = {~0ULL, ~0ULL};
        return value;
    }
    if (count < 64) {
        U128 value = {(1ULL << count) - 1ULL, 0};
        return value;
    }
    if (count == 64) {
        U128 value = {~0ULL, 0};
        return value;
    }
    U128 value = {~0ULL, (1ULL << (count - 64)) - 1ULL};
    return value;
}

static U128 bit_mask(int bit) {
    if (bit < 64) {
        U128 value = {1ULL << bit, 0};
        return value;
    }
    U128 value = {0, 1ULL << (bit - 64)};
    return value;
}

static int u128_is_zero(U128 value) {
    return value.lo == 0 && value.hi == 0;
}

static void u128_and_in_place(U128* value, U128 other) {
    value->lo &= other.lo;
    value->hi &= other.hi;
}

static void u128_or_in_place(U128* value, U128 other) {
    value->lo |= other.lo;
    value->hi |= other.hi;
}

static void u128_add_in_place(U128* value, U128 other) {
    uint64_t old = value->lo;
    value->lo += other.lo;
    value->hi += other.hi + (value->lo < old);
}

static U128 u128_shl1(U128 value) {
    U128 out = {value.lo << 1, (value.hi << 1) | (value.lo >> 63)};
    return out;
}

static void u128_add_mul_u64(U128* value, U128 addend, uint64_t multiplier) {
    while (multiplier != 0) {
        if ((multiplier & 1ULL) != 0) {
            u128_add_in_place(value, addend);
        }
        multiplier >>= 1;
        if (multiplier != 0) {
            addend = u128_shl1(addend);
        }
    }
}

static int ctz_mask(U128 value) {
    if (value.lo != 0) {
        return __builtin_ctzll(value.lo);
    }
    return 64 + __builtin_ctzll(value.hi);
}

static void clear_lowest_bit(U128* value) {
    if (value->lo != 0) {
        value->lo &= value->lo - 1ULL;
    } else {
        value->hi &= value->hi - 1ULL;
    }
}

static void build_map(int n) {
    int n2 = n * n;
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            grid[x][y] = n2;
        }
    }

    int x = 0;
    int y = 0;
    int dx = 1;
    int dy = 0;
    for (int i = n2 - 1; i >= 0; --i) {
        grid[x][y] = i;
        int x1 = x + dx;
        int y1 = y + dy;
        if (x1 < 0 || x1 >= n || y1 < 0 || y1 >= n || grid[x1][y1] < n2) {
            int old_dy = dy;
            dy = dx;
            dx = -old_dy;
            x += dx;
            y += dy;
        } else {
            x = x1;
            y = y1;
        }
    }
}

static void make_edges(int n) {
    int n2 = n * n;
    for (int page = 0; page < 4; ++page) {
        for (int i = 0; i < n2; ++i) {
            edges[page][i] = (uint8_t)n2;
        }
    }

    build_map(n);
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            int idx = grid[x][y];
            if (x > 0) {
                int page = x % 2;
                int other = grid[x - 1][y];
                edges[page][idx] = (uint8_t)other;
                edges[page][other] = (uint8_t)idx;
            }
            if (y > 0) {
                int page = 2 + (y % 2);
                int other = grid[x][y - 1];
                edges[page][idx] = (uint8_t)other;
                edges[page][other] = (uint8_t)idx;
            }
        }
    }
}

static U128 get_good(const uint8_t* cycle, int i) {
    U128 good = low_bits(i);
    for (int page = 0; page < 4; ++page) {
        const uint8_t* edge = edges[page];
        int other = edge[i];
        if (other >= i) {
            continue;
        }

        U128 allowed = bit_mask(other);
        int j = cycle[other];
        while (j != other) {
            int j1 = edge[j];
            if (j1 < i) {
                j = j1;
            }
            u128_or_in_place(&allowed, bit_mask(j));
            j = cycle[j];
        }

        u128_and_in_place(&good, allowed);
        if (u128_is_zero(good)) {
            return u128_zero();
        }
    }
    return good;
}

static U128 recurse_count(uint8_t* cycle, int i, int stop_i) {
    ++node_count;
    if (i == stop_i) {
        return u128_one();
    }

    U128 good = get_good(cycle, i);
    U128 out = u128_zero();
    while (!u128_is_zero(good)) {
        int j = ctz_mask(good);
        clear_lowest_bit(&good);
        uint8_t next = cycle[j];
        cycle[i] = next;
        cycle[j] = (uint8_t)i;
        u128_add_in_place(&out, recurse_count(cycle, i + 1, stop_i));
        cycle[j] = next;
    }
    return out;
}

static uint64_t read_u64_le(const uint8_t* ptr) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | ptr[i];
    }
    return value;
}

static void write_output(U128 count, uint64_t nodes, uint32_t status) {
    output_words[0] = (uint32_t)count.lo;
    output_words[1] = (uint32_t)(count.lo >> 32);
    output_words[2] = (uint32_t)count.hi;
    output_words[3] = (uint32_t)(count.hi >> 32);
    output_words[4] = (uint32_t)nodes;
    output_words[5] = (uint32_t)(nodes >> 32);
    output_words[6] = status;
    output_words[7] = 0;
}

uint32_t solve_cases(uint32_t n, uint32_t start_i, uint32_t stop_i, uint32_t case_count) {
    if (n < 2 || n > MAX_SIDE || case_count > MAX_CASES) {
        write_output(u128_zero(), 0, 1);
        return 1;
    }
    uint32_t n2 = n * n;
    if (start_i > stop_i || stop_i > n2 || n2 > MAX_N) {
        write_output(u128_zero(), 0, 2);
        return 2;
    }

    make_edges((int)n);
    node_count = 0;
    U128 total = u128_zero();

    for (uint32_t case_i = 0; case_i < case_count; ++case_i) {
        uint8_t* item = case_buffer + case_i * CASE_STRIDE;
        for (uint32_t i = 0; i < n2; ++i) {
            cycle_work[i] = item[i];
        }
        uint64_t multiplier = read_u64_le(item + MAX_N);
        u128_add_mul_u64(&total, recurse_count(cycle_work, (int)start_i, (int)stop_i), multiplier);
    }

    write_output(total, node_count, 0);
    return 0;
}
