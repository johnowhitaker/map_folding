from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Iterable


KNOWN_SQUARE_COUNTS = {
    1: 1,
    2: 8,
    3: 1368,
    4: 300608,
    5: 186086600,
    6: 123912532224,
    7: 129950723279272,
    8: 162403827553180928,
}


@dataclass(frozen=True)
class CampaignConfig:
    n: int = 5
    prefix_depth: int = 14
    prefixes_per_unit: int = 64

    @property
    def n2(self) -> int:
        return self.n * self.n

    @property
    def factor(self) -> int:
        if self.n <= 1:
            return 1
        return 2 * self.n2

    @property
    def campaign_id(self) -> str:
        return f"raw-n{self.n}-d{self.prefix_depth}-p{self.prefixes_per_unit}-v1"


def build_spiral_map(width: int, height: int) -> list[list[int]]:
    grid = [[width * height for _ in range(height)] for _ in range(width)]
    x = 0
    y = 0
    dx = 1
    dy = 0
    for i in range(width * height - 1, -1, -1):
        grid[x][y] = i
        x1 = x + dx
        y1 = y + dy
        if (
            x1 < 0
            or x1 >= width
            or y1 < 0
            or y1 >= height
            or grid[x1][y1] < width * height
        ):
            dy, dx = dx, -dy
            x += dx
            y += dy
        else:
            x = x1
            y = y1
    return grid


def make_edges(n: int) -> list[list[int]]:
    n2 = n * n
    edges = [[n2 for _ in range(n2)] for _ in range(4)]
    grid = build_spiral_map(n, n)
    for x in range(n):
        for y in range(n):
            idx = grid[x][y]
            if x > 0:
                edge = edges[x % 2]
                other = grid[x - 1][y]
                edge[idx] = other
                edge[other] = idx
            if y > 0:
                edge = edges[2 + (y % 2)]
                other = grid[x][y - 1]
                edge[idx] = other
                edge[other] = idx
    return edges


def good_mask(cycle: list[int], i: int, edges: list[list[int]]) -> int:
    good = (1 << i) - 1
    for edge in edges:
        other = edge[i]
        if other >= i:
            continue

        allowed = 1 << other
        j = cycle[other]
        while j != other:
            j1 = edge[j]
            if j1 < i:
                j = j1
            allowed |= 1 << j
            j = cycle[j]

        good &= allowed
        if good == 0:
            return 0
    return good


def initial_cycle(n2: int) -> list[int]:
    cycle = [0] * n2
    if n2 >= 3:
        cycle[0] = 1
        cycle[1] = 2
        cycle[2] = 0
    return cycle


def encode_prefix(cycle: list[int], depth: int) -> str:
    return base64.b64encode(bytes(cycle[:depth])).decode("ascii")


def decode_prefix(prefix: str, n2: int) -> list[int]:
    raw = base64.b64decode(prefix.encode("ascii"))
    cycle = [0] * n2
    for i, value in enumerate(raw):
        cycle[i] = value
    return cycle


def generate_prefixes(n: int, depth: int) -> list[str]:
    if n < 2:
        return [encode_prefix([0], 1)]
    n2 = n * n
    if depth < 3 or depth > n2:
        raise ValueError(f"prefix depth must be between 3 and {n2}")

    edges = make_edges(n)
    cycle = initial_cycle(n2)
    prefixes: list[str] = []

    def visit(i: int) -> None:
        if i == depth:
            prefixes.append(encode_prefix(cycle, depth))
            return

        good = good_mask(cycle, i, edges)
        while good:
            bit = good & -good
            j = bit.bit_length() - 1
            good -= bit
            nxt = cycle[j]
            cycle[i] = nxt
            cycle[j] = i
            visit(i + 1)
            cycle[j] = nxt

    visit(3)
    return prefixes


def count_from_cycle(cycle: list[int], i: int, edges: list[list[int]], n2: int) -> int:
    good = good_mask(cycle, i, edges)
    if good == 0:
        return 0
    if i == n2 - 1:
        return good.bit_count()

    total = 0
    while good:
        bit = good & -good
        j = bit.bit_length() - 1
        good -= bit
        nxt = cycle[j]
        cycle[i] = nxt
        cycle[j] = i
        total += count_from_cycle(cycle, i + 1, edges, n2)
        cycle[j] = nxt
    return total


def count_prefix_raw(n: int, depth: int, prefix: str) -> int:
    n2 = n * n
    cycle = decode_prefix(prefix, n2)
    return count_from_cycle(cycle, depth, make_edges(n), n2)


def count_payload_raw(payload: dict) -> int:
    n = int(payload["n"])
    depth = int(payload["depth"])
    edges = make_edges(n)
    n2 = n * n
    total = 0
    for prefix in payload["prefixes"]:
        cycle = decode_prefix(prefix, n2)
        total += count_from_cycle(cycle, depth, edges, n2)
    return total


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]

