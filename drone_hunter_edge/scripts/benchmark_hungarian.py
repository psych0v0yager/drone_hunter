#!/usr/bin/env python3
"""Benchmark different Hungarian algorithm implementations."""

import numpy as np
import time
import sys
from typing import List, Tuple
from collections import deque
from dataclasses import dataclass

# Import current implementation
sys.path.insert(0, '.')
from core.kalman_tracker import hungarian_algorithm as current_hungarian

# Try to import scipy
try:
    from scipy.optimize import linear_sum_assignment as scipy_hungarian
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not available, skipping scipy benchmark")


# Kwok algorithm implementation from GitHub
@dataclass
class Matching:
    left_pairs: List[int]
    right_pairs: List[int]
    total_weight: int


def kwok_algorithm(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kwok algorithm wrapper that matches scipy/current interface.

    Converts cost matrix to adjacency list format and runs kwok matching.
    Returns (row_indices, col_indices) like scipy.
    """
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Convert cost to weight (negate for max matching, add offset to keep positive)
    max_cost = cost_matrix.max()
    weight_matrix = (max_cost - cost_matrix).astype(np.int64)

    # Build adjacency list
    adj = []
    for i in range(n_rows):
        edges = [(j, int(weight_matrix[i, j])) for j in range(n_cols)]
        adj.append(edges)

    result = kwok(n_rows, n_cols, adj)

    rows = []
    cols = []
    for i, j in enumerate(result.left_pairs):
        if j >= 0 and j < n_cols:
            rows.append(i)
            cols.append(j)

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


def kwok(L_size: int, R_size: int, adj: List[List[Tuple[int, int]]]) -> Matching:
    """Kwok maximum weight matching algorithm."""
    left_pairs = [-1] * L_size
    right_pairs = [-1] * R_size
    right_parents = [-1] * R_size
    right_visited = [False] * R_size

    visited_lefts = []
    visited_rights = []
    on_edge_rights = []
    right_on_edge = [False] * R_size

    left_labels = [max((w for _, w in edges), default=0) for edges in adj]
    right_labels = [0] * R_size
    slacks = [sys.maxsize] * R_size
    q = deque()

    def advance(r: int) -> bool:
        right_on_edge[r] = False
        right_visited[r] = True
        visited_rights.append(r)
        l = right_pairs[r]
        if l != -1:
            q.append(l)
            visited_lefts.append(l)
            return False
        current_r = r
        while current_r != -1:
            l = right_parents[current_r]
            prev_r = left_pairs[l]
            left_pairs[l] = current_r
            right_pairs[current_r] = l
            current_r = prev_r
        return True

    def bfs_until_applies_augment_path(first_unmatched_r: int):
        while True:
            while q:
                l = q.popleft()
                if left_labels[l] == 0:
                    right_parents[first_unmatched_r] = l
                    if advance(first_unmatched_r):
                        return
                if slacks[first_unmatched_r] > left_labels[l]:
                    slacks[first_unmatched_r] = left_labels[l]
                    right_parents[first_unmatched_r] = l
                    if not right_on_edge[first_unmatched_r]:
                        on_edge_rights.append(first_unmatched_r)
                        right_on_edge[first_unmatched_r] = True

                for r, w in adj[l]:
                    if right_visited[r]:
                        continue
                    diff = left_labels[l] + right_labels[r] - w
                    if diff == 0:
                        right_parents[r] = l
                        if advance(r):
                            return
                    elif slacks[r] > diff:
                        right_parents[r] = l
                        slacks[r] = diff
                        if not right_on_edge[r]:
                            on_edge_rights.append(r)
                            right_on_edge[r] = True

            delta = sys.maxsize
            for r in on_edge_rights:
                if right_on_edge[r]:
                    delta = min(delta, slacks[r])

            for l in visited_lefts:
                left_labels[l] -= delta
            for r in visited_rights:
                right_labels[r] += delta
            for r in on_edge_rights:
                if right_on_edge[r]:
                    slacks[r] -= delta
                    if slacks[r] == 0 and advance(r):
                        return

    # Initial greedy matching
    for l in range(L_size):
        for r, w in adj[l]:
            if right_pairs[r] == -1 and left_labels[l] + right_labels[r] == w:
                left_pairs[l] = r
                right_pairs[r] = l
                break

    # Augment unmatched
    for l in range(L_size):
        if left_pairs[l] != -1:
            continue
        q.clear()
        for r in visited_rights:
            right_visited[r] = False
        for r in on_edge_rights:
            right_on_edge[r] = False
            slacks[r] = sys.maxsize
        visited_lefts.clear()
        visited_rights.clear()
        on_edge_rights.clear()

        visited_lefts.append(l)
        q.append(l)

        # Find first unmatched right
        first_unmatched_r = -1
        for r in range(R_size):
            if right_pairs[r] == -1:
                first_unmatched_r = r
                break

        if first_unmatched_r >= 0:
            bfs_until_applies_augment_path(first_unmatched_r)

    total = 0
    for l in range(L_size):
        if left_pairs[l] != -1:
            for r, w in adj[l]:
                if r == left_pairs[l]:
                    total += w
                    break

    return Matching(left_pairs=left_pairs, right_pairs=right_pairs, total_weight=total)


def benchmark(func, cost_matrix, n_runs=100):
    """Run benchmark and return average time in microseconds."""
    # Warmup
    for _ in range(5):
        func(cost_matrix.copy())

    times = []
    for _ in range(n_runs):
        m = cost_matrix.copy()
        t0 = time.perf_counter()
        func(m)
        times.append(time.perf_counter() - t0)

    return np.mean(times) * 1e6, np.std(times) * 1e6


def verify_results(cost_matrix):
    """Verify all implementations give same total cost."""
    rows1, cols1 = current_hungarian(cost_matrix.copy())
    cost1 = sum(cost_matrix[r, c] for r, c in zip(rows1, cols1))

    results = {"current": cost1}

    if HAS_SCIPY:
        rows2, cols2 = scipy_hungarian(cost_matrix.copy())
        cost2 = sum(cost_matrix[r, c] for r, c in zip(rows2, cols2))
        results["scipy"] = cost2

    try:
        rows3, cols3 = kwok_algorithm(cost_matrix.copy())
        cost3 = sum(cost_matrix[r, c] for r, c in zip(rows3, cols3))
        results["kwok"] = cost3
    except Exception as e:
        results["kwok"] = f"error: {e}"

    return results


def main():
    print("Hungarian Algorithm Benchmark")
    print("=" * 60)

    # Test sizes typical for drone tracking
    sizes = [
        (1, 1),
        (2, 3),
        (3, 5),
        (5, 5),
        (5, 10),
        (10, 10),
        (10, 20),
        (20, 20),
    ]

    np.random.seed(42)

    print("\nVerifying correctness (total assignment cost):")
    print("-" * 40)
    test_matrix = np.random.rand(5, 5).astype(np.float32) * 100
    results = verify_results(test_matrix)
    for name, cost in results.items():
        print(f"  {name}: {cost}")

    print("\n\nBenchmark Results (microseconds per call):")
    print("-" * 60)
    print(f"{'Size':>10} | {'Current':>12} | {'Scipy':>12} | {'Kwok':>12}")
    print("-" * 60)

    for n_rows, n_cols in sizes:
        cost_matrix = np.random.rand(n_rows, n_cols).astype(np.float32) * 100

        # Current implementation
        t_current, std_current = benchmark(current_hungarian, cost_matrix)

        # Scipy
        if HAS_SCIPY:
            t_scipy, std_scipy = benchmark(scipy_hungarian, cost_matrix)
        else:
            t_scipy, std_scipy = float('nan'), float('nan')

        # Kwok
        try:
            t_kwok, std_kwok = benchmark(kwok_algorithm, cost_matrix)
        except Exception as e:
            t_kwok, std_kwok = float('nan'), float('nan')
            print(f"Kwok error at {n_rows}x{n_cols}: {e}")

        print(f"{n_rows}x{n_cols:>3} | {t_current:>8.1f} ± {std_current:>3.0f} | "
              f"{t_scipy:>8.1f} ± {std_scipy:>3.0f} | {t_kwok:>8.1f} ± {std_kwok:>3.0f}")

    print("\n" + "=" * 60)
    print("Lower is better. Times in microseconds (μs).")


if __name__ == "__main__":
    main()
