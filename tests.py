from typing import Callable
import numpy as np
import sys
import pytest
from pytest_codspeed import BenchmarkFixture
from functools import partial

from pagerank import pagerank_multiprocess, pagerank_multithread, pagerank_single

PagerankFunc = Callable[[np.ndarray, int], np.ndarray]


def create_test_graph(size: int) -> np.ndarray:
    """Create a random graph for testing"""
    # Fixed seed
    np.random.seed(0)
    # Create random adjacency matrix with ~5 outgoing edges per node
    matrix = np.random.choice([0, 1], size=(size, size), p=[1 - 5 / size, 5 / size])

    # Find nodes with no outgoing edges
    zero_outdegree = ~matrix.any(axis=1)
    zero_indices = np.where(zero_outdegree)[0]

    # For each node with no outgoing edges, add a random edge
    if len(zero_indices) > 0:
        random_targets = np.random.randint(0, size, size=len(zero_indices))
        matrix[zero_indices, random_targets] = 1

    return matrix


@pytest.fixture(scope="session", autouse=True)
def print_gil_status():
    print()
    print(f"Running {sys.version}")
    if "_is_gil_enabled" not in dir(sys):
        print("sys._is_gil_enabled() is not available in this Python version.")
    else:
        print(f"GIL is {"enabled" if sys._is_gil_enabled() else "disabled"}")
    print()


@pytest.mark.parametrize(
    "pagerank",
    [
        pagerank_single,
        partial(pagerank_multiprocess, num_processes=8),
        partial(pagerank_multithread, num_threads=8),
    ],
    ids=["single", "8-processes", "8-threads"],
)
@pytest.mark.parametrize(
    "graph",
    [
        create_test_graph(100),
        create_test_graph(1000),
        create_test_graph(2000),
    ],
    ids=["XS", "L", "XL"],
)
def test_pagerank(
    benchmark: BenchmarkFixture,
    pagerank: PagerankFunc,
    graph: np.ndarray,
):
    benchmark(pagerank, graph, num_iterations=10)
