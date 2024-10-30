import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading


DAMPING = 0.85


def pagerank_single(matrix: np.ndarray, num_iterations: int) -> np.ndarray:
    """Single-threaded PageRank implementation"""
    size = matrix.shape[0]
    # Initialize scores
    scores = np.ones(size) / size

    for _ in range(num_iterations):
        new_scores = np.zeros(size)
        for i in range(size):
            # Get nodes that point to current node
            incoming = np.where(matrix[:, i])[0]
            for j in incoming:
                # Add score contribution from incoming node
                new_scores[i] += scores[j] / np.sum(matrix[j])

        # Apply damping factor
        new_scores = (1 - DAMPING) / size + DAMPING * new_scores
        scores = new_scores

    return scores


def _process_chunk(
    matrix: np.ndarray, scores: np.ndarray, start_idx: int, end_idx: int
) -> np.ndarray:
    """Helper function for multiprocessing implementation"""
    size = matrix.shape[0]
    chunk_scores = np.zeros(size)

    for i in range(start_idx, end_idx):
        incoming = np.where(matrix[:, i])[0]
        for j in incoming:
            chunk_scores[i] += scores[j] / np.sum(matrix[j])

    return chunk_scores


def pagerank_multiprocess(
    matrix: np.ndarray, num_iterations: int, num_processes: int
) -> np.ndarray:
    """Multi-process PageRank implementation"""
    size = matrix.shape[0]
    scores = np.ones(size) / size

    # Split work into chunks
    chunk_size = size // num_processes
    chunks = [
        (matrix, scores, i, min(i + chunk_size, size))
        for i in range(0, size, chunk_size)
    ]

    for _ in range(num_iterations):
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process chunks in parallel
            chunk_results = pool.starmap(_process_chunk, chunks)
            # Combine results
            new_scores = sum(chunk_results)
            new_scores = (1 - DAMPING) / size + DAMPING * new_scores
            scores = new_scores

    return scores


def _thread_worker(
    matrix: np.ndarray,
    scores: np.ndarray,
    new_scores: np.ndarray,
    start_idx: int,
    end_idx: int,
    lock: threading.Lock,
):
    """Helper function for multi-threaded implementation"""
    size = matrix.shape[0]
    local_scores = np.zeros(size)

    for i in range(start_idx, end_idx):
        incoming = np.where(matrix[:, i])[0]
        for j in incoming:
            local_scores[i] += scores[j] / np.sum(matrix[j])

    with lock:
        new_scores += local_scores


def pagerank_multithread(
    matrix: np.ndarray, num_iterations: int, num_threads: int
) -> np.ndarray:
    """Multi-threaded PageRank implementation"""
    size = matrix.shape[0]
    scores = np.ones(size) / size

    # Split work into chunks
    chunk_size = size // num_threads
    chunks = [(i, min(i + chunk_size, size)) for i in range(0, size, chunk_size)]

    for _ in range(num_iterations):
        new_scores = np.zeros(size)
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Process chunks in parallel
            executor.map(
                lambda args: _thread_worker(*args),  # starmap isn't available
                [
                    (matrix, scores, new_scores, start_idx, end_idx, lock)
                    for start_idx, end_idx in chunks
                ],
            )
        # Apply damping factor
        new_scores = (1 - DAMPING) / size + DAMPING * new_scores
        scores = new_scores

    return scores
