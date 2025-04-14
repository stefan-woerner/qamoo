from collections.abc import Sequence
from joblib import parallel_config, Parallel, delayed
import sys
import os
file_path = os.path.abspath(os.path.join('..', 'qamoo', 'utils'))
sys.path.append(file_path)
from graphs import evaluate_sample, linearize_graphs
from data_structures import ProblemGraphBuilder
from itertools import product
import numpy as np

def generate_bitstrings(n):
    for v in product([0, 1], repeat=n):
        yield v


def run_problem_instance(problem_set_index: int, num_swaps: int, num_obj: int, num_qubits: int) -> None:
    """Runs a single problem instance"""
    print(f'Starting {problem_set_index}th problem set for {num_swaps} swaps and {num_obj} objectives', flush=True)
    bitstrings = generate_bitstrings(num_qubits)
    problem_directory = f'../data/problems/{num_qubits}q/problem_set_{num_qubits}q_{num_swaps}s_{num_obj}o_{problem_set_index}'
    problem_graphs = [ProblemGraphBuilder.deserialize(f'{problem_directory}/problem_graph_{idx}.json')
                      for idx in range(num_obj)]
    avg_graph = linearize_graphs(problem_graphs, [1 / num_obj] * num_obj)
    objectives = []
    for bs in bitstrings:
        fval = evaluate_sample(bs, avg_graph)
        objectives.append(fval)
    file = f"maxcut_{num_swaps}s_{num_obj}o_{problem_set_index}_{num_qubits}q.npz"
    np.savez_compressed("cost_values/"+file, data=np.array(objectives))
    print(f'{problem_set_index}th problem set for {num_swaps} swaps and {num_obj} objectives is finished', flush=True)


def write_to_files(num_objectives: Sequence[int], num_swap_layers: Sequence[int],
                   problem_set_indices: Sequence[int]) -> None:
    num_qubits = 27
    for num_obj in num_objectives:
        for num_s in num_swap_layers:
            with parallel_config(backend='multiprocessing', n_jobs=20):
                # run problem instances in parallel
                Parallel()(
                    delayed(run_problem_instance)(p_i, num_s, num_obj, num_qubits) for p_i in problem_set_indices)


if __name__ == '__main__':
	write_to_files(num_objectives=[3, 4], num_swap_layers=[0], problem_set_indices=[0, 1, 2, 3, 4])
