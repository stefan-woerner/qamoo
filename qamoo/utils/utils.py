import os
import fnmatch
import json
from tqdm import tqdm
import numpy as np

from qamoo.utils.data_structures import ProblemGraphBuilder

import pygmo as pg


def evaluate_sample_objectives(samples, objective_graphs):

    # evaluate all objective functions for the given samples
    n = len(samples[0])
    Qs = []
    for i, graph in enumerate(objective_graphs):
        Q = np.zeros((n, n))
        for u, v, data in graph.edges(data=True):
            Q[u, v] = data['weight']
            Q[v, u] = data['weight']
        Qs += [Q]

    return _eval_loop(samples.astype('float64'), Qs)


# speed-up objective evaluation using numba
from numba import njit
@njit(cache=True)
def _eval_loop(samples, Qs):
    n = len(samples)
    m = len(Qs)
    f_values = np.zeros((n, m))
    for i in range(n):
        x = samples[i]
        f_values[i, :] = [x@Q@(1-x) for Q in Qs]
    return f_values


def pareto_front(data: np.ndarray) -> np.ndarray:
    """Find the Pareto-efficient points."""

    is_efficient = np.arange(len(data), dtype=np.int64)
    for i in tqdm(range(len(data))):
        if i in is_efficient:
            c = data[i,:]
            non_dominated = is_efficient[np.where(np.any(data[is_efficient] > c, axis=1))[0]]
            is_efficient = np.intersect1d(non_dominated, is_efficient)
            is_efficient = np.append(is_efficient, i)
    is_efficient = np.unique(is_efficient)
    return np.sort(is_efficient)


def find_first_positions_hashing(x, y):
    # Convert y into a tuple-based dictionary for fast lookup
    y_dict = {tuple(row): idx for idx, row in enumerate(y)}

    # Lookup each row in x, return index if found, else -1
    return np.array([y_dict.get(tuple(row), -1) for row in x])
    

def compute_hypervolume_progress(problem_folder, results_folder, steps):

    # load objective lower bounds (= reference point for HV)
    with open(problem_folder + 'lower_bounds.json', 'r') as f:
        lower_bounds = json.load(f)
    lower_bounds = np.array(lower_bounds)
    
    # load objective graphs
    objective_graphs = [ProblemGraphBuilder.deserialize(problem_folder + g_f) 
                        for g_f in sorted(fnmatch.filter(os.listdir(problem_folder), 'problem_graph_*.json'))]
    
    # load samples and determine hypervolume
    print('evaluate samples objective... ', end='') 
    samples = np.load(results_folder + 'samples.npy')
    fvals = evaluate_sample_objectives(samples, objective_graphs)
    print('done.')
    
    # initialize non-dominated samples and progress
    non_dominated_samples = []
    progress = {}
    
    # iterate over steps to evaluate HV
    for j in range(1, len(steps)):
    
        # set start and end of current batch
        print(j, '/', len(steps)-1)        
        j_start = steps[j-1]
        j_end = steps[j]
        samples_batch = fvals[j_start:j_end, :]
        
        # append previous non-dominated points
        if len(non_dominated_samples) > 0:
            samples_batch = np.append(non_dominated_samples, samples_batch, axis=0)
        
        # remove duplicates while keeping order
        _, idx = np.unique(samples_batch, axis=0, return_index=True)
        samples_batch = samples_batch[np.sort(idx), :]
    
        # get non-dominated indices
        nd_batch = pareto_front(samples_batch)
        non_dominated_samples = samples_batch[nd_batch, :]
        print('#NDP =', len(non_dominated_samples))
    
        # compute current HV
        f_hv = pg.hypervolume(-non_dominated_samples)
        hv = f_hv.compute(-lower_bounds)
        progress[str(j_end)] = hv
        print('HV   = ', hv)
    
    # store progress
    with open(results_folder + 'progress.json', 'w') as f:
        json.dump(progress, f)
    
    # get first positions of non dominated samples in all samples
    positions = find_first_positions_hashing(non_dominated_samples, fvals)
    
    # sort samples according to positions
    idx = np.argsort(positions)
    non_dominated_samples = non_dominated_samples[idx]
    positions = positions[idx]
    
    # store samples and positions
    np.save(results_folder + 'non_dominated_samples.npy', non_dominated_samples)
    np.save(results_folder + 'non_dominated_positions.npy', positions)