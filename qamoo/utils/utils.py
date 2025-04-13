import fnmatch
import json
import os.path
from collections.abc import Sequence
from pathlib import Path
from tqdm import tqdm

import networkx as nx
import numpy as np


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
