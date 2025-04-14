import os
import numpy as np
import json
import networkx as nx
from collections.abc import Sequence
from tqdm import tqdm
from time import time

from qamoo.utils.data_structures import ProblemGraphBuilder
from qamoo.utils.graphs import linearize_graphs
from qamoo.utils.utils import save_results, write_runtime

from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer
from qiskit_optimization.applications import Maxcut

import gurobipy as gp
from gurobipy import GRB
from gurobipy import tupledict, Var


def random(config, overwrite_results=False):
    
    t1 = time()

    # get problem specification
    problem = config.problem
    problem_folder = problem.problem_folder

    # load problem
    objective_graphs = problem.objective_graphs
    num_qubits = len(objective_graphs[0].nodes)
    
    # create results directory and raise exception if it already exists
    try:
        os.makedirs(config.results_folder)
    except FileExistsError:
        if overwrite_results:
            print('Folder exists!')
        else:
            raise FileExistsError()
    
    t2 = time()

    # generate random samples
    samples = np.random.randint(low=0, high=2, size=(config.num_samples, num_qubits), dtype='b')

    t3 = time()

    # save results into results folder
    save_results(config.results_folder, samples, None)
    write_runtime(config.results_folder, t2-t1, t3-t2)


def _graph2f(graph: nx.Graph, x: tupledict[int, Var]) -> gp.QuadExpr:
    """Returns the Maxcut objective function of a given graph for gurobi applications."""
    return sum(x[u] * (1 - x[v]) * data['weight'] + x[v] * (1 - x[u]) * data['weight'] for u, v, data in
               graph.edges(data=True))


def _gurobi_objective_function(graphs: list[nx.Graph],
                              x: tupledict[int, Var], c_values: list[float]) -> gp.QuadExpr:
    """Assumes 1-1 correspondence between graphs and c values.
    Creates a gurobi quadratic expression representing the MaxCut objective function
    corresponding to the linearization of given graphs."""
    assert len(graphs) == len(c_values), (f"Length of graphs ({len(graphs)})"
                                          f" must be equal to length of c values ({len(c_values)}).")
    graph = linearize_graphs(graphs, c_values)
    return _graph2f(graph, x)


def _get_gurobi_samples(linearizations: list[list[float]],
                       problem_graphs: list[nx.Graph]) -> np.ndarray:
    """Returns a (num_linearizations, num_problem_graphs)-shaped array of Gurobi-optimal solutions."""
    samples = []
    for lin_vec in tqdm(linearizations):
        model = gp.Model('MaxCut')
        model.Params.LogToConsole = 0
        model.params.Threads = 1
        x = model.addVars(len(problem_graphs[0].nodes), vtype=GRB.BINARY, name='x')
        f = _gurobi_objective_function(problem_graphs, x, lin_vec)
        model.setObjective(f, sense=GRB.MAXIMIZE)
        model.optimize()
        solution = model.x
        samples.append(solution)
    return np.array(samples, dtype='b')


def weighted_sum(config, overwrite_results=False):
    
    t1 = time()

    # get problem specification
    problem = config.problem
    problem_folder = problem.problem_folder

    # load problem
    objective_graphs = problem.objective_graphs
    
    # create results directory and raise exception if it already exists
    try:
        os.makedirs(config.results_folder)
    except FileExistsError:
        if overwrite_results:
            print('Folder exists!')
        else:
            raise FileExistsError()
    
    # load linearizations
    objective_weights = config.objective_weights
    
    t2 = time()

    # solve problems and store samples in binary format
    samples = _get_gurobi_samples(objective_weights, objective_graphs)

    t3 = time()

    # save results into results folder
    save_results(config.results_folder, samples, objective_weights)
    write_runtime(config.results_folder, t2-t1, t3-t2)
    

def _get_gwo_samples(num_cuts: int , c_values: np.ndarray, problem_graphs: Sequence[nx.Graph]) -> np.ndarray:
    gwo = GoemansWilliamsonOptimizer(num_cuts=num_cuts, unique_cuts=False)

    gwo_pgs = (linearize_graphs(problem_graphs, c_vector) for c_vector in c_values)
    gwo_qps = (Maxcut(g).to_quadratic_program() for g in gwo_pgs)

    gwo_samples = []
    for qp in gwo_qps:
        samples = gwo.solve(qp).samples
        samples = [s.x for s in samples]
        gwo_samples.append(samples)
    return np.array(gwo_samples, dtype='b')


def goemans_williamson(config, overwrite_results=False):
    
    t1 = time()

    # get problem specification
    problem = config.problem
    problem_folder = problem.problem_folder

    # load problem
    objective_graphs = problem.objective_graphs
    
    # create results directory and raise exception if it already exists
    try:
        os.makedirs(config.results_folder)
    except FileExistsError:
        if overwrite_results:
            print('Folder exists!')
        else:
            raise FileExistsError()
        
    # load linearizations
    objective_weights = config.objective_weights

    t2 = time()

    # solve problems via Goemans-Williamson
    gwo_samples = _get_gwo_samples(config.shots, objective_weights, objective_graphs)
    samples = []
    for i in range(gwo_samples.shape[0]):
        samples += list(gwo_samples[i, :, :])
    samples = np.array([[int(x) for x in sample] for sample in list(samples)], dtype='b')
    
    t3 = time()

    # save results into results folder
    save_results(config.results_folder, samples, objective_weights)
    write_runtime(config.results_folder, t2-t1, t3-t2)


def _mo_maxcut_milp_gurobi(graphs, c, lb):

    model = gp.Model('mo_maxcut')
    model.Params.LogToConsole = 0
    model.params.Threads = 1

    x = [model.addVar(vtype=GRB.BINARY, name=f'x_{i}') for i in range(len(graphs[0].nodes))]

    e = {(i, j): model.addVar(vtype=GRB.BINARY, name=f'e_{i}_{j}') for (i, j) in graphs[0].edges}
    for edge in graphs[0].edges:
        model.addConstr(e[edge] <= x[edge[0]] + x[edge[1]])
        model.addConstr(e[edge] <= 2 - x[edge[0]] - x[edge[1]])
        model.addConstr(e[edge] >= x[edge[0]] - x[edge[1]])
        model.addConstr(e[edge] >= x[edge[1]] - x[edge[0]])

    objectives = []
    for k, graph in enumerate(graphs):        
        objectives += [sum([e[edge[:2]]*edge[2]['weight'] for edge in graph.edges(data=True)])]
        model.addConstr(objectives[k] >= lb[k])

    weighted_objective = sum([c[k] * objectives[k] for k in range(len(graphs))])
    model.setObjective(weighted_objective, GRB.MAXIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return [x_.X for x_ in x]
    elif model.Status == GRB.INFEASIBLE:
        return None
    else:
        raise ValueError(f'Unexpected GUROBI model status: {model.Status}')


def random_eps_constraint(config, overwrite_results=False):

    t1 = time()

    # load problem
    objective_graphs = config.problem.objective_graphs

    # folder to store results to
    results_folder = config.results_folder
    
    # create results directory and raise exception if it already exists
    try:
        os.makedirs(results_folder)
    except FileExistsError:
        if overwrite_results:
            print('Folder exists!')
        else:
            raise FileExistsError()
    
    # load linearizations
    objective_weights = config.objective_weights
    
    # load input samples
    lower_bounds = config.problem.lower_bounds
    upper_bounds = config.problem.upper_bounds
    np.random.seed(config.eps_seed)
    eps_samples = np.random.rand(config.total_num_samples, config.problem.num_objectives)
    eps_samples = (upper_bounds - lower_bounds) * eps_samples + lower_bounds
    
    t2 = time()

    samples = []
    num_infeasible = 0
    for i in tqdm(range(config.total_num_samples)):
        eps_sample = eps_samples[i]  # use this as lower bound for the objectives
        j = i // config.shots
        c = objective_weights[j]
        sample = _mo_maxcut_milp_gurobi(objective_graphs, c, eps_sample)
        if sample is None:
            num_infeasible += 1
            if len(samples) > 0:
                # if there are already feasible samples, just fill up with the previous one
                sample = samples[-1]
            else:
                # if already the very first sample is infeasible, just take the lower bound as eps constraint
                sample = _mo_maxcut_milp_gurobi(objective_graphs, c, lower_bounds)
        
        samples += [sample]

    samples = np.array(samples, dtype='b')

    t3 = time()

    # save results into results folder
    save_results(results_folder, samples, objective_weights)
    write_runtime(config.results_folder, t2-t1, t3-t2)

    # write eps_samples and fraction of feasible solutions
    np.save(results_folder + 'eps_samples.npy', eps_samples)
    with open(results_folder + 'hv_estimate.json', 'w') as f:
        feasible_fraction = 1 - num_infeasible/len(samples)
        v_bounds = np.prod(upper_bounds - lower_bounds)
        hv_estimate = {
            'feasible_fraction': feasible_fraction,
            'v_bounds': v_bounds,
            'hv_estimate': v_bounds * feasible_fraction,
            'num_eps_samples': len(samples)
        }
        json.dump(hv_estimate, f)
