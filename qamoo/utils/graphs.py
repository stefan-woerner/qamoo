import json
import os
import random
from collections.abc import Sequence

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx import maximal_matching
from networkx.generators.random_graphs import random_regular_graph
from networkx.readwrite import json_graph
from qiskit.transpiler import CouplingMap


def evaluate_sample(x: Sequence[int], graph: nx.Graph) -> float:
    """
    Return the cut value of the given graph specified by the binary vector x,
    where x[k] is the binary assignment value of the kth node in the graph.

    :param x: The binary sample.
    :param graph: The graph subject to the Maxcut problem.
                  Required to have edge weights specified by the edge attribute 'weight'.
    :return: The sum of weights of the cut edges.
    """
    assert len(x) == len(graph.nodes), "The length of x must coincide with the number of nodes in the graph."
    return sum(x[u] * (1 - x[v]) * data['weight'] + x[v] * (1 - x[u]) * data['weight'] for u, v, data in
               graph.edges(data=True))


def linearize_graphs(graphs: list[nx.Graph], c_values: list[float]) -> nx.Graph:
    """
    Creates a single graph as a linearization of the given graphs.
    Assumes 1-1 correspondence between graphs and c values.
    Assumes same topology among graphs, i.e. all graphs share the same edge set but differ in edge weights.
    Graphs are required to have edge weights specified by the edge attribute 'weight'.

    :param graphs: Graphs to be combined.
    :param c_values: The weights for the linearization.
    :return: Linearized graph.
    """
    assert len(graphs) == len(
        c_values), f"Length of graphs ({len(graphs)}) must be equal to length of c values ({len(c_values)})."

    graph = nx.Graph()
    graph.add_edges_from(graphs[0].edges())
    edge_attributes = {(u, v): {'weight': sum(c * graph[u][v]['weight'] for c, graph in zip(c_values, graphs))} for u, v
                       in graphs[0].edges()}
    nx.set_edge_attributes(graph, edge_attributes)
    return graph


def weighted_graph_from_coupling_map(coupling_map: CouplingMap,
                                     weight_dist: str, random_weight_seed: int | None = None):
    """Creates a weighted graph from the edge list of a coupling map according to the given weight distribution."""
    return weighted_graph_from_edge_list(coupling_map.get_edges(), weight_dist, random_weight_seed)


def weighted_graph_from_edge_list(edge_list: Sequence[tuple[int, int]], weight_dist: str,
                                  random_weight_seed: int = None) -> nx.Graph:
    """Creates a graph from an edge list with randomly sampled weights
    according to the given distribution ('uniform' or 'normal')."""
    edges_num = len(edge_list)
    if weight_dist == "uniform":
        random.seed(random_weight_seed)
        weights = random.choices([-1, 1], k=edges_num)
    elif weight_dist == "normal":
        np.random.seed(random_weight_seed)
        weights = np.random.normal(size=edges_num)
    else:
        raise ValueError(f"{weight_dist} is now a valid weight distribution.")

    graph = nx.Graph()
    graph.add_edges_from(edge_list)
    nx.set_edge_attributes(graph, {(e, v): {"weight": w} for (e, v), w in zip(edge_list, weights)})
    return graph


def weighted_graphs_from_edge_list(n: int, edge_list: Sequence[tuple[int, int]],
                                   weight_dist: str, random_weight_seeds: list[int] = None) -> list[nx.Graph]:
    """Creates n graphs from an edge list with randomly sampled weights
    according to the given distribution ('uniform' or 'normal')."""
    if not random_weight_seeds:
        random_weight_seeds = [None] * n
    assert len(random_weight_seeds) == n, (f"Length of random weight seeds ({len(random_weight_seeds)})"
                                           f" should coincide with n ({n}).")
    return [weighted_graph_from_edge_list(edge_list, weight_dist, s) for s in random_weight_seeds]


def random_weighted_d_regular_graph(d: int, n: int,
                                    graph_seed: int | None = None,
                                    weight_seed: int | None = None) -> nx.Graph:
    """
    Creates a random d-regular graph with edge weights samples according to the normal distribution.

    :param d: Regularity of graph.
    :param n: Number of graph nodes.
    :param graph_seed: Seed for the random graph.
    :param weight_seed: Seed for the edge weights.
    :return: d-regular graph of length n.
    """
    graph = random_regular_graph(d=d, n=n, seed=graph_seed)
    np.random.seed(weight_seed)
    weights = np.random.normal(size=len(graph.edges))
    nx.set_edge_attributes(graph, {(e, v): {"weight": w} for (e, v), w in zip(graph.edges, weights)})
    return graph


def draw_graph(graph: nx.Graph, draw_weights: bool = False):
    """Draw graph with networkx."""
    pos = nx.nx_agraph.graphviz_layout(graph)
    nx.draw_networkx(graph, pos)
    if draw_weights:
        weights = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels={k: round(v, 2) for k, v in weights.items()})
    plt.show()


def extract_colors_from_graph(graph: nx.Graph) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Given a graph, returns and an edge coloring of the graph where every color is a tuple of edge tuples."""
    graph = graph.copy()
    colors = []
    while graph.edges:
        color = maximal_matching(graph)
        graph.remove_edges_from(color)
        colors.append(tuple(color))
    return tuple(colors)


def assign_weights_to_graph(graph: nx.Graph, weight_set: Sequence | np.ndarray) -> nx.Graph:
    """Returns a new graph where weights are assigned to numerically sorted edges.
     E.g. for a graph with 2 edges: (0, 1): w_0, (0, 2): w_1, (1, 2): w_2 """
    sorted_edges = sorted([sorted(e) for e in graph.edges])
    assert len(weight_set) == len(sorted_edges), (f"Length of weights ({len(weight_set)})"
                                                  f" must be equal to the number of edges ({len(sorted_edges)})")
    weight_dict = {tuple(sorted(tuple(e))): {"weight": w} for e, w in zip(sorted_edges, weight_set)}
    weighted_graph = nx.Graph()
    weighted_graph.graph = graph.graph
    weighted_graph.add_edges_from(sorted_edges)
    nx.set_edge_attributes(weighted_graph, weight_dict)
    return weighted_graph


def graph_to_adjacency_array(graph: nx.Graph) -> np.ndarray:
    """Returns the graph's adjacency matrix as a np.ndarray.
    Graph is required to have edge weights specified by the edge attribute 'weight'."""
    num_nodes = len(graph.nodes)
    matrix = np.zeros((num_nodes, num_nodes))
    for edge in graph.edges(data=True):
        u, v, data = edge
        matrix[u, v] = data["weight"] if 'weight' in data else 1
        matrix[v, u] = data["weight"] if 'weight' in data else 1
    return matrix


def construct_extended_graph(graph: nx.Graph, swap_layers: Sequence[Sequence[tuple[int, ...]]]) -> nx.Graph:
    """Given a graph, constructs a new graph where graph's edge connections are extended
     by provided swaps."""
    hard2log = {k: k for k in range(len(graph.nodes))}
    extended_graph = graph.copy()
    hardware_graph = graph.copy()
    for swap_layer in swap_layers:
        partial_mapping = {}
        for swap in swap_layer:
            q1, q2 = swap
            partial_mapping[hard2log[q1]] = hard2log[q2]
            partial_mapping[hard2log[q2]] = hard2log[q1]

            log1, log2 = hard2log[q1], hard2log[q2]
            hard2log[q1] = log2
            hard2log[q2] = log1

        hardware_graph = nx.relabel_nodes(hardware_graph, partial_mapping,
                                          copy=True)  # note: cyclic dependency, must copy
        for edge in hardware_graph.edges:
            q1, q2 = edge  # edge is virtual
            if (q1, q2) and (q2, q1) not in graph.edges:
                extended_graph.add_edge(*edge)
    return extended_graph


def serialize_graph(graph: nx.Graph, name: str, target_dir: str) -> None:
    target_file = os.path.join(target_dir, name)
    j_graph = json_graph.node_link_data(graph)
    with open(target_file + '.json', 'w') as f:
        json.dump(j_graph, f, indent=2)
