import json
import os
from collections.abc import Sequence

import networkx as nx
from networkx.readwrite import json_graph

from qamoo.utils.graphs import construct_extended_graph


class ProblemGraphBuilder:

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.graph = None

    def build(self, edge_list: Sequence[Sequence[int]]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_edges_from(edge_list)
        self.graph = graph
        assert self.num_nodes == len(graph.nodes), (f'Number of nodes {self.num_nodes} '
                                                    f'does not match the number of nodes'
                                                    f' in given edge set {len(graph.nodes)}.')

    def assign_weights(self, weights: Sequence[float]) -> None:
        sorted_edges = sorted([sorted(e) for e in self.graph.edges])
        assert len(sorted_edges) == len(weights), (f'Length of edges ({len(sorted_edges)})'
                                                   f' in the graph and weights ({len(weights)}) do not match')

        weight_dict = {tuple(e): {"weight": w} for e, w in zip(sorted_edges, weights)}
        nx.set_edge_attributes(self.graph, weight_dict)  # inplace

    def extend_by_swap_layers(self, swap_layers: Sequence[Sequence[tuple[int, ...]]]):
        self.graph = construct_extended_graph(self.graph, swap_layers)  # replaces graph

    def serialize(self, name: str, target_dir: str) -> None:
        target_file = os.path.join(target_dir, name)
        j_graph = json_graph.node_link_data(self.graph)
        with open(target_file + '.json', 'w') as f:
            json.dump(j_graph, f, indent=2)

    @staticmethod
    def deserialize(filename: str) -> nx.Graph:
        with open(filename) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

