from typing import TextIO
import numpy as np
import os
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2, FakeSherbrooke
from qamoo.utils.data_structures import ProblemGraphBuilder


def generate_lp_file_dpa(num_qubits, num_swap_layers, num_objectives, version, factor_integer=None) -> TextIO:
    if num_qubits == 27:
        backend = FakeMumbaiV2()
    elif num_qubits == 127:
        backend = FakeSherbrooke()
    else:
        ValueError()

    # Specify problems
    working_directory = os.getcwd()
    problem_path = f"{working_directory}/../data/problems/{num_qubits}q/problem_set_{num_qubits}q_{num_swap_layers}s_{num_objectives}o_{version}"
    pg_builder = ProblemGraphBuilder(num_qubits)
    problem_graphs = list()
    for obj in range(num_objectives):
        problem_graph = pg_builder.deserialize(f"{problem_path}/problem_graph_{obj}.json")
        problem_graphs.append(problem_graph)
    graph_edges = list(problem_graphs[0].edges())
    graph_nodes = problem_graphs[0].nodes()

    # Start building the LP file
    lp_file = open(f"{problem_path}/problem_{num_qubits}_{num_swap_layers}_{num_objectives}_{version}_dpa.lp", "w")
    lp_file.write("Maximize\n")
    lp_file.write(" obj:\n")
    lp_file.write("Subject To\n")

    constr_names = []
    constr_strings = ["L1", "L2", "G1", "G2"]
    edge_vars = []
    for edge in graph_edges:
            u, v = edge
            edge_var = f"e_{u}_{v}"
            edge_vars.append(edge_var)
    for str in constr_strings:
        for edge in graph_edges:
            u, v = edge
            edge_var = f"e_{u}_{v}"
            node1_var = f"n_{u}"
            node2_var = f"n_{v}"
            constr_name = f"{str}_{u}_{v}"
            constr_names.append(constr_name)
            rhs = 2 if constr_name.startswith("L1") else 0
            direction = "<=" if constr_name.startswith("L") else ">="
            sign1 = "+" if (constr_name.startswith("L1") or constr_name.startswith("G1")) else "-"
            sign2 = "+" if (constr_name.startswith("L1") or constr_name.startswith("G2")) else "-"
            lp_file.write(f" {constr_name}:   {edge_var} {sign1} {node1_var} {sign2} {node2_var} {direction} {rhs}\n")


    for idx in range(1, num_objectives + 1):
        lp_file.write(f" OBJ{idx}:    ")
        pg = problem_graphs[idx-1]
        for edge in graph_edges:
            u, v = edge
            edge_var = f"e_{u}_{v}"

            w = pg[u][v]["weight"]

            if factor_integer is not None:
                # rounding:
                w = int(np.round(w * factor_integer))
            
            if w >=0:
                lp_file.write(f" +{w} {edge_var}")
            else:
                lp_file.write(f" {w} {edge_var}")
        lp_file.write(f" > {idx}\n")

    

    lp_file.write("Bounds\n")
    for edge_var in edge_vars:
        lp_file.write(f" 0 <= {edge_var} <= 1\n")
    for n in graph_nodes:
        node_var = f"n_{n}"
        lp_file.write(f" 0 <= {node_var} <= 1\n")

    lp_file.write("Binaries\n")
    for edge_var in edge_vars:
        lp_file.write(f" {edge_var} ")
    for n in graph_nodes:
        node_var = f"n_{n}"
        lp_file.write(f" {node_var} ")
    lp_file.write(f"\n")
    lp_file.write("End")

    lp_file.close()
    return lp_file

def generate_lp_file_dcm(num_qubits, num_swap_layers, num_objectives, version, factor_integer) -> TextIO:
    if num_qubits == 27:
        backend = FakeMumbaiV2()
    elif num_qubits == 127:
        backend = FakeSherbrooke()
    else:
        ValueError()

    # Specify problems
    working_directory = os.getcwd()
    problem_path = f"{working_directory}/../data/problems/{num_qubits}q/problem_set_{num_qubits}q_{num_swap_layers}s_{num_objectives}o_{version}"
    pg_builder = ProblemGraphBuilder(num_qubits)
    problem_graphs = list()
    for obj in range(num_objectives):
        problem_graph = pg_builder.deserialize(f"{problem_path}/problem_graph_{obj}.json")
        problem_graphs.append(problem_graph)
    graph_edges = list(problem_graphs[0].edges())
    graph_nodes = problem_graphs[0].nodes()

    # Start building the LP file
    lp_file = open(f"{problem_path}/problem_{num_qubits}_{num_swap_layers}_{num_objectives}_{version}_dcm.lp", "w")
    lp_file.write("Minimize\n")
    lp_file.write(" obj: ")
    for idx in range(1, num_objectives + 1):
        lp_file.write(f"- x{idx} ")
    lp_file.write("+ xy ")
    lp_file.write("\n")
    lp_file.write("Subject To\n")

    for idx in range(1, num_objectives + 1):
        lp_file.write(f" c{idx}:  x{idx}  = 0\n")

    for idx in range(1, num_objectives + 1):
        lp_file.write(f" c{idx + num_objectives}:  x{idx} ")
        pg = problem_graphs[idx-1]
        for edge in graph_edges:
            u, v = edge
            edge_var = f"e_{u}_{v}"

            w = pg[u][v]["weight"]

            if factor_integer is not None:
                # rounding:
                w = int(np.round(w * factor_integer))   #(-1) * 
            
            if w >=0:
                lp_file.write(f" +{w} {edge_var}")
            else:
                lp_file.write(f" {w} {edge_var}")
        lp_file.write(f" = 0\n")

    constr_names = []
    constr_strings = ["L1", "L2", "G1", "G2"]
    edge_vars = []
    for edge in graph_edges:
            u, v = edge
            edge_var = f"e_{u}_{v}"
            edge_vars.append(edge_var)
    for str in constr_strings:
        for edge in graph_edges:
            u, v = edge
            edge_var = f"e_{u}_{v}"
            node1_var = f"n_{u}"
            node2_var = f"n_{v}"
            constr_name = f"{str}_{u}_{v}"
            constr_names.append(constr_name)
            rhs = 2 if constr_name.startswith("L1") else 0
            direction = "<=" if constr_name.startswith("L") else ">="
            sign1 = "+" if (constr_name.startswith("L1") or constr_name.startswith("G1")) else "-"
            sign2 = "+" if (constr_name.startswith("L1") or constr_name.startswith("G2")) else "-"
            lp_file.write(f" {constr_name}:   {edge_var} {sign1} {node1_var} {sign2} {node2_var} {direction} {rhs}\n")

    

    lp_file.write("Bounds\n")
    for idx in range(1, num_objectives + 1):
        lp_file.write(f"       x{idx} Free\n")
    for edge_var in edge_vars:
        lp_file.write(f" 0 <= {edge_var} <= 1\n")
    for n in graph_nodes:
        node_var = f"n_{n}"
        lp_file.write(f" 0 <= {node_var} <= 1\n")
    lp_file.write("     xy = 0\n")

    lp_file.write("Binaries\n")
    for edge_var in edge_vars:
        lp_file.write(f" {edge_var} ")
    for n in graph_nodes:
        node_var = f"n_{n}"
        lp_file.write(f" {node_var} ")
    lp_file.write(f"\n")
    lp_file.write("End")

    lp_file.close()
    return lp_file


if __name__ == "__main__":

    generate_lp_file_dpa(42, 0, 3, 0, 100)
    generate_lp_file_dcm(42, 0, 3, 0, 1000)
