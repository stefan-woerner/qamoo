from collections.abc import Sequence
from copy import deepcopy

import networkx as nx
import numpy as np
from qiskit.circuit import ParameterVector, Parameter, Measure
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate, RZZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import Commuting2qGateRouter
from qiskit.transpiler.passes.routing import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import FindCommutingPauliEvolutions
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import Commuting2qBlock
from qiskit_optimization.applications import Maxcut

from qamoo.utils.graphs import extract_colors_from_graph, construct_extended_graph


def decompose_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Decomposes the PauliEvolutionGates and exp(*) gates."""
    circuit = circuit.decompose(PauliEvolutionGate)
    circuit = circuit.decompose("exp*")
    return circuit


def transpile_qaoa_circuit(qaoa_ansatz: QuantumCircuit, p: int,
                           swap_strategy: SwapStrategy) -> QuantumCircuit:
    """Transpilation steps are done on DAG level and achieve the following:

    1. Running the FindCommutingPauliEvolutions Pass which marks all Rzz gates as one commuting block
    2. Going through the PauliEvolutionGates in the commuting block to give them a unique name based on the virtual qubits they act on
    3. Running the Commuting2qGateRouter Pass to add SWAP gates
    4. Going through the (renamed) PauliEvolutionGates and substituting them with parametrized Rzz gates
    5. For p>1, the QAOA blocks are attached by reversing the operations of the previous QAOA block
    6. Transforming the DAG to a circuit and decomposing the circuit

    """
    decomposed_ansatz = qaoa_ansatz.decompose()  # need PauliEvolutionGates from QAOA block
    dag = circuit_to_dag(decomposed_ansatz)
    dag = FindCommutingPauliEvolutions().run(dag)

    for node in dag.topological_op_nodes():
        if isinstance(node.op, Commuting2qBlock):
            for node in node.op.node_block:
                weight_id = f"_{node.qargs[0]._index}_{node.qargs[1]._index}"
                node.op.name += f"_{weight_id}"

    dag = Commuting2qGateRouter(swap_strategy).run(dag)

    gamma_vector = ParameterVector("γ", p)
    for node in dag.op_nodes(PauliEvolutionGate):
        if not node.op.name == "PauliEvolution":  # exclude Rx gates
            weight_id = node.op.name.split("PauliEvolution_", 1)[1]  # TODO: Why are there 2 underscores?
            dag.substitute_node(node, op=RZZGate(gamma_vector[0] * Parameter(weight_id)))

    if p == 1:
        return decompose_circuit(dag_to_circuit(dag))

    beta_vector = ParameterVector("β", p)

    measurement_nodes = dag.op_nodes(Measure)
    dag.remove_all_ops_named("measure")
    dag.remove_all_ops_named("barrier")
    dag_circuit = deepcopy(dag)
    dag.remove_all_ops_named("h")
    dag.remove_all_ops_named("PauliEvolution")  # remove Rx gates
    reversed_dag = dag.reverse_ops()
    for block in range(1, p):
        if block % 2 == 1:  # even p
            current_dag = reversed_dag
        else:  # odd p
            current_dag = dag

        for node in current_dag.op_nodes(RZZGate):
            gate_params = [p for p in node.op.params[0].parameters if "γ" not in p.name]
            current_dag.substitute_node(node, op=RZZGate(gamma_vector[block] * gate_params[0]))

        dag_circuit.compose(current_dag)

        rx_layer = QuantumCircuit(qaoa_ansatz.num_qubits)
        rx_layer.rx(2 * beta_vector[block], range(rx_layer.num_qubits))
        rx_layer_dag = circuit_to_dag(rx_layer)
        dag_circuit.compose(rx_layer_dag)

    if p % 2 == 1:  # odd p, permute measurements
        for node in measurement_nodes:
            dag_circuit.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
        circuit = dag_to_circuit(dag_circuit)
    else:  # even p, measurements have trivial layout
        circuit = dag_to_circuit(dag_circuit)
        circuit.barrier()
        for q in range(circuit.num_qubits):
            circuit.measure(q, q)
    return decompose_circuit(circuit)


def check_swap_layers(swap_layers: tuple[tuple[tuple[int, ...], ...], ...],
                      edges: Sequence) -> list[list[tuple[int, ...]]]:
    """To accommodate for exact tuple comparisons, this method returns swap layers where each swap
    is guaranteed to be in edges."""
    checked_swap_layers = []
    for layer in swap_layers:
        checked_swaps = []
        for swap in layer:
            swap = tuple(swap)
            if swap not in edges:
                checked_swaps.append(swap[::-1])
            else:
                checked_swaps.append(swap)
        checked_swap_layers.append(checked_swaps)
    return checked_swap_layers


def build_parameterized_qaoa_circuits(p_values: list[int],
                                      ising: SparsePauliOp,
                                      swap_strategy: SwapStrategy | None = None) -> dict[int, QuantumCircuit]:
    """Returns a dictionary of QAOA circuits where each edge in the graph is mapped to a Rzz gate
    that encodes the information about the virtual qubits it acts on as an additional parameter.
    The circuit can be used as a template to bind the coefficients of different sets of Hamiltonians,
    given that all Hamiltonians correspond to the same graph topology."""
    qaoa_block = construct_qaoa_circuit_from_ising(ising, reps=1)
    if swap_strategy is None:
        edges = []
        for p in ising.paulis:
            indices = list(idx for idx, v in enumerate(p.z) if v)
            if len(indices) == 2:
                edges.append(tuple(indices))
                edges.append(tuple(indices[::-1]))
        cp = CouplingMap(edges)
        cp.make_symmetric()
        swap_strategy = SwapStrategy(cp, swap_layers=[])
    parameterized_qaoa_circuits = {p: transpile_qaoa_circuit(qaoa_block, p, swap_strategy) for p in p_values}
    return parameterized_qaoa_circuits


def get_swap_layers(edges: Sequence, num_swap_layers: int) -> list[list[tuple[int, ...]]]:
    """Returns parallelizable sets of swap layers from edges."""
    graph = nx.Graph()
    graph.add_edges_from(edges)

    swap_layers = extract_colors_from_graph(graph)
    swap_layers = swap_layers[:num_swap_layers]
    swap_layers = check_swap_layers(swap_layers, edges)  # SwapStrategy checks for exact tuple comparison in swap layers
    return swap_layers


def construct_qaoa_circuit_from_ising(ising: SparsePauliOp, reps: int) -> QuantumCircuit:
    """Given a graph, builds the QAOA circuit corresponding to the Maxcut Hamiltonian."""
    ansatz = QAOAAnsatz(cost_operator=ising, reps=reps)
    ansatz.measure_all()
    return ansatz


def extended_problem_from_coupling(coupling_map: CouplingMap,
                                   reps: list[int], num_swap_layers: int) -> tuple[dict[int, QuantumCircuit], nx.Graph]:
    """Extends a native problem graph compliant with the given coupling map by swap layers."""
    swap_layers = get_swap_layers(coupling_map.get_edges(), num_swap_layers)
    swap_strategy = SwapStrategy(coupling_map, swap_layers)

    hardware_graph = nx.Graph()
    hardware_graph.add_edges_from(coupling_map.get_edges())

    extended_graph = construct_extended_graph(hardware_graph, swap_layers)
    parameterized_qaoa_circuits = build_parameterized_qaoa_circuits(reps, extended_graph, swap_strategy)
    return parameterized_qaoa_circuits, extended_graph


def validate_qcs_and_params(qcs: list[QuantumCircuit], params: list[list[float]]):
    """Validates that parameters dimensions and quantum circuits dimensions match."""
    assert (isinstance(qcs, Sequence)
            and isinstance(params, Sequence)), "Quantum circuits and parameters must be of type Sequence."
    assert len(qcs) == len(params), "The number of quantum circuits must be equal to the number of parameter sequences."
    assert all([len(qc.parameters) == len(p) for qc, p in
                zip(qcs, params)]), "The number of parameters in a circuit must be equal to the number of parameters."


def bind_qc_by_param_sets(qc: QuantumCircuit,
                          parameter_index_dict: dict[Parameter, list[int]],  # TODO: Change to tuple[int, int]
                          parameter_vectors: np.ndarray = None,
                          index_weight_dicts: [dict[tuple[int, int], float]] = None) -> list[QuantumCircuit]:
    """Given a list of parameter vectors, returns the list of quantum circuits with bound parameters."""

    if index_weight_dicts is not None:
        parameter_weight_dicts = []
        for index_weight_dict in index_weight_dicts:
            parameter_weight_dict = {}
            for p_k in parameter_index_dict:
                indices = tuple(sorted(parameter_index_dict[p_k]))  # list of ints
                parameter_weight_dict[p_k] = index_weight_dict[indices]
            parameter_weight_dicts.append(parameter_weight_dict)
        qcs = [qc.assign_parameters(p_w_d) for p_w_d in parameter_weight_dicts]
        return qcs

    parameter_keys = list(parameter_index_dict.keys())
    assert len(parameter_keys) == parameter_vectors.shape[1]
    qcs = [qc.assign_parameters({k: p for k, p in zip(parameter_keys, params)}) for params in parameter_vectors]
    return qcs


def parameterized_circuit_from_connectivity(connectivity: nx.Graph | SparsePauliOp,
                                            coupling_map: CouplingMap, p: int,
                                            num_swap_layers: int) -> QuantumCircuit:
    if isinstance(connectivity, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_edges_from(connectivity.edges)
        empty_ising, _ = Maxcut(empty_graph).to_quadratic_program().to_ising()
    elif isinstance(connectivity, SparsePauliOp):
        empty_ising = SparsePauliOp(data=PauliList(connectivity.paulis))
    else:
        TypeError(f'Connectivity must be of type nx.Graph or SparsePauliOp. Got {type(connectivity)} instead.')

    swap_layers = get_swap_layers(coupling_map.get_edges(), num_swap_layers)
    swap_strategy = SwapStrategy(coupling_map, swap_layers)

    parameterized_circuit = build_parameterized_qaoa_circuits([p], empty_ising, swap_strategy)[p]

    return parameterized_circuit


def get_parameter_dict_from_circuit(circuit: QuantumCircuit) -> dict[Parameter, list[int]]:
    """Returns the parameter dictionary of a circuit in form {Parameter: [q0, q1]},
    where q0 and q1 are the virtual qubits the parametrized gates act on. The dictionary is sorted in increasing qubit
    indices.
    Requires the circuit to have parametrized gates with unique name identifiers of the form '*qi_qj'."""
    parameter_dict = {}  # {Parameter: [int, int]}
    for param in circuit.parameters:
        if not (param.name.startswith("β") or param.name.startswith("γ")):
            name = param.name[1:]  # removing the first underscore
            qubits = name.split("_", 1)
            q0, q1 = int(qubits[0]), int(qubits[1])
            parameter_dict[param] = [q0, q1]
    parameter_dict = {k: v for k, v in sorted(parameter_dict.items(), key=lambda item: sorted(item[1]))}
    return parameter_dict


def get_coeff_dict_from_ising(ising: SparsePauliOp) -> dict[tuple[int, int], float]:
    coeff_dict = {}
    for p, coeff in zip(ising.paulis, ising.coeffs):
        q_term = tuple(sorted(idx for idx, v in enumerate(p.z) if v))
        if len(q_term) == 2:
            coeff_dict[q_term] = 2 * coeff
    return coeff_dict
