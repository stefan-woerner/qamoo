from lp_generation import generate_lp_file_dcm
import os

def compute_dcm(num_qubits, num_swap_layers, num_objectives, version, factor_integer):
    working_directory = os.getcwd()
    dcm = f"classical_algorithms_moip/dcm/DCM"
    problem_file = f"{working_directory}/../data/problems/{num_qubits}q/problem_set_{num_qubits}q_{num_swap_layers}s_{num_objectives}o_{version}/problem_{num_qubits}_{num_swap_layers}_{num_objectives}_{version}_dcm.lp"
    output = f"{working_directory}/../data/problems/{num_qubits}q/problem_set_{num_qubits}q_{num_swap_layers}s_{num_objectives}o_{version}/results/dcm_{factor_integer}.sol"
    os.system(f"{dcm} {problem_file} {num_objectives} {output}")


if __name__ == "__main__":

    # timeout = 20000 can be changed in dcm/Headers/extern_parameters.h
    factor_integer = 100
    num_objectives = 3
    num_qubits = 27
    num_swap_layers = 0
    version = 0
                        
    generate_lp_file_dcm(num_qubits, num_swap_layers, num_objectives, version, factor_integer)
    compute_dcm(num_qubits, num_swap_layers, num_objectives, version, factor_integer)



