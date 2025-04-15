from lp_generation import generate_lp_file_dpa
import os


def compute_dpa(num_qubits, num_swap_layers, num_objectives, version, timeout, factor_integer):
    working_directory = os.getcwd()
    dpa = f"classical_algorithms_moip/dpa/main"
    problem_file = f"{working_directory}/../data/problems/{num_qubits}q/problem_set_{num_qubits}q_{num_swap_layers}s_{num_objectives}o_{version}/problem_{num_qubits}_{num_swap_layers}_{num_objectives}_{version}_dpa.lp"
    flag = f"-a"    # augmented epsilon constraint method
    output = f"{working_directory}/../data/problems/{num_qubits}q/problem_set_{num_qubits}q_{num_swap_layers}s_{num_objectives}o_{version}/results/dpa_{factor_integer}"
    os.system(f"{dpa} {problem_file} {flag} {output} {timeout}")


if __name__ == "__main__":

    timeout = 20000
    factor_integer = 100
    num_objectives = 3
    num_qubits = 27
    num_swap_layers = 0
    version = 0

    generate_lp_file_dpa(num_qubits, num_swap_layers, num_objectives, version, factor_integer)
    compute_dpa(num_qubits, num_swap_layers, num_objectives, version, timeout, factor_integer)

