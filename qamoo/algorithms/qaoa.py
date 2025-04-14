import os
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time

from qamoo.configs.configs import QAOAConfig
from qamoo.utils.data_structures import ProblemGraphBuilder

from qamoo.utils.graphs import linearize_graphs
from qamoo.utils.transpilation import (get_coeff_dict_from_ising,
                                      parameterized_circuit_from_connectivity,
                                      get_parameter_dict_from_circuit)

from qiskit import qpy, QuantumCircuit
from qiskit.circuit import Parameter, CircuitInstruction
from qiskit.circuit.library import RZZGate, RXGate, HGate
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendSamplerV2
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit_ibm_runtime import SamplerV2, Batch, Session
from qiskit_ibm_runtime.transpiler.passes.basis.fold_rzz_angle import convert_to_rzz_valid_pub
 
from qiskit_optimization.applications import Maxcut


def prepare_qaoa_circuits(config: QAOAConfig, backend, overwrite_results=False):
    # get problem specification
    problem = config.problem
    problem_folder = problem.problem_folder
    num_objectives = problem.num_objectives
    num_swaps = problem.num_swap_layers

    # create results directory and raise exception if it already exists
    try:
        os.makedirs(config.results_folder)
    except FileExistsError:
        if overwrite_results:
            print('Folder exists!')
        else:
            raise FileExistsError()

    p = config.p

    objective_graphs = [
        ProblemGraphBuilder.deserialize(
            problem_folder + f'problem_graph_{idx}.json')
        for idx in range(num_objectives)
    ]

    objective_weights = config.objective_weights
    linearized_graphs = [linearize_graphs(objective_graphs, l) for l in objective_weights]
    isings = (Maxcut(g).to_quadratic_program().to_ising()[0] for g in linearized_graphs)
    coeff_dicts = [get_coeff_dict_from_ising(i) for i in isings]  # dict[tuple[int, int], float]

    if num_swaps > 0:
        coupling_map_edges = objective_graphs[0].graph['coupling_map_edges']
    else:
        #coupling_map_edges = linearized_graphs[0].edges
        coupling_map_edges = objective_graphs[0].edges
    coupling_map = CouplingMap(coupling_map_edges)
    coupling_map.make_symmetric()

    parameterized_circuit = parameterized_circuit_from_connectivity(linearized_graphs[0], coupling_map, p, num_swaps)

    # save parameter dicts
    pd = get_parameter_dict_from_circuit(parameterized_circuit)  # dict[Parameter, list[int]]
    parameter_dicts = []
    for coeff_dict in coeff_dicts:
        assert len(pd) == len(coeff_dict)
        parameter_dict = {param.name: coeff_dict[tuple(sorted(value))] for param, value in
                          pd.items()}  # dict[Parameter, float]
        parameter_dicts.append(parameter_dict)

    # resolve parameter expressions
    # get QAOA parameters (identical accross samples)
    betas = config.optimal_parameters[:config.p]
    gammas = config.optimal_parameters[config.p:]

    # create new parameters for beta
    beta_params = [Parameter(fr'$\beta_{i}$') for i in range(config.p)]

    # initialize list of parameter dicts
    mapped_param_values = []
    for i in range(config.num_samples):
        mapped_param_values += [{}]

    # copy input circuit
    mapped_qc = parameterized_circuit.copy()
    mapped_qc.clear()

    # process circuit
    i_rzz = 0
    for d in parameterized_circuit.data:

        if d.operation.name == 'h':
            mapped_qc.data += [CircuitInstruction(HGate(), d.qubits, d.clbits)]
            
        elif d.operation.name == 'rzz':
            params = list(d.operation.params[0].parameters)
            i_problem = params[1].name  # get Hamiltonian parameter index
            i_gamma = int(params[0].name[2:-1])  # get gamma index 

            # generate new parameter to replace expression
            theta = Parameter(fr'$\theta_{{{i_rzz}}}$')
            i_rzz += 1

            for i in range(config.num_samples):
                mapped_param_values[i][theta.name] = np.real(gammas[i_gamma] * parameter_dicts[i][i_problem])
            mapped_qc.data += [CircuitInstruction(RZZGate(theta), d.qubits, d.clbits)]
        
        elif d.operation.name == 'rx':
            params = list(d.operation.params[0].parameters)
            i_beta = int(params[0].name[2:-1])  # get beta index
            for i in range(config.num_samples):
                mapped_param_values[i][beta_params[i_beta].name] = np.real(2 * betas[i_beta])
            mapped_qc.data += [CircuitInstruction(RXGate(beta_params[i_beta]), d.qubits, d.clbits)]

        elif d.operation.name == 'measure':
            mapped_qc.data += [d]

        elif d.operation.name == 'barrier':
            continue

        else:
            raise Exception(f'Unhandled operation ({d.operation.name})!')

    # save mapped parameterized circuit
    with open(f'{config.results_folder}parameterized_circuit.qpy', 'wb') as f:
        qpy.dump(mapped_qc, f)
        # pickle.dump(parameterized_circuit, f)

    # save mapped parameter values
    with open(config.results_folder + 'parameter_dicts.json', 'w') as f:
        json.dump(mapped_param_values, f)

def transpile_qaoa_circuits_parametrized(config: QAOAConfig, backend):

    # load parameterized circuit + parameter dicts
    with open(config.results_folder + 'parameterized_circuit.qpy', 'rb') as f:
        parameterized_circuit = qpy.load(f)[0]

    if backend.name != 'aer_simulator_matrix_product_state':
        preset_manager = generate_preset_pass_manager(backend=backend,
                                                    optimization_level=3,
                                                    scheduling_method="alap",  # TODO: should we use this?
                                                    initial_layout=config.initial_layout)
        # transpile
        transpiled_qc = preset_manager.run(parameterized_circuit)
    else:
        # use circuit as is
        transpiled_qc = parameterized_circuit

    # save transpiled parametrized circuit and parameters
    with open(config.results_folder + 'transpiled_parametrized_circuit.qpy', 'wb') as f:
        qpy.dump(transpiled_qc, f)
    with open(config.results_folder + 'initial_layout.json', 'w') as f:
        json.dump(config.initial_layout, f)

def batch_execute_qaoa_circuits_parametrized(configs, backend):
    # load circuits for all configs
    circuits = []
    parameter_dicts = []
    for config in configs:
        with open(config.results_folder + 'transpiled_parametrized_circuit.qpy', 'rb') as f:
            circuits += [qpy.load(f)[0]]
        with open(config.results_folder + 'parameter_dicts.json', 'r') as f:
            parameter_dicts += [json.load(f)]

    if backend.name != 'aer_simulator_matrix_product_state':

        with Batch(backend=backend) as batch:

            sampler = SamplerV2(batch)

            # use settings from first config
            sampler.options.dynamical_decoupling.enable = configs[0].dynamic_decoupling
            if configs[0].dynamic_decoupling:
                sampler.options.dynamical_decoupling.sequence_type = 'XY4'
            sampler.options.twirling.enable_gates = configs[0].twirling
            sampler.options.twirling.enable_measure = configs[0].twirling
            
            # turn on new compiler to speed 
            if not backend.options.use_fractional_gates:
                print('use gen3-turbo!')
                sampler.options.experimental = {"execution_path" : "gen3-turbo"}
            
            jobs = []
            max_params_sets = 200  # assuming we never take more than 5k shots per circuit (limit is 5M total shots)
            for i in range(len(configs)):

                # set rep_delay if given and use default otherwise
                if configs[i].rep_delay is not None:
                    sampler.options.execution.rep_delay = configs[i].rep_delay
                else:
                    sampler.options.execution.rep_delay = backend.configuration().default_rep_delay
                print(f'set rep_delay to {sampler.options.execution.rep_delay}!')

                param_names = [p.name for p in circuits[i].parameters]
                param_values = [[d[pn] for pn in param_names] for d in parameter_dicts[i]]

                num_param_sets = len(parameter_dicts[i])
                num_jobs = int(np.ceil(num_param_sets / max_params_sets))
                jobs_i = []
                jobs += [jobs_i]
                for j in range(num_jobs):
                    start = j*max_params_sets
                    end = min([num_param_sets, (j+1)*max_params_sets])
                    
                    # update pub to satisfy RZZ angle constraints
                    pub = [(circuits[i], param_values[start:end], configs[i].shots)]
                    if backend.options.use_fractional_gates:
                        print('convert to valid rzz angles for using fractional gates.')
                        pub = convert_to_rzz_valid_pub('sampler', pub)

                    job = sampler.run(pub)
                    jobs_i += [job]

                with open(configs[i].results_folder + 'job_id.json', 'w') as f:
                    json.dump([job.job_id() for job in jobs_i], f)
                if m3:
                    with open(configs[i].results_folder + 'm3_job_id.json', 'w') as f:
                        json.dump([job.job_id() for job in mit_jobs], f)
    else:
        print('running with simulator')
        sampler = BackendSamplerV2(backend=backend)

        t_start = time.time()

        jobs = []
        max_params_sets = 200
        for i in range(len(configs)):

            param_names = [p.name for p in circuits[i].parameters]
            param_values = [[d[pn] for pn in param_names] for d in parameter_dicts[i]]

            num_param_sets = len(parameter_dicts[i])
            num_jobs = int(np.ceil(num_param_sets / max_params_sets))
            jobs_i = []
            jobs += [jobs_i]
            results = []
            for j in tqdm(range(num_jobs)):
                start = j*max_params_sets
                end = min([num_param_sets, (j+1)*max_params_sets])

                # qc = remove_idle_qwires(circuits[i])
                
                # update pub to satisfy RZZ angle constraints
                pub = (circuits[i], param_values[start:end], configs[i].shots)
                if backend.options.use_fractional_gates:
                    print('convert to valid rzz angles for using fractional gates.')
                    pub = convert_to_rzz_valid_pub('sampler', pub)

                job = sampler.run([pub])
                result = job.result()
                results += [result]

            samples = []
            for r in tqdm(results):
                bitstrings = r[0].data.meas.get_bitstrings()
                samples += [[int(x) for x in reversed(b)] for b in bitstrings]

            samples = np.array(samples, dtype='b')
            np.save(configs[i].results_folder + 'samples.npy', samples)
        runtime = time.time() - t_start    
        print('simulation time:', runtime)

def get_initial_layout(config: QAOAConfig, backend):

    # determine initial layout
    # select best layout for 42-qubit circuits
    problem = config.problem
    if (problem.num_qubits == 42 or problem.num_qubits == 96) and backend.name != 'aer_simulator_matrix_product_state':

        # load parametrized circuit
        circuit_path = config.results_folder + 'parameterized_circuit.qpy'
        with open(circuit_path, 'rb') as f:
            # qc = pickle.load(f)
            qc = qpy.load(f)[0]
        
        # load potential initial layouts
        if backend.num_qubits == 127:
            with open(problem.data_folder + 'initial_layouts/42q_eagle.json', 'r') as f:
                initial_layouts = json.load(f)
        elif backend.num_qubits == 133:
            with open(problem.data_folder + 'initial_layouts/42q_heron_133.json', 'r') as f:
                initial_layouts = json.load(f)
        elif backend.num_qubits == 156:
            with open(problem.data_folder + f'initial_layouts/{problem.num_qubits}q_heron_156.json', 'r') as f:
                initial_layouts = json.load(f)

        # transpile different layouts to backend
        qcs = []
        for initial_layout in tqdm(initial_layouts):
            preset_manager = generate_preset_pass_manager(backend=backend,
                                                        optimization_level=3,
                                                        initial_layout=initial_layout)
            qc_transpiled = preset_manager.run(qc)
            qcs += [qc_transpiled]
        
        # score results 
        fidelities = []
        measurement_fidelities = []
        all_avg_fidelities = []
        for i in range(len(initial_layouts)):
            initial_layout = initial_layouts[i]
            fidelity = 1
            measurement_fidelity = 1
            avg_fidelities = {}
            avg_counter = {}
            for d in qcs[i].data:
                if d.operation.name == 'measure':
                    error = backend.properties().readout_error(d.qubits[0]._index)
                    fidelity *= (1 - error)
                    measurement_fidelity *= (1 - error)
                    avg_fidelities['measure'] = avg_fidelities.get('measure', 1) * (1 - error)
                    avg_counter['measure'] = avg_counter.get('measure', 0) + 1
                elif d.operation.name != 'barrier':
                    name = d.operation.name
                    qubits = [q._index for q in d.qubits]
                    if name == 'rzz':
                        name = 'cz'
                    error = backend.properties().gate_error(name, qubits)
                    fidelity *= (1 - error)
                    avg_fidelities[name] = avg_fidelities.get(name, 1) * (1 - error)
                    avg_counter[name] = avg_counter.get(name, 0) + 1
                
            fidelities += [fidelity]
            measurement_fidelities += [measurement_fidelity]
            all_avg_fidelities += [{}]
            for k, v in avg_fidelities.items():
                all_avg_fidelities[-1][k] = np.round(v ** ( 1 / avg_counter[k] ), decimals=4)

        print('total fidelities:       ', np.round(fidelities, decimals=3))
        print('measurement fidelities: ', np.round(measurement_fidelities, decimals=3))

        i_opt = np.argmax(fidelities)
        print('max. fidelity:          ', np.round(fidelities[i_opt], decimals=6))
        config.initial_layout = initial_layouts[i_opt]
        with open(config.results_folder + 'initial_layout.json', 'w') as f:
            json.dump(config.initial_layout, f)
        for i in range(len(all_avg_fidelities)):
            if i != i_opt:
                print(all_avg_fidelities[i])
            else:
                print(all_avg_fidelities[i], '*')
        with open(config.results_folder + 'estimated_fidelity.json', 'w') as f:
            json.dump(fidelities[i_opt], f)

def session_execute_qaoa_circuits_parametrized(configs, backend):
    # load circuits for all configs
    circuits = []
    parameter_dicts = []
    for config in configs:
        # with open(config.results_folder + 'transpiled_parametrized_circuit.qpy', 'rb') as f:
        with open(config.results_folder + 'parameterized_circuit.qpy', 'rb') as f:
            circuits += [qpy.load(f)[0]]
        with open(config.results_folder + 'parameter_dicts.json', 'r') as f:
            parameter_dicts += [json.load(f)]

    if backend.name != 'aer_simulator_matrix_product_state':
        m3 = np.any([config.m3 for config in configs])
        with Session(backend=backend, max_time=6000) as session:

            sampler = SamplerV2(session)
            
            # send dummy job to trigger start of session
            preset_manager_start = generate_preset_pass_manager(backend=backend, optimization_level=0)
            qc_session_start = QuantumCircuit(1)
            qc_session_start.measure_all()
            qc_session_start = preset_manager_start.run(qc_session_start)
            job_session_start = sampler.run([(qc_session_start, [], 1)])
            job_session_start.result()  # this is blocking until the session starts
            print('session started!')

            # use settings from first config
            sampler.options.dynamical_decoupling.enable = configs[0].dynamic_decoupling
            if configs[0].dynamic_decoupling:
                sampler.options.dynamical_decoupling.sequence_type = 'XY4'
            sampler.options.twirling.enable_gates = configs[0].twirling
            sampler.options.twirling.enable_measure = configs[0].twirling
            
            # turn on new compiler to speed 
            if not backend.options.use_fractional_gates:
                print('use gen3-turbo!')
                sampler.options.experimental = {"execution_path" : "gen3-turbo"}

            jobs = []
            max_params_sets = 200  # assuming we never take more than 5k shots per circuit (limit is 5M total shots)
            for i in range(len(configs)):

                # determine initial layout and transpile circuit
                get_initial_layout(config, backend)
                preset_manager = generate_preset_pass_manager(backend=backend,
                                                    optimization_level=3,
                                                    scheduling_method="alap",  # TODO: should we use this?
                                                    initial_layout=config.initial_layout)
                qc = preset_manager.run(circuits[i])

                # set rep_delay if given and use default otherwise
                if configs[i].rep_delay is not None:
                    sampler.options.execution.rep_delay = configs[i].rep_delay
                else:
                    sampler.options.execution.rep_delay = backend.configuration().default_rep_delay
                print(f'set rep_delay to {sampler.options.execution.rep_delay}!')

                param_names = [p.name for p in qc.parameters]
                param_values = [[d[pn] for pn in param_names] for d in parameter_dicts[i]]

                num_param_sets = len(parameter_dicts[i])
                num_jobs = int(np.ceil(num_param_sets / max_params_sets))
                jobs_i = []
                jobs += [jobs_i]
                for j in range(num_jobs):
                    start = j*max_params_sets
                    end = min([num_param_sets, (j+1)*max_params_sets])
                    
                    # update pub to satisfy RZZ angle constraints
                    pub = (qc, param_values[start:end], configs[i].shots)
                    if backend.options.use_fractional_gates:
                        qc.remove_final_measurements()
                        pub = convert_to_rzz_valid_pub('sampler', pub)
                        pub.circuit.measure_all()

                    job = sampler.run([pub])
                    jobs_i += [job]

                print('done')
                with open(configs[i].results_folder + 'job_id.json', 'w') as f:
                    json.dump([job.job_id() for job in jobs_i], f)
    else:
        print('running with simulator')
        sampler = BackendSamplerV2(backend=backend)

        jobs = []
        max_params_sets = 200
        for i in range(len(configs)):

            param_names = [p.name for p in circuits[i].parameters]
            param_values = [[d[pn] for pn in param_names] for d in parameter_dicts[i]]

            num_param_sets = len(parameter_dicts[i])
            num_jobs = int(np.ceil(num_param_sets / max_params_sets))
            jobs_i = []
            jobs += [jobs_i]
            results = []
            for j in tqdm(range(num_jobs)):
                start = j*max_params_sets
                end = min([num_param_sets, (j+1)*max_params_sets])

                # qc = remove_idle_qwires(circuits[i])

                # update pub to satisfy RZZ angle constraints
                pub = (circuits[i], param_values[start:end], configs[i].shots)
                if backend.options.use_fractional_gates:
                    circuits[i].remove_final_measurements()
                    pub = convert_to_rzz_valid_pub('sampler', pub)
                    pub.circuit.measure_all()
                
                job = sampler.run([pub])
                result = job.result()
                results += [result]

            samples = []
            for r in tqdm(results):
                bitstrings = r[0].data.meas.get_bitstrings()
                samples += [[int(x) for x in reversed(b)] for b in bitstrings]

            samples = np.array(samples, dtype='b')
            np.save(configs[i].results_folder + 'samples.npy', samples)


def load_hardware_results(configs, backend):

    for i in range(len(configs)):

        quantum_time = 0
        quantum_metrics = []

        # load config job ids
        with open(configs[i].results_folder + 'job_id.json', 'r') as f:
            job_ids = json.load(f)

        # get jobs from ids
        # the overall job for the config is split into multipl parametrizations
        jobs = []
        for job_id in tqdm(job_ids):
            jobs += [backend.service.job(job_id)]

        results = []
        for job in tqdm(jobs):
            try:
                results += [job.result()]
                quantum_metrics += [job.metrics()]
                quantum_time += quantum_metrics[-1]['usage']['quantum_seconds']
            except:
                print(f'skipping job {job.job_id()}')

        # store quantum time and metrics for each config
        with open(configs[i].results_folder + 'quantum_time.json', 'w') as f:
            json.dump(quantum_time, f)
        with open(configs[i].results_folder + 'quantum_metrics.json', 'w') as f:
            json.dump(quantum_metrics, f)

        samples = []
        for r in tqdm(results):
            bitstrings = r[0].data.meas.get_bitstrings()
            samples += [[int(x) for x in reversed(b)] for b in bitstrings]

        samples = np.array(samples, dtype='b')
        np.save(configs[i].results_folder + 'samples.npy', samples)


def remove_idle_qwires(qc):
    dag = circuit_to_dag(qc)

    idle_wires = list(dag.idle_wires())
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    dag.qregs = OrderedDict()

    return dag_to_circuit(dag)