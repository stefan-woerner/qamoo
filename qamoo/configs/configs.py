import numpy as np
from dataclasses import dataclass
import json
from typing import List

from qamoo.utils.data_structures import ProblemGraphBuilder


@dataclass
class ProblemSpecification:

    data_folder: str = ''
    num_qubits: int = 0
    num_objectives: int = 0
    num_swap_layers: int = 0
    problem_id: int = 0

    @property
    def problem_folder(self):
        return self.data_folder + \
            f'problems/{self.num_qubits}q/'+\
            f'problem_set_{self.num_qubits}q_{self.num_swap_layers}s_{self.num_objectives}o_{self.problem_id}/'

    @property
    def objective_graphs(self):
        return [
                ProblemGraphBuilder.deserialize(
                    self.problem_folder + f'problem_graph_{idx}.json') 
                    for idx in range(self.num_objectives)
            ]
    
    @property
    def lower_bounds(self):
        return np.array(json.load(open(self.problem_folder + 'lower_bounds.json')))

    @property        
    def upper_bounds(self):
        return np.array(json.load(open(self.problem_folder + 'upper_bounds.json')))
    
    @property
    def optimal_hypervolume(self):
        try:
            return np.array(json.load(open(self.problem_folder + 'optimal_hypervolume.json')))
        except:
            return None


@dataclass
class AlgorithmConfig:

    name: str = ''
    problem: ProblemSpecification = None
    run_id: int = 0
    num_steps: int = 0

    @property
    def total_num_samples(self):
        raise NotImplementedError()

    @property
    def algorithm_identifier(self):
        raise NotImplementedError()

    @property
    def results_folder(self):
        return self.problem.problem_folder + \
            f'results/{self.algorithm_identifier}/'
    
    @property
    def progress(self):
        results_folder = self.results_folder
        with open(results_folder + 'progress.json') as f:
            progress = json.load(f)
        return progress

    
    @property
    def runtime(self):
        results_folder = self.results_folder
        with open(results_folder + 'runtime.json') as f:
            runtime = json.load(f)
        return runtime

    def progress_x_y(self):
        progress = self.progress
        x = list(progress.keys())
        x = [int(x_) for x_ in x]     
        y = [progress[str(x_)] for x_ in x]
        return x, y


@dataclass
class AlgorithmConfigWithObjectiveWeights(AlgorithmConfig):

    objective_weights_id: int = 0
    num_samples: int = 0

    @property
    def objective_weights_file(self):
        return self.problem.data_folder + \
            f'objective_weights/objective_weights_{self.problem.num_objectives}o_{self.objective_weights_id}.json'
    
    @property
    def objective_weights(self):
        with open(self.objective_weights_file, 'r') as f:
            return json.load(f)[:self.num_samples]


@dataclass
class RandomConfig(AlgorithmConfig):

    name: str = 'random'
    num_samples: int = 0

    @property
    def total_num_samples(self):
        return self.num_samples

    @property
    def algorithm_identifier(self):
        return f'{self.name}_{self.num_samples}_samples_{self.run_id}'
    

@dataclass
class WeightedSumConfig(AlgorithmConfigWithObjectiveWeights):

    name: str = 'weighted_sum'

    @property
    def total_num_samples(self):
        return self.num_samples

    @property
    def algorithm_identifier(self):
        return f'{self.name}_{self.num_samples}_samples_{self.run_id}'


@dataclass
class GoemansWilliamsonConfig(AlgorithmConfigWithObjectiveWeights):

    name: str = 'goemans_williamson'
    shots: int = 0

    @property
    def total_num_samples(self):
        return self.num_samples * self.shots

    @property
    def algorithm_identifier(self):
        return f'{self.name}_{self.num_samples}_samples_{self.shots}_shots_{self.run_id}'


@dataclass
class BoostingConfig(AlgorithmConfigWithObjectiveWeights):

    name: str = 'boosting'
    input_config: AlgorithmConfig = None

    @property
    def total_num_samples(self):
        return self.input_config.total_num_samples
    
    @property
    def algorithm_identifier(self):
        return self.input_config.algorithm_identifier + '_boosted'


@dataclass
class RandomEpsConstraintConfig(AlgorithmConfigWithObjectiveWeights):

    name: str = 'random_eps_constraint'
    shots: int = 0  # = number of random eps samples per objective weights
    eps_seed: int = 42

    @property
    def total_num_samples(self):
        return self.num_samples * self.shots

    @property
    def algorithm_identifier(self):
        return f'{self.name}_{self.num_samples}_samples_{self.shots}_shots_{self.run_id}'
    


@dataclass
class QAOAConfig(AlgorithmConfigWithObjectiveWeights):

    name: str = 'qaoa'
    p: int = 0
    parameter_file: str = None
    shots: int = 0
    backend_name: str = ''
    initial_layout: List[int] = None
    v2: bool = True
    dynamic_decoupling: bool = True
    m3: bool = False
    twirling: bool = False
    run_id: int = 0
    rep_delay: int = None

    @property
    def total_num_samples(self):
        return self.num_samples * self.shots

    @property
    def optimal_parameters(self):
        all_parameters = None
        file = self.parameter_file if self.parameter_file is not None \
                else self.problem.problem_folder + 'optimal_parameters.json'
        with open(file) as f:
            all_parameters = json.load(f)
        return all_parameters[str(self.p)]
            
    @property
    def algorithm_identifier(self):
        version = 'v2' if self.v2 else 'v1'
        return f'{self.name}_{self.backend_name}_{version}_' +\
               f'p_{self.p}_{self.num_samples}_samples_{self.shots}_shots_{self.run_id}'
