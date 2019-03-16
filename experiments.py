import os
from itertools import product
from typing import Dict, Set, List, Any, Tuple
from uuid import uuid4

import pandas as pd
from joblib import Parallel, delayed

import params
import run
from run import ExperimentResults
from utils import log, merge_dicts, uuid_to_str, pickle_object, unpickle

ParamSet = Dict[str, Any]
ParamGrid = List[ParamSet]
RunnerUUID = str


class ExperimentRunner:
    def __init__(self, experiment_parameters: Dict[str, Set[Any]], n_jobs: int):
        self.experiment_parameters = experiment_parameters
        self.n_jobs = n_jobs
        self.uuid: str = uuid_to_str(uuid4())

    @staticmethod
    def _get_param_grid(parameters: Dict[str, Set[Any]]) -> ParamGrid:
        return [dict(zip(parameters.keys(), t)) for t in product(*parameters.values())]

    @staticmethod
    def _file_path_experiment_results(runner_uuid: RunnerUUID) -> str:
        return f'results/{runner_uuid}_experiment_results.pkl'

    def _experiment_result_exists(self, runner_uuid: RunnerUUID) -> bool:
        return os.path.isfile(self._file_path_experiment_results(runner_uuid))

    def _param_run(self, param_set: ParamSet) -> Tuple[ExperimentResults, RunnerUUID]:
        log(f'Running param set: {param_set}')
        runner = run.Runner(**param_set)
        if self._experiment_result_exists(runner.uuid):
            log('Loading experiment results from cache')
            experiment_results = unpickle(self._file_path_experiment_results(runner.uuid))
        else:
            experiment_results = runner.run()
            pickle_object(experiment_results, self._file_path_experiment_results(runner.uuid))

        return experiment_results, runner.uuid

    def run(self):
        param_grid = self._get_param_grid(self.experiment_parameters)
        if self.n_jobs > 1:
            run_output = Parallel(n_jobs=self.n_jobs)(delayed(self._param_run)(param) for param in param_grid)
        else:
            run_output = [self._param_run(param) for param in param_grid]
        results_enriched = [
            merge_dicts(result._asdict(), param_set, {'runner_uuid': runner_uuid}, {'experiment_uuid': self.uuid})
            for (result, runner_uuid), param_set in zip(run_output, param_grid)
        ]
        pd.DataFrame(results_enriched).to_csv(f'results/results_{self.uuid}.csv', index=False)


if __name__ == '__main__':
    experiment1 = ExperimentRunner(params.experiment1, n_jobs=params.n_jobs)
    experiment1.run()
