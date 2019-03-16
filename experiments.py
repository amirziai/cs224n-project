import pickle
from itertools import product
from typing import Dict, Set, List, Any, Tuple
from uuid import uuid4

import pandas as pd
from joblib import Parallel, delayed

import params
import run
from run import ExperimentResults
from utils import log, merge_dicts

ParamSet = Dict[str, Any]
ParamGrid = List[ParamSet]
RunnerUUID = str


class ExperimentRunner:
    def __init__(self, experiment_parameters: Dict[str, Set[Any]], n_jobs: int):
        self.experiment_parameters = experiment_parameters
        self.n_jobs = n_jobs
        self.uuid = uuid4()

    @staticmethod
    def _get_param_grid(parameters: Dict[str, Set[Any]]) -> ParamGrid:
        return [dict(zip(parameters.keys(), t)) for t in product(*parameters.values())]

    @staticmethod
    def _pickle(obj: Any, file_path_out) -> None:
        with open(file_path_out, 'wb') as f:
            pickle.dump(obj, f)

    def _param_run(self, param_set: ParamSet) -> Tuple[ExperimentResults, RunnerUUID]:
        log(f'Running param set: {param_set}')
        runner = run.Runner(**param_set)
        experiment_results = runner.run()

        # persist
        self._pickle(experiment_results, f'results/{self.uuid}_experiment_results_exp_{runner.uuid}.pkl')
        self._pickle(runner, f'results/{self.uuid}_runner_exp_{runner.uuid}.pkl')
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
