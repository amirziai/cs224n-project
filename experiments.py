from itertools import product
from typing import Dict, Set, List, Any

import params
import run
import pickle
import pandas as pd
from uuid import uuid4
from joblib import Parallel, delayed
from utils import log, merge_dicts

ParamSet = Dict[str, Any]
ParamGrid = List[ParamSet]


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

    def _param_run(self, param_set: ParamSet):
        log(f'Running param set: {param_set}')
        runner = run.Runner(**param_set)
        experiment_results = runner.run()

        # persist
        self._pickle(experiment_results, f'results/runner_experiment_results_{self.uuid}_exp_{runner.uuid}.pkl')
        self._pickle(runner, f'results/runner_{self.uuid}_exp_{runner.uuid}.pkl')
        return experiment_results

    def run(self):
        param_grid = self._get_param_grid(self.experiment_parameters)
        if self.n_jobs > 1:
            results = Parallel(n_jobs=self.n_jobs)(delayed(self._param_run)(param) for param in param_grid)
        else:
            results = [self._param_run(param) for param in param_grid]
        results_enriched = [merge_dicts(result._asdict(), param_set) for result, param_set in zip(results, param_grid)]
        pd.DataFrame(results_enriched).to_csv(f'results/results_{self.uuid}.csv', index=False)


if __name__ == '__main__':
    experiment1 = ExperimentRunner(params.experiment1, n_jobs=params.n_jobs)
    experiment1.run()
