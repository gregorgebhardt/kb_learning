import abc

from cluster_work import ClusterWork

from kb_learning.kernel import StateKernel
from kb_learning.reps.sparse_gp_policy import SparseGPPolicy


class KilobotLearner(ClusterWork):
    def __init__(self, environment):
        self._environment = environment

    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        raise NotImplementedError


class AcRepsLearner(KilobotLearner):
    _default_params = {
        'sampling': {
            'num_kilobots': 1,
            'num_episodes': 121,
            'num_steps_per_episode': 1,
            'num_samples_per_iteration': 20,
            'num_SARS_samples': 100,
            'sampling_type_ratio': -.95,
            'sampling_max_radius': .5,
            'spawn_radius_variance': .5
        },
        'kernel': {
            'bandwidth_factor_kb': 1.,
            'bandwidth_factor_la': 1.,
            'weight': .5,
            'num_processes': 4
        },
        'lstd': {
            'discount_factor': .99,
            'num_features': 100
        },
        'reps': {
            'epsilon': .5
        },
        'gp': {
            'min_variance': .0,
            'regularizer': .05,
            'num_sparse_states': 100,
            'num_learn_iterations': 1,
            'epsilon': .0,
            'epsilon_factor': 1.
        }
    }

    def __init__(self, weight=.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight = weight

        self.policy = SparseGPPolicy(StateKernel(weight=self.weight), action_space=self._environment.action_space)

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        pass

    def reset(self, config: dict, rep: int) -> None:
        params = config['params']
        pass
