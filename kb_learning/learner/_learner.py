import abc

from cluster_work import ClusterWork

from kb_learning.kernel import StateKernel, StateActionKernel
from kb_learning.reps.lstd import LeastSquaresTemporalDifference
from kb_learning.reps.ac_reps import ActorCriticReps
from kb_learning.reps.sparse_gp_policy import SparseGPPolicy

import gym
import kb_learning.envs as kb_envs
from gym_kilobots.envs import KilobotsEnv

import numpy as np
import pandas as pd


class KilobotLearner(ClusterWork):
    def __init__(self):
        self._environment: KilobotsEnv = None

    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        raise NotImplementedError


class AcRepsLearner(KilobotLearner):
    _default_params = {
        'sampling': {
            'num_kilobots': 10,
            'num_episodes': 120,
            'num_steps_per_episode': 1,
            'num_samples_per_iteration': 20,
            'num_SARS_samples': 100
        },
        'reward': {
            'w_factor': .0
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

    def __init__(self):
        super().__init__()

        self._state_kernel = None
        self._state_action_kernel = None

        self.lstd = None
        self.ac_reps = None
        self.policy = None

        self._states

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        params = config['params']

        _num_episodes = params['num_episodes']
        _steps_per_episode = params['steps_per_episode']

        # create data frames to store data
        _index = pd.MultiIndex.from_product([range(_num_episodes), range(_steps_per_episode)])
        _state_columns = pd.MultiIndex.from_product([['kb_{}'.format(i) for i in range(10)] + ['light'], ['x', 'y']])

        _states = pd.DataFrame(index=_index, columns=_state_columns)
        rewards = pd.Series(index=_index)
        actions = pd.DataFrame(index=_index, columns=['x', 'y'])

        # generate samples
        for ep in range(params['num_episodes']):
            for step in range(params['steps_per_episode']):
                # TODO get initial state
                _action = self.policy(_s)
                _state, _reward, _done, _info = self._environment.step(_action)

                # collect samples in DataFrame
                kilobot_states.loc[ep, step] = _state['kilobots'].flat
                light_states.loc[ep, step] = _state['light'].flat
                rewards.loc[ep, step] = _reward
                actions.loc[ep, step] = _action

                _s = np.r_[_state['kilobots'].flat, _state['light']]

                if _info:
                    print(_info)

        # TODO add samples to data set and select subset


        # TODO compute kernel parameters

        # TODO compute feature matrices

        for i in range(params['learn_iterations']):  # TODO check parameter 'learn_iterations'
            # TODO compute feature expectation

            # TODO learn theta (parameters of Q-function) using lstd

            # TODO compute sample weights using AC-REPS

            # TODO fit weighted GP to samples
            pass

    def reset(self, config: dict, rep: int) -> None:
        params = config['params']

        env_id = kb_envs.register_kilobot_environment(params['reward']['w_factor'], params['sampling']['num_kilobots'])
        self._environment = gym.make(env_id)

        kernel_params = params['kernel']
        self._state_kernel = StateKernel(weight=kernel_params['weight'],
                                         bandwidth_kb=kernel_params['bandwidth_factor_kb'],
                                         bandwidth_light=kernel_params['bandwidth_factor_la'],
                                         num_processes=kernel_params['num_processes'])

        self._state_action_kernel = StateActionKernel(weight=kernel_params['weight'],
                                                      bandwidth_kb=kernel_params['bandwidth_factor_kb'],
                                                      bandwidth_light=kernel_params['bandwidth_factor_la'],
                                                      num_processes=kernel_params['num_processes'])

        self.lstd = LeastSquaresTemporalDifference()
        self.lstd.discount_factor = params['lstd']['discount_factor']

        self.ac_reps = ActorCriticReps()
        self.ac_reps.epsilon_action = params['reps']['epsilon']

        self.policy = SparseGPPolicy(kernel=self._state_kernel, action_space=self._environment.action_space)


