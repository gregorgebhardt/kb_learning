import abc

from cluster_work import ClusterWork

from kb_learning.kernel import StateKernel, StateActionKernel
from kb_learning.kernel import compute_median_bandwidth
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
            'num_SARS_samples': 10000
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
        'learn_iterations': 1,
        'lstd': {
            'discount_factor': .99,
            'num_features': 200
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

        self._kilobots_columns = None
        self._light_columns = None
        self._state_columns = None
        self._action_columns = None
        self._SARS_columns = None

        self._SARS = None

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        params = config['params']

        _num_episodes = params['num_episodes']
        _steps_per_episode = params['steps_per_episode']

        # create data frames to store data
        _index = pd.MultiIndex.from_product([range(_num_episodes), range(_steps_per_episode)])

        it_sars = pd.DataFrame(index=_index, columns=self._SARS_columns)

        _state = self._environment.get_state()
        _s = np.r_[_state['kilobots'].flat, _state['light'].flat]

        # generate samples
        for ep in range(params['num_episodes']):
            for step in range(params['steps_per_episode']):
                it_sars.loc[(ep, step), 'S'] = _s

                _action = self.policy(_s)
                _state, _reward, _done, _info = self._environment.step(_action)

                # collect samples in DataFrame
                _s = np.r_[_state['kilobots'].flat, _state['light'].flat]
                it_sars.loc[(ep, step), ['A', 'R', 'S_']] = np.r_[_action, _reward, _s]

                if _info:
                    print(_info)

        # add samples to data set and select subset
        _extended_SARS = self._SARS.append(it_sars, ignore_index=True)
        self._SARS = _extended_SARS.sample(params['samples']['num_SARS_samples'])

        # compute kernel parameters
        bandwidth = compute_median_bandwidth(self._SARS[['S', 'A']], sample_size=500)
        bandwidth_s = bandwidth[:len(self._state_columns)]

        self._state_kernel.set_params(bandwidth=bandwidth_s)
        self._state_action_kernel.set_params(bandwidth=bandwidth)

        # compute feature matrices
        lstd_samples = self._SARS[['S', 'A']].sample(params['lstd']['num_features'])

        phi_s = self._state_kernel(self._SARS['S'], lstd_samples['S'])
        phi_sa = self._state_action_kernel(self._SARS[['S', 'A']], lstd_samples[['S', 'A']])

        for i in range(params['learn_iterations']):
            # compute feature expectation
            phi_sa_hat = self._getFeatureExpectation(self._SARS['S_'], 5, lstd_samples[['S', 'A']])

            # learn theta (parameters of Q-function) using lstd
            theta = self.lstd.learnLSTD(phi_sa, phi_sa_hat, self._SARS['R'])

            # compute sample weights using AC-REPS
            q_fct = phi_sa * theta
            weights = self.ac_reps.compute_weights(q_fct, phi_s)

            # get subset for sparse GP
            gp_samples = self._SARS['S'].sample(params['gp']['num_sparse_states'])

            # fit weighted GP to samples
            self.policy.train(self._SARS['S'], self._SARS['A'], weights, gp_samples)

        # TODO save some stuff
        # TODO compile return dict (what to save after each iteration?)
        return_dict = dict()

        return return_dict

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

        _kb_columns_level = ['kb_{}'.format(i) for i in range(params['sampling']['num_kilobots'])]
        self._kilobots_columns = pd.MultiIndex.from_product([_kb_columns_level, ['x', 'y']])
        self._light_columns = pd.MultiIndex.from_product([['light'], ['x', 'y']])
        self._state_columns = self._kilobots_columns.append(self._light_columns)
        self._action_columns = pd.MultiIndex.from_product([['action'], ['x', 'y']])

        self._SARS_columns = pd.MultiIndex.from_product([['S'] * len(self._state_columns) +
                                                        ['A'] * len(self._action_columns) + ['R'] +
                                                        ['S_'] * len(self._state_columns),
                                                        [*range(len(self._state_columns))] +
                                                        [*range(len(self._action_columns))] + [0] +
                                                        [*range(len(self._state_columns))]])

        self._SARS = pd.DataFrame(index=range(params['samples']['num_SARS_samples']),
                                  columns=self._SARS_columns)

        self._SARS.sample()

    def _getFeatureExpectation(self, states, n, feature_state_actions):
        actions = self.policy.sampleActions(states)
        state_actions = np.concatenate((states, actions), axis=1)

        features = self._state_action_kernel(state_actions, feature_state_actions)

        for i in range(n - 1):
            actions = self.policy.sampleActions(states)
            state_actions = np.concatenate((states, actions), axis=1)
            features += self._state_action_kernel(state_actions, feature_state_actions)

        return features / n
