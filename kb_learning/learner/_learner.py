import abc
import os

import multiprocessing
from typing import Tuple

from cluster_work import ClusterWork

from kb_learning.kernel import StateKernel, StateActionKernel
from kb_learning.kernel import compute_median_bandwidth
from kb_learning.reps.lstd import LeastSquarsTemporalDifferenceOptimized
from kb_learning.reps.ac_reps import ActorCriticReps
from kb_learning.reps.sparse_gp_policy import SparseGPPolicy

import gym
import kb_learning.envs as kb_envs
from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import CircularGradientLight

import numpy as np
import pandas as pd

import pickle


class KilobotLearner(ClusterWork):
    def __init__(self):
        self._env_id: str = None

    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        raise NotImplementedError


class ACRepsLearner(KilobotLearner):
    _default_params = {
        'sampling': {
            'num_kilobots': 10,
            'w_factor': .0,
            'num_episodes': 120,
            'num_steps_per_episode': 1,
            'num_samples_per_iteration': 20,  # todo: clarify this variable
            'num_SARS_samples': 5000
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
            'num_features': 200,
            'num_policy_samples': 5,
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
        sampling_params = params['sampling']

        if self._VERBOSE:
            print('sampling environment')
        it_sars, it_info = self._get_samples(sampling_params['num_episodes'],
                                             sampling_params['num_steps_per_episode'],
                                             sampling_params['w_factor'], sampling_params['num_kilobots'])

        # TODO reward is always zero!!!!

        # add samples to data set and select subset
        _extended_SARS = self._SARS.append(it_sars, ignore_index=True)
        if _extended_SARS.shape[0] <= sampling_params['num_SARS_samples']:
            self._SARS = _extended_SARS
        else:
            self._SARS = _extended_SARS.sample(sampling_params['num_SARS_samples'])

        # compute kernel parameters
        if self._VERBOSE:
            print('computing kernel bandwidths')
        bandwidth = compute_median_bandwidth(self._SARS[['S', 'A']], sample_size=500)
        bandwidth_s = bandwidth[:len(self._state_columns)]

        self._state_kernel.set_params(bandwidth=bandwidth_s)
        self._state_action_kernel.set_params(bandwidth=bandwidth)

        # compute feature matrices
        if self._VERBOSE:
            print('computing lstd features')
        lstd_samples = self._SARS[['S', 'A']].sample(params['lstd']['num_features'])

        # phi_s = self._state_kernel(self._SARS['S'].values, lstd_samples['S'].values)
        # phi_sa = self._state_action_kernel(self._SARS[['S', 'A']].values, lstd_samples[['S', 'A']].values)

        def state_features(state):
            if state.ndim == 1:
                state = state.reshape((1, -1))
            return self._state_kernel(state, lstd_samples['S'].values)

        def state_action_features(state, action):
            if state.ndim == 1:
                state = state.reshape((1, -1))
            if action.ndim == 1:
                action = action.reshape((1, -1))
            return self._state_action_kernel(np.c_[state, action], lstd_samples[['S', 'A']].values)

        for i in range(params['learn_iterations']):
            # compute feature expectation (is included in lstd code)
            # learn theta (parameters of Q-function) using lstd
            if self._VERBOSE:
                print('lstd: learning theta')
            theta = self.lstd.learn_q_function(feature_mapping=state_action_features,
                                               feature_dim=params['lstd']['num_features'], policy=self.policy,
                                               num_policy_samples=params['lstd']['num_policy_samples'],
                                               states=self._SARS['S'].values, actions=self._SARS['A'].values,
                                               rewards=self._SARS['R'].values, next_states=self._SARS['S_'].values,
                                               chunk_size=10)

            # compute sample weights using AC-REPS
            if self._VERBOSE:
                print('ac-reps: learning weights')
            q_fct = state_action_features(self._SARS['S'].values, self._SARS['A'].values).dot(theta)
            weights = self.ac_reps.compute_weights(q_fct, state_features(self._SARS['S'].values))

            # get subset for sparse GP
            if self._VERBOSE:
                print('gp: fitting to policy')
            gp_samples = self._SARS['S'].sample(params['gp']['num_sparse_states']).values

            # fit weighted GP to samples
            self.policy.train(self._SARS['S'].values, self._SARS['A'].values, weights, gp_samples)

        # save some stuff
        log_path = os.path.join(config['log_path'], '{:02d}'.format(rep), '{:02d}'.format(n), '')
        with open(log_path + 'policy.pkl', mode='w') as policy_file:
            pickle.dump(self.policy, policy_file)

        # evaluate policy
        if self._VERBOSE:
            print('evaluating learned policy')
        it_sars, _ = self._get_samples(sampling_params['num_episodes'],
                                       sampling_params['num_steps_per_episode'],
                                       sampling_params['w_factor'], sampling_params['num_kilobots'])

        return_dict = dict(mean_R_old=self._SARS['R'].groupby(level=1).sum().mean(),
                           mean_R_new=it_sars['R'].groupby(level=1).sum().mean())

        return return_dict

    def reset(self, config: dict, rep: int) -> None:
        params = config['params']

        kernel_params = params['kernel']
        self._state_kernel = StateKernel(weight=kernel_params['weight'],
                                         bandwidth_kb=kernel_params['bandwidth_factor_kb'],
                                         bandwidth_light=kernel_params['bandwidth_factor_la'],
                                         num_processes=kernel_params['num_processes'])

        self._state_action_kernel = StateActionKernel(weight=kernel_params['weight'],
                                                      bandwidth_kb=kernel_params['bandwidth_factor_kb'],
                                                      bandwidth_light=kernel_params['bandwidth_factor_la'],
                                                      num_processes=kernel_params['num_processes'])

        self.lstd = LeastSquarsTemporalDifferenceOptimized()
        self.lstd.discount_factor = params['lstd']['discount_factor']

        self.ac_reps = ActorCriticReps()
        self.ac_reps.epsilon_action = params['reps']['epsilon']

        self.policy = SparseGPPolicy(kernel=self._state_kernel, action_space=CircularGradientLight().action_space)

        _kb_columns_level = ['kb_{}'.format(i) for i in range(params['sampling']['num_kilobots'])]
        self._kilobots_columns = pd.MultiIndex.from_product([_kb_columns_level, ['x', 'y']])
        self._light_columns = pd.MultiIndex.from_product([['light'], ['x', 'y']])
        self._state_columns = self._kilobots_columns.append(self._light_columns)
        self._action_columns = pd.MultiIndex.from_product([['action'], ['x', 'y']])

        self._SARS_columns = pd.MultiIndex.from_arrays([['S'] * len(self._state_columns) +
                                                        ['A'] * len(self._action_columns) + ['R'] +
                                                        ['S_'] * len(self._state_columns),
                                                        [*range(len(self._state_columns))] +
                                                        [*range(len(self._action_columns))] + [0] +
                                                        [*range(len(self._state_columns))]])

        self._SARS = pd.DataFrame(columns=self._SARS_columns)

    @staticmethod
    @abc.abstractmethod
    def _get_environment(w_factor: float, num_kilobots: int) -> Tuple[KilobotsEnv, str]:
        pass

    def _simulate_episode(self, num_steps):
        env = gym.make(self._env_id)
        state = env.get_state()
        state_dims = len(state)
        sars_columns = 2 * state_dims + 1 + len(env.action_space.sample())

        ep_sars = np.empty((num_steps, sars_columns))
        ep_info = []

        # if self._VERBOSE:
        #     env.render()

        for step in range(num_steps):
            ep_sars[step, :state_dims] = state

            action = self.policy(state.reshape((1, -1)))[0]
            state, reward, done, info = env.step(action)

            # collect samples in DataFrame
            ep_sars[step, state_dims:] = np.r_[action, reward, state]

            ep_info.append(info)

            if self._VERBOSE and info:
                print(info)

            if done:
                return ep_sars, ep_info

    def _get_samples_parallel(self, num_episodes: int, steps_per_episode: int):
        pool = multiprocessing.Pool(4)
        results = pool.map(lambda ep: self._simulate_episode(steps_per_episode), range(num_episodes))
        # TODO implement

        pass

    def _get_samples(self, num_episodes, num_steps_per_episode, w_factor, num_kilobots):

        it_sars_data = np.empty((num_episodes * num_steps_per_episode, len(self._SARS_columns)))
        it_info = []

        env, env_id = self._get_environment(w_factor, num_kilobots)

        state = env.get_state()
        state_dims = len(state)

        # generate samples
        for ep in range(num_episodes):
            env.reset()
            # if self._VERBOSE:
            #     env.render()
            for step in range(num_steps_per_episode):
                i = ep * num_steps_per_episode + step

                it_sars_data[i, :state_dims] = state

                action = self.policy(state.reshape((1, -1)))[0]
                state, reward, done, info = env.step(action)

                # collect samples in DataFrame
                it_sars_data[i, state_dims:] = np.r_[action, reward, state]

                it_info.append(info)

                if self._VERBOSE and info:
                    print(info)

                if done:
                    _index = pd.MultiIndex.from_product([range(num_episodes), range(num_steps_per_episode)])
                    it_sars = pd.DataFrame(data=it_sars_data, index=_index, columns=self._SARS_columns)
                    return it_sars, it_info

        _index = pd.MultiIndex.from_product([range(num_episodes), range(num_steps_per_episode)])
        it_sars = pd.DataFrame(data=it_sars_data, index=_index, columns=self._SARS_columns)
        return it_sars, it_info


class QuadPushingACRepsLearner(ACRepsLearner):
    @staticmethod
    def _get_environment(w_factor: float, num_kilobots: int):
        env_id = kb_envs.register_quadpushing_environment(w_factor, num_kilobots)
        environment = gym.make(env_id)

        return environment, env_id
