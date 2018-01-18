import abc
import os
import gc

import multiprocessing
from typing import Tuple

from cluster_work import ClusterWork

from kb_learning.kernel import StateKernel, StateActionKernel
from kb_learning.kernel import compute_median_bandwidth
from kb_learning.reps.lstd import LeastSquaresTemporalDifferenceOptimized, LeastSquaresTemporalDifference
from kb_learning.reps.ac_reps import ActorCriticReps
from kb_learning.reps.sparse_gp_policy import SparseGPPolicy

# import pyximport
# pyximport.install()
#
# from kb_learning.reps.sparse_gp_policy_c import SparseGPPolicy

import gym
import kb_learning.envs as kb_envs
from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import CircularGradientLight

import numpy as np
import pandas as pd

import pickle

from kb_learning.tools import np_chunks


class KilobotSampler:
    _VERBOSE = False

    def __init__(self, num_episodes: int, num_steps_per_episode: int, *args, **kwargs):
        self.num_episodes = num_episodes
        self.num_steps_per_episode = num_steps_per_episode

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError


class KilobotLearner(ClusterWork):
    _sampler_class = None

    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        raise NotImplementedError


class ACRepsLearner(KilobotLearner):
    _default_params = {
        'sampling': {
            'num_kilobots': 15,
            'w_factor': .0,
            'num_episodes': 100,
            'num_steps_per_episode': 125,
            'num_SARS_samples': 10000
        },
        'kernel': {
            'bandwidth_factor_kb': 1.,
            'bandwidth_factor_la': 1.,
            'weight': .5,
            'num_processes': 2
        },
        'learn_iterations': 1,
        'lstd': {
            'discount_factor': .99,
            'num_features': 1000,
            'num_policy_samples': 1,
        },
        'reps': {
            'epsilon': .3
        },
        'gp': {
            'min_variance': .0,
            'regularizer': .05,
            'num_sparse_states': 1000,
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
                                             sampling_params['w_factor'],
                                             sampling_params['num_kilobots'],
                                             parallel=True)

        mean_R = it_sars['R'].groupby(level=1).sum().mean().values
        if self._VERBOSE:
            print('mean reward: {}'.format(mean_R))

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
            print('selecting lstd samples')
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
            # print('learning iteration _{}_'.format(i))

            # compute features
            if self._VERBOSE:
                print('lstd: compute features')
            phi_SA = state_action_features(self._SARS['S'].values, self._SARS['A'].values)
            phi_SA_next = state_action_features(self._SARS['S'].values, self.policy(self._SARS['A'].values))

            # learn theta (parameters of Q-function) using lstd
            if self._VERBOSE:
                print('lstd: learning theta')

            # theta = self.lstd.learn_q_function(feature_mapping=state_action_features,
            #                                    feature_dim=params['lstd']['num_features'], policy=self.policy,
            #                                    num_policy_samples=params['lstd']['num_policy_samples'],
            #                                    states=self._SARS['S'].values, actions=self._SARS['A'].values,
            #                                    rewards=self._SARS['R'].values, next_states=self._SARS['S_'].values,
            #                                    chunk_size=100)

            theta = self.lstd.learn_q_function(phi_SA, phi_SA_next, rewards=self._SARS['R'].values)

            # compute q-function
            if self._VERBOSE:
                print('ac-reps: compute q-function')
            q_fct = phi_SA.dot(theta)

            # compute state features
            if self._VERBOSE:
                print('ac-reps: compute state features')
            phi_S = state_features(self._SARS['S'].values)

            # for s, a in zip(np_chunks(self._SARS['S'].values, 100), np_chunks(self._SARS['S'].values, 100)):
            #     q_fct = np.append(q_fct, state_action_features(s, a).dot(theta))
            #     phi_S = np.append(phi_S, state_features(s))
            # q_fct = state_action_features(self._SARS['S'].values, self._SARS['A'].values).dot(theta)

            # compute sample weights using AC-REPS
            if self._VERBOSE:
                print('ac-reps: learning weights')
            weights = self.ac_reps.compute_weights(q_fct, phi_S)

            # get subset for sparse GP
            if self._VERBOSE:
                print('gp: select samples for gp')
            gp_samples = self._SARS['S'].sample(params['gp']['num_sparse_states']).values

            # fit weighted GP to samples
            if self._VERBOSE:
                print('gp: fitting to policy')
            self.policy.train(self._SARS['S'].values, self._SARS['A'].values, weights, gp_samples)

        # save some stuff
        log_path = os.path.join(config['log_path'], '{:02d}'.format(rep), '{:02d}'.format(n), '')
        os.makedirs(log_path, exist_ok=True)
        with open(log_path + 'policy.pkl', mode='w+b') as policy_file:
            pickle.dump(self.policy, policy_file)

        return_dict = dict(mean_R=mean_R)

        gc.collect()

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

        self.lstd = LeastSquaresTemporalDifference()
        # self.lstd = LeastSquaresTemporalDifference()
        self.lstd.discount_factor = params['lstd']['discount_factor']

        self.ac_reps = ActorCriticReps()
        self.ac_reps.epsilon_action = params['reps']['epsilon']

        self.policy = SparseGPPolicy(kernel=self._state_kernel, action_space=CircularGradientLight().action_space,
                                     gp_min_variance=0.005, gp_regularizer=0.001)

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

    def _get_samples_parallel(self, num_episodes: int, num_steps_per_episode: int, w_factor: float, num_kilobots: int):
        num_workers = 4

        pool = multiprocessing.Pool(num_workers)
        # TODO replace number of workers with parameter
        episodes_per_worker = [num_episodes // num_workers] * num_workers
        episodes_per_worker[-1] += num_episodes % num_workers  # the last one takes the shitload

        samplers = list(map(lambda eps_p_w: self._sampler_class(eps_p_w, num_steps_per_episode, w_factor,
                                                                num_kilobots, self.policy),
                            episodes_per_worker))

        results = pool.map(self._sampler_class.__call__, samplers)

        pool.close()

        # combine results
        it_sars_data = results[0][0]
        it_info = results[0][1]

        for sars_i, info_i in results[1:]:
            it_sars_data = np.concatenate((it_sars_data, sars_i))
            it_info += info_i

        _index = pd.MultiIndex.from_product([range(num_episodes), range(num_steps_per_episode)])
        it_sars = pd.DataFrame(data=it_sars_data, index=_index, columns=self._SARS_columns)
        return it_sars, it_info

    def _get_samples(self, num_episodes: int, num_steps_per_episode: int, w_factor: float, num_kilobots: int,
                     parallel: bool = False):
        if parallel:
            return self._get_samples_parallel(num_episodes, num_steps_per_episode, w_factor, num_kilobots)

        sampler = self._sampler_class(num_episodes, num_steps_per_episode, w_factor, num_kilobots, self.policy)

        it_sars_data, it_info = sampler()

        _index = pd.MultiIndex.from_product([range(num_episodes), range(num_steps_per_episode)])
        it_sars = pd.DataFrame(data=it_sars_data, index=_index, columns=self._SARS_columns)
        return it_sars, it_info


class QuadPushingSampler(KilobotSampler):
    def __init__(self, num_episodes: int, num_steps_per_episode: int,
                 w_factor: float, num_kilobots: int, policy=None):
        super().__init__(num_episodes, num_steps_per_episode)

        self.w_factor = w_factor
        self.num_kilobots = num_kilobots

        self.policy = policy

    def set_policy(self, policy):
        self.policy = policy

    def __call__(self):
        env_id = kb_envs.get_quadpushing_environment(weight=self.w_factor, num_kilobots=self.num_kilobots)
        env = gym.make(env_id)

        state = env.get_state()
        state_dims = len(state) - 3

        # TODO remove magic numbers
        it_sars_data = np.empty((self.num_episodes * self.num_steps_per_episode, 2 * state_dims + 3))
        it_info = []

        # TODO sample episodes in parallel
        # generate samples
        for ep in range(self.num_episodes):
            print(ep)
            env.reset()
            state = env.get_state()
            state = state[:-3]
            if self._VERBOSE:
                env.render()
            for step in range(self.num_steps_per_episode):
                i = ep * self.num_steps_per_episode + step

                it_sars_data[i, :state_dims] = state

                action = self.policy(state.reshape((1, -1)))[0]
                state, reward, done, info = env.step(action)
                state = state[:-3]

                # collect samples in DataFrame
                it_sars_data[i, state_dims:] = np.r_[action, reward, state]

                it_info.append(info)

                if self._VERBOSE and info:
                    print(info)

                if done:
                    return it_sars_data, it_info

        return it_sars_data, it_info


class QuadPushingParallelSampler(QuadPushingSampler):
    def __init__(self, num_episodes: int, num_steps_per_episode: int,
                 w_factor: float, num_kilobots: int, policy=None):
        super().__init__(num_episodes, num_steps_per_episode, w_factor, num_kilobots, policy)

    def __call__(self):
        env_id = kb_envs.get_quadpushing_environment(weight=self.w_factor, num_kilobots=self.num_kilobots)
        envs = [gym.make(env_id) for i in range(self.num_episodes)]

        states = np.array([e.get_state() for e in envs])
        reward = np.empty((self.num_episodes, 1))
        info = list()

        state_dims = states.shape[1] - 3

        # TODO remove magic numbers
        it_sars_data = np.empty((self.num_episodes * self.num_steps_per_episode, 2 * state_dims + 3))
        # it_info = []

        for step in range(self.num_steps_per_episode):
            print(step)
            # i = self.num_steps_per_episode + step

            it_sars_data[step::self.num_steps_per_episode, :state_dims] = states[:, :-3]

            actions = self.policy(states[:, :-3])
            srdi = [e.step(a) for e, a in zip(envs, actions)]
            for i in range(self.num_episodes):
                states[i, :] = srdi[i][0]
                reward[i] = srdi[i][1]
                info.append(srdi[i][3])
            # state = state[:-3]

            # collect samples in DataFrame
            it_sars_data[step::self.num_steps_per_episode, state_dims:] = np.c_[actions, reward, states[:, :-3]]

            # if done:
            #     return it_sars_data, it_info

        return it_sars_data, info


class QuadPushingACRepsLearner(ACRepsLearner):
    _sampler_class = QuadPushingParallelSampler
    # _sampler_class._VERBOSE = True
