import abc
import os
import gc
import sys

import logging

import multiprocessing

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

from gym_kilobots.lib import CircularGradientLight

from kb_learning.envs.sampler import KilobotSampler
from kb_learning.envs.sampler import ParallelQuadPushingSampler as Sampler


import numpy as np
import pandas as pd

import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KilobotLearner(ClusterWork):
    _sampler_class: KilobotSampler = None

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
            'num_SARS_samples': 10000,
            'num_workers': 4
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

        self._sampler = None

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        params = config['params']
        sampling_params = params['sampling']

        logger.info('sampling environment')
        it_sars, it_info = self._sampler()
        import matplotlib.pyplot as plt
        from kb_learning.tools.plotting import plot_light_trajectory
        fig, ax = plt.subplots(1)
        plot_light_trajectory(it_sars['S'], ax)
        fig.show()

        mean_R = it_sars['R'].groupby(level=1).sum().mean().values
        logger.info('mean reward: {}'.format(mean_R))

        # add samples to data set and select subset
        _extended_SARS = self._SARS.append(it_sars, ignore_index=True)
        if _extended_SARS.shape[0] <= sampling_params['num_SARS_samples']:
            self._SARS = _extended_SARS
        else:
            self._SARS = _extended_SARS.sample(sampling_params['num_SARS_samples'])

        # compute kernel parameters
        logger.debug('computing kernel bandwidths')
        bandwidth = compute_median_bandwidth(self._SARS[['S', 'A']], sample_size=500)
        bandwidth_s = bandwidth[:len(self._state_columns)]

        self._state_kernel.set_params(bandwidth=bandwidth_s)
        self._state_action_kernel.set_params(bandwidth=bandwidth)

        # compute feature matrices
        logger.debug('selecting lstd samples')
        lstd_samples = self._SARS[['S', 'A']].sample(params['lstd']['num_features'])

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
            # compute features
            logger.info('compute state-action features')
            phi_SA = state_action_features(self._SARS['S'].values, self._SARS['A'].values)
            phi_SA_next = state_action_features(self._SARS['S_'].values, self.policy(self._SARS['S_'].values))

            # learn theta (parameters of Q-function) using lstd
            logger.info('learning theta [LSTD]')
            theta = self.lstd.learn_q_function(phi_SA, phi_SA_next, rewards=self._SARS['R'].values)

            # compute q-function
            logger.debug('compute q-function')
            q_fct = phi_SA.dot(theta)

            # compute state features
            logger.debug('compute state features')
            phi_S = state_features(self._SARS['S'].values)

            # compute sample weights using AC-REPS
            logger.info('learning weights [AC-REPS]')
            weights = self.ac_reps.compute_weights(q_fct, phi_S)

            # get subset for sparse GP
            logger.debug('select samples for GP')
            gp_samples = self._SARS['S'].sample(params['gp']['num_sparse_states']).values

            # fit weighted GP to samples
            logger.info('fitting GP to policy')
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
                                         bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                         bandwidth_light=kernel_params['bandwidth_factor_la'],
                                         num_processes=kernel_params['num_processes'])

        self._state_action_kernel = StateActionKernel(weight=kernel_params['weight'],
                                                      bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                                      bandwidth_light=kernel_params['bandwidth_factor_la'],
                                                      num_processes=kernel_params['num_processes'])

        self.lstd = LeastSquaresTemporalDifference()
        # self.lstd = LeastSquaresTemporalDifference()
        self.lstd.discount_factor = params['lstd']['discount_factor']

        self.ac_reps = ActorCriticReps()
        self.ac_reps.epsilon_action = params['reps']['epsilon']

        self.policy = SparseGPPolicy(kernel=self._state_kernel, action_space=CircularGradientLight().action_space)

        kb_columns_level = ['kb_{}'.format(i) for i in range(params['sampling']['num_kilobots'])]
        kilobots_columns = pd.MultiIndex.from_product([['S'], kb_columns_level, ['x', 'y']])
        light_columns = pd.MultiIndex.from_product([['S'], ['light'], ['x', 'y']])
        state_columns = kilobots_columns.append(light_columns)
        action_columns = pd.MultiIndex.from_product([['A'], ['x', 'y'], ['']])
        reward_columns = pd.MultiIndex.from_arrays([['R'], [''], ['']])
        next_state_columns = state_columns.copy()
        next_state_columns.set_levels(['S_'], 0, inplace=True)

        # self._SARS_columns = pd.MultiIndex.from_arrays([['S'] * len(state_columns) +
        #                                                 ['A'] * len(action_columns) + ['R'] +
        #                                                 ['S_'] * len(state_columns),
        #                                                 [*range(len(state_columns))] +
        #                                                 [*range(len(action_columns))] + [0] +
        #                                                 [*range(len(state_columns))]])

        self._SARS_columns = state_columns.append(action_columns).append(reward_columns).append(next_state_columns)

        self._SARS = pd.DataFrame(columns=self._SARS_columns)

        sampling_params = params['sampling']
        self._sampler = self._sampler_class(num_episodes=sampling_params['num_episodes'],
                                            num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                            num_kilobots=sampling_params['num_kilobots'],
                                            column_index=self._SARS_columns,
                                            w_factor=sampling_params['w_factor'],
                                            policy=self.policy,
                                            num_workers=sampling_params['num_workers'])


class QuadPushingACRepsLearner(ACRepsLearner):
    _sampler_class = Sampler
