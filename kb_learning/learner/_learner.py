import abc
import os
import sys
import gc

import logging

from cluster_work import ClusterWork

from kb_learning.kernel import KilobotStateKernel, KilobotStateActionKernel
from kb_learning.kernel import MeanStateKernel, MeanStateActionKernel
from kb_learning.kernel import MeanCovStateKernel, MeanCovStateActionKernel
from kb_learning.kernel import compute_median_bandwidth
from kb_learning.reps.lstd import LeastSquaresTemporalDifference
from kb_learning.reps.ac_reps import ActorCriticReps
from kb_learning.reps.sparse_gp_policy import SparseGPPolicy

from gym_kilobots.lib import CircularGradientLight

from kb_learning.envs.sampler import KilobotSampler
from kb_learning.envs.sampler import ParallelQuadPushingSampler as Sampler

import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from kb_learning.tools.plotting import plot_light_trajectory, plot_value_function, plot_policy, show_plot_in_browser, \
    save_plot_as_html, plot_objects, plot_trajectory_reward_distribution

import numpy as np
import pandas as pd

import pickle

formatter = logging.Formatter('[%(asctime)s] [KBL] %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger = logging.getLogger('kb_learning')
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class KilobotLearner(ClusterWork):
    Sampler: KilobotSampler = None

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
            'type': 'kilobot',
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
            'prior_variance': .006 ** 2,
            'noise_variance': 1e-6,
            'min_variance': 1e-8,
            'chol_regularizer': 1e-9,
            'num_sparse_states': 1000,
        }
    }

    StateKernel = KilobotStateKernel
    StateActionKernel = KilobotStateActionKernel
    Sampler = Sampler

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

        self._state_features = None
        self._state_action_features = None

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        sampling_params = self._params['sampling']

        logger.info('sampling environment')
        it_sars, it_info = self._sampler()
        # fig, ax = plt.subplots(1)
        # plot_light_trajectory(it_sars['S'], ax)
        # fig.show()

        sum_R = it_sars['R'].groupby(level=0).sum()

        mean_sum_R = sum_R.mean()
        median_sum_R = sum_R.median()
        std_sum_R = sum_R.std()
        max_sum_R = sum_R.max()
        min_sum_R = sum_R.min()

        logger.info('statistics on sum R -- mean: {:.6f} median: {:.6f} std: {:.6f} max: {:.6f} min: {:.6f}'.format(
            mean_sum_R, median_sum_R, std_sum_R, max_sum_R, min_sum_R))

        # add samples to data set and select subset
        _extended_SARS = self._SARS.append(it_sars, ignore_index=True)
        if _extended_SARS.shape[0] <= sampling_params['num_SARS_samples']:
            self._SARS = _extended_SARS
        else:
            self._SARS = _extended_SARS.sample(sampling_params['num_SARS_samples'])

        # compute kernel parameters
        logger.debug('computing kernel bandwidths')
        bandwidth = compute_median_bandwidth(self._SARS[['S', 'A']], sample_size=500)
        bandwidth_s = bandwidth[:len(self._SARS['S'].columns)]

        self._state_kernel.set_params(bandwidth=bandwidth_s)
        self._state_action_kernel.set_params(bandwidth=bandwidth)

        # compute feature matrices
        logger.debug('selecting lstd samples')
        lstd_samples = self._SARS[['S', 'A']].sample(self._params['lstd']['num_features'])

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

        for i in range(self._params['learn_iterations']):
            # compute features
            logger.info('compute state-action features')
            phi_SA = state_action_features(self._SARS['S'].values, self._SARS['A'].values)
            phi_SA_next = state_action_features(self._SARS['S_'].values, self.policy(self._SARS['S_'].values))

            # learn theta (parameters of Q-function) using lstd
            logger.info('learning theta [LSTD]')
            theta = self.lstd.learn_q_function(phi_SA, phi_SA_next, rewards=self._SARS[['R']].values)

            # plotting
            fig = plt.figure(figsize=(10, 18))
            gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[20, 1], height_ratios=[1, 3, 3, 3])
            ax_R = fig.add_subplot(gs[0, :])
            plot_trajectory_reward_distribution(ax_R, it_sars['R'])
            ax_bef_V = fig.add_subplot(gs[1, 0])
            ax_bef_V_cb = fig.add_subplot(gs[1, 1])
            cmap_plasma = cm.get_cmap('plasma')
            cmap_gray = cm.get_cmap('gray')
            V = self._compute_value_function_grid(state_action_features, theta)
            im = plot_value_function(ax_bef_V, *V, cmap=cmap_plasma)
            plot_objects(ax_bef_V, env=self._sampler.env, fill=False)
            ax_bef_V.set_title('value function, before reps, iteration {}'.format(self._it))
            fig.colorbar(im, cax=ax_bef_V_cb)
            ax_bef_T = fig.add_subplot(gs[2, 0])
            ax_bef_T_cb = fig.add_subplot(gs[2, 1])
            ax_bef_T.set_title('trajectories, iteration {}'.format(self._it))
            plot_value_function(ax_bef_T, *V, cmap=cmap_gray)
            plot_objects(ax_bef_T, env=self._sampler.env)
            tr = plot_light_trajectory(ax_bef_T, it_sars['S']['light'])
            fig.colorbar(tr[0], cax=ax_bef_T_cb)

            # compute q-function
            logger.debug('compute q-function')
            q_fct = phi_SA.dot(theta)

            # compute state features
            logger.info('compute state features')
            phi_S = state_features(self._SARS['S'].values)

            # compute sample weights using AC-REPS
            logger.info('learning weights [AC-REPS]')
            weights = self.ac_reps.compute_weights(q_fct, phi_S)

            # get subset for sparse GP
            logger.debug('select samples for GP')
            gp_samples = self._SARS['S'].sample(self._params['gp']['num_sparse_states']).values

            # fit weighted GP to samples
            logger.info('fitting GP to policy')
            self.policy.train(self._SARS['S'].values, self._SARS['A'].values, weights, gp_samples)

            # plotting
            ax_aft_P = fig.add_subplot(gs[3, 0])
            ax_aft_P_cb = fig.add_subplot(gs[3, 1])
            plot_value_function(ax_aft_P, *self._compute_value_function_grid(state_action_features, theta),
                                cmap=cmap_gray)
            plot_objects(ax_aft_P, env=self._sampler.env)
            qv = plot_policy(ax_aft_P, *self._compute_policy_quivers(), cmap=cmap_plasma)
            fig.colorbar(qv, cax=ax_aft_P_cb)
            ax_aft_P.set_title('value function and policy, after reps, iteration {}'.format(self._it))

            # finalize plotting
            # fig.show()
            # save_plot_as_html(fig, path=self._log_path_rep, filename='plot_{:02d}.html'.format(self._it))
            show_plot_in_browser(fig, path=self._log_path_rep, filename='plot_{:02d}.html'.format(self._it),
                                 overwrite=True, save_only=self._no_gui)
            plt.close(fig)

        # save some stuff
        policy_file_name = os.path.join(self._log_path_rep, 'policy_{:02d}.pkl'.format(self._it))
        with open(policy_file_name, mode='w+b') as policy_file:
            pickle.dump(self.policy, policy_file)

        return_dict = dict(mean_sum_R=mean_sum_R, median_sum_R=median_sum_R, std_sum_R=std_sum_R,
                           max_sum_R=max_sum_R, min_sum_R=min_sum_R)

        gc.collect()

        return return_dict

    def reset(self, config: dict, rep: int) -> None:
        kernel_params = self._params['kernel']
        if kernel_params['type'] == 'mean':
            self.StateKernel = MeanStateKernel
            self.StateActionKernel = MeanStateActionKernel
        elif kernel_params['type'] == 'mean-cov':
            self.StateKernel = MeanCovStateKernel
            self.StateActionKernel = MeanCovStateActionKernel

        self._state_kernel = self.StateKernel(weight=kernel_params['weight'],
                                              bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                              bandwidth_light=kernel_params['bandwidth_factor_la'],
                                              num_processes=kernel_params['num_processes'])

        self._state_action_kernel = self.StateActionKernel(weight=kernel_params['weight'],
                                                           bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                                           bandwidth_light=kernel_params['bandwidth_factor_la'],
                                                           num_processes=kernel_params['num_processes'])

        self.lstd = LeastSquaresTemporalDifference()
        # self.lstd = LeastSquaresTemporalDifference()
        self.lstd.discount_factor = self._params['lstd']['discount_factor']

        self.ac_reps = ActorCriticReps()
        self.ac_reps.epsilon_action = self._params['reps']['epsilon']

        action_bounds = CircularGradientLight.action_space.low, CircularGradientLight.action_space.high
        self.policy = SparseGPPolicy(kernel=self._state_kernel, action_bounds=action_bounds)
        self.policy.gp_prior_variance = self._params['gp']['prior_variance']
        self.policy.gp_noise_variance = self._params['gp']['noise_variance']
        self.policy.gp_min_variance = self._params['gp']['min_variance']
        self.policy.gp_chol_regularizer = self._params['gp']['chol_regularizer']

        kb_columns_level = ['kb_{}'.format(i) for i in range(self._params['sampling']['num_kilobots'])]
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

        sampling_params = self._params['sampling']
        self._sampler = self.Sampler(num_episodes=sampling_params['num_episodes'],
                                     num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                     num_kilobots=sampling_params['num_kilobots'],
                                     column_index=self._SARS_columns,
                                     w_factor=sampling_params['w_factor'],
                                     policy=self.policy,
                                     num_workers=sampling_params['num_workers'],
                                     seed=self._seed)

    def _compute_value_function_grid(self, state_action_features, theta, steps_x=50, steps_y=25):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = -Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots+1])

        # get mean actions
        actions = self.policy.get_mean_action(states)

        value_function = state_action_features(states, actions).dot(theta).reshape((steps_y, steps_x))

        return value_function, x_range, y_range

    def _compute_policy_quivers(self, steps_x=50, steps_y=25):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = -Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots + 1])

        # get mean actions
        mean_actions, sigma_actions = self.policy.get_mean_action(states, return_sigma=True)
        # mean_actions /= np.linalg.norm(mean_actions, axis=1, keepdims=True)
        mean_actions = mean_actions.reshape((steps_y, steps_x, 2))
        sigma_actions = sigma_actions.reshape((steps_y, steps_x))

        actions = mean_actions, sigma_actions

        return actions, x_range, y_range
