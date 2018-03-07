import abc
import os
import gc

import logging
from typing import Generator

from cluster_work import ClusterWork, InvalidParameterArgument

from kb_learning.kernel import KilobotEnvKernel
from kb_learning.kernel import KilobotEnvKernelWithWeight, MeanEnvKernelWithWeight, MeanCovEnvKernelWithWeight
from kb_learning.kernel import EmbeddedSwarmDistance, MahaDist, MeanSwarmDist, MeanCovSwarmDist, PeriodicDist
from kb_learning.kernel import compute_median_bandwidth, select_reference_set_by_kernel_activation, \
    compute_mean_position_pandas, angle_from_swarm_mean
from kb_learning.reps.lstd import LeastSquaresTemporalDifference
from kb_learning.reps.ac_reps import ActorCriticReps
from kb_learning.reps.sparse_gp_policy import SparseGPPolicy

from kb_learning.envs.sampler import FixedWeightQuadEnvSampler, SampleWeightQuadEnvSampler, ComplexObjectEnvSampler, \
    GradientLightComplexObjectEnvSampler

import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from kb_learning.tools.plotting import plot_trajectories, plot_value_function, plot_policy, plot_objects, \
    plot_trajectory_reward_distribution, show_plot_as_pdf

import numpy as np
import pandas as pd

import pickle

logger = logging.getLogger('kb_learning')


class KilobotLearner(ClusterWork):
    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        raise NotImplementedError


class ACRepsLearner(KilobotLearner):
    _restore_supported = True
    _default_params = {
        'sampling': {
            'num_kilobots': 15,
            'w_factor': .0,
            'num_episodes': 100,
            'num_steps_per_episode': 125,
            'num_SARS_samples': 10000,
            'num_workers': None,
            'object_shape': 'quad',
            'object_width': .15,
            'object_height': .15,
        },
        'kernel': {
            'kb_dist': 'embedded',
            'v_dist': 'maha',
            'w_dist': 'maha',
            'bandwidth_factor_kb': .8,
            'bandwidth_factor_light': .8,
            'bandwidth_factor_light_action': .8,
            'bandwidth_factor_weight': 1.,
            'weight': .5,
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
            'prior_variance': .01 ** 2,
            'noise_variance': 2e-5,
            'min_variance': 1e-8,
            'chol_regularizer': 1e-9,
            'num_sparse_states': 1000,
        }
    }

    def __init__(self):
        super().__init__()

        self._state_preprocessor = None
        self._state_kernel = None
        self._state_action_kernel = None

        self._lstd = None
        self._ac_reps = None
        self._policy = None

        self._SARS = None

        self._sampler = None

        self.kilobots_columns = None
        self.light_columns = None
        self.state_columns = None
        self.action_columns = None
        self.reward_columns = None
        self.next_state_columns = None
        self._SARS_columns = None
        self.extra_dim_columns = None

        self._light_dimensions = 2
        self._action_dimensions = 2

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        sampling_params = self._params['sampling']

        logger.info('sampling environment')
        self._sampler.seed = rep * 100 + n
        it_sars, it_info = self._sampler(self._policy)
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
        # _extended_SARS.reset_index(drop=True)

        # compute kernel parameters
        logger.debug('computing kernel bandwidths.')
        bandwidth_kb = compute_median_bandwidth(_extended_SARS[self.kilobots_columns], sample_size=500,
                                                preprocessor=self._state_preprocessor)
        if self.extra_dim_columns is not None:
            bandwidth_ed = compute_median_bandwidth(_extended_SARS[self.extra_dim_columns], sample_size=500)
        else:
            bandwidth_ed = []
        bandwidth_a = compute_median_bandwidth(_extended_SARS['A'], sample_size=500)

        self._state_kernel.set_params(bandwidth=np.r_[bandwidth_kb, bandwidth_ed])
        self._state_action_kernel.set_params(bandwidth=np.r_[bandwidth_kb, bandwidth_ed, bandwidth_a])

        logger.debug('selecting SARS samples.')
        if _extended_SARS.shape[0] <= sampling_params['num_SARS_samples']:
            self._SARS = _extended_SARS
        else:
            self._SARS = _extended_SARS.sample(sampling_params['num_SARS_samples'])
        self._SARS.reset_index(drop=True, inplace=True)
        del _extended_SARS

        # compute feature matrices
        logger.debug('selecting lstd samples.')
        lstd_reference = select_reference_set_by_kernel_activation(data=self._SARS[['S', 'A']],
                                                                   size=self._params['lstd']['num_features'],
                                                                   kernel_function=self._state_action_kernel,
                                                                   batch_size=10)
        # lstd_samples = self._SARS.loc[lstd_reference]
        lstd_state_samples = self._SARS.loc[lstd_reference]['S'].values
        lstd_state_action_samples = self._SARS.loc[lstd_reference][['S', 'A']].values

        # lstd_samples = self._SARS[['S', 'A']].sample(self._params['lstd']['num_features'])

        # def state_features(state):
        #     if state.ndim == 1:
        #         state = state.reshape((1, -1))
        #     return self._state_kernel(state, lstd_state_samples)
        #
        # def state_action_features(state, action):
        #     if state.ndim == 1:
        #         state = state.reshape((1, -1))
        #     if action.ndim == 1:
        #         action = action.reshape((1, -1))
        #     return self._state_action_kernel(np.c_[state, action], lstd_state_action_samples)

        for i in range(self._params['learn_iterations']):
            # compute features
            logger.info('compute state-action features')
            phi_SA = self._state_action_kernel(self._SARS[['S', 'A']].values, lstd_state_action_samples)
            phi_SA_next = self._state_action_kernel(
                np.c_[self._SARS['S_'].values, self._policy.get_mean_action(self._SARS['S_'].values)],
                lstd_state_action_samples)

            # learn theta (parameters of Q-function) using lstd
            logger.info('learning theta [LSTD]')
            lstd = LeastSquaresTemporalDifference()
            lstd.discount_factor = self._params['lstd']['discount_factor']
            theta = lstd.learn_q_function(phi_SA, phi_SA_next, rewards=self._SARS[['R']].values)

            # plotting

            # compute q-function
            logger.debug('compute q-function')
            q_fct = phi_SA.dot(theta)

            # compute state features
            logger.info('compute state features')
            phi_S = self._state_kernel(self._SARS['S'].values, lstd_state_samples)

            # compute sample weights using AC-REPS
            logger.info('learning weights [AC-REPS]')
            ac_reps = ActorCriticReps()
            ac_reps.epsilon_action = self._params['reps']['epsilon']
            weights = ac_reps.compute_weights(q_fct, phi_S)

            # get subset for sparse GP
            logger.debug('select samples for GP')
            gp_reference = select_reference_set_by_kernel_activation(data=self._SARS[['S']],
                                                                     size=self._params['lstd']['num_features'],
                                                                     kernel_function=self._state_kernel,
                                                                     batch_size=10)
            gp_samples = self._SARS['S'].loc[gp_reference].values
            # gp_samples = self._SARS['S'].sample(self._params['gp']['num_sparse_states']).values

            # fit weighted GP to samples
            logger.info('fitting GP to policy')
            self._policy = self._init_policy((self._sampler.env.action_space.low, self._sampler.env.action_space.high))
            self._policy.train(self._SARS['S'].values, self._SARS['A'].values, weights, gp_samples)

            # plotting
            self._plot_iteration_results(it_sars,
                                         lambda s, a: self._state_action_kernel(np.c_[s, a], lstd_state_action_samples),
                                         theta)

            del phi_SA, phi_SA_next

        return_dict = dict(mean_sum_R=mean_sum_R, median_sum_R=median_sum_R, std_sum_R=std_sum_R,
                           max_sum_R=max_sum_R, min_sum_R=min_sum_R)

        del lstd_state_samples, lstd_state_action_samples
        del it_sars
        gc.collect()

        return return_dict

    def reset(self, config: dict, rep: int) -> None:
        self._init_kernels()

        self._init_SARS()

        self._sampler = self._init_sampler()

        self._policy = self._init_policy((self._sampler.env.action_space.low, self._sampler.env.action_space.high))

    def _init_kernels(self):
        kernel_params = self._params['kernel']
        if kernel_params['kb_dist'] in ['kilobot', 'embedded']:
            kb_dist_class = EmbeddedSwarmDistance
        elif kernel_params['kb_dist'] == 'mean':
            from kb_learning.kernel import compute_mean_position
            self._state_preprocessor = compute_mean_position
            kb_dist_class = MeanSwarmDist
        elif kernel_params['kb_dist'] == 'mean-cov':
            from kb_learning.kernel import compute_mean_and_cov_position
            self._state_preprocessor = compute_mean_and_cov_position
            kb_dist_class = MeanCovSwarmDist
        else:
            raise InvalidParameterArgument

        if kernel_params['v_dist'] == 'maha':
            v_dist_class = MahaDist
        elif kernel_params['v_dist'] == 'periodic':
            v_dist_class = PeriodicDist
        else:
            raise InvalidParameterArgument

        self._state_kernel = KilobotEnvKernel(weight=kernel_params['weight'],
                                              bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                              bandwidth_factor_v=kernel_params['bandwidth_factor_light'],
                                              v_dims=self._light_dimensions,
                                              kb_dist_class=kb_dist_class,
                                              v_dist_class=v_dist_class)
        self._state_action_kernel = KilobotEnvKernel(weight=kernel_params['weight'],
                                                     bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                                     bandwidth_factor_v=kernel_params['bandwidth_factor_light_action'],
                                                     v_dims=self._light_dimensions + self._action_dimensions,
                                                     kb_dist_class=kb_dist_class,
                                                     v_dist_class=v_dist_class)

    def _init_policy(self, action_bounds):
        policy = SparseGPPolicy(kernel=self._state_kernel)
        policy.gp_prior_variance = self._params['gp']['prior_variance']
        policy.gp_noise_variance = self._params['gp']['noise_variance']
        policy.gp_min_variance = self._params['gp']['min_variance']
        policy.gp_chol_regularizer = self._params['gp']['chol_regularizer']
        policy.action_bounds = action_bounds

        return policy

    def _init_SARS(self):
        kb_columns_level = ['kb_{}'.format(i) for i in range(self._params['sampling']['num_kilobots'])]
        self.kilobots_columns = pd.MultiIndex.from_product([['S'], kb_columns_level, ['x', 'y']])
        self.light_columns = pd.MultiIndex.from_product([['S'], ['light'], ['x', 'y']])
        self.state_columns = self.kilobots_columns.append(self.light_columns)
        self.action_columns = pd.MultiIndex.from_product([['A'], ['x', 'y'], ['']])
        self.reward_columns = pd.MultiIndex.from_arrays([['R'], [''], ['']])
        self.next_state_columns = self.state_columns.copy()
        self.next_state_columns.set_levels(['S_'], 0, inplace=True)
        self.extra_dim_columns = self.light_columns

        self._SARS_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
            self.next_state_columns)

        self._SARS = pd.DataFrame(columns=self._SARS_columns)

    def _init_sampler(self):
        sampling_params = self._params['sampling']
        sampler = FixedWeightQuadEnvSampler(num_episodes=sampling_params['num_episodes'],
                                            num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                            num_kilobots=sampling_params['num_kilobots'],
                                            column_index=self._SARS_columns,
                                            w_factor=sampling_params['w_factor'],
                                            num_workers=sampling_params['num_workers'],
                                            seed=self._seed, mp_context=self._MP_CONTEXT)
        return sampler

    def finalize(self):
        pass

    def save_state(self, config: dict, rep: int, n: int) -> None:
        # save policy
        logger.info('pickling policy...')
        policy_file_name = os.path.join(self._log_path_rep, 'policy_{:02d}.pkl'.format(self._it))
        with open(policy_file_name, mode='w+b') as policy_file:
            pickle.dump(self._policy, policy_file)

        # save SARS data
        logger.info('pickling SARS...')
        SARS_file_name = os.path.join(self._log_path_rep, 'last_SARS.pkl.gz')
        self._SARS.to_pickle(SARS_file_name, compression='gzip')

    def restore_state(self, config: dict, rep: int, n: int) -> bool:
        # restore policy
        policy_file_name = os.path.join(self._log_path_rep, 'policy_{:02d}.pkl'.format(n - 1))
        with open(policy_file_name, mode='r+b') as policy_file:
            self._policy = pickle.load(policy_file)
        self._sampler.policy = self._policy

        # restore SARS data
        SARS_file_name = os.path.join(self._log_path_rep, 'last_SARS.pkl.gz')
        if not os.path.exists(SARS_file_name):
            return False
        self._SARS = pd.read_pickle(SARS_file_name, compression='gzip')
        return True

    @classmethod
    def plot_results(cls, results_config: Generator):
        fig = plt.figure()
        axes = fig.add_subplot(111)

        for config, results in results_config:
            mean_sum_R = results['mean_sum_R']

            mean = mean_sum_R.groupby(level=1).mean()
            std = mean_sum_R.groupby(level=1).std()

            axes.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=.5)
            axes.plot(mean.index, mean, label=config['name'])

        axes.legend()
        plt.show(block=True)
        # fig.show()

    def _plot_iteration_results(self, it_sars, state_action_features, theta):
        # setup figure
        fig = plt.figure(figsize=(10, 20))
        gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[20, 1], height_ratios=[1, 3, 3, 3])

        # reward plot
        ax_R = fig.add_subplot(gs[0, :])
        plot_trajectory_reward_distribution(ax_R, it_sars['R'])

        # value function plot
        ax_bef_V = fig.add_subplot(gs[1, 0])
        ax_bef_V_cb = fig.add_subplot(gs[1, 1])
        cmap_plasma = cm.get_cmap('plasma')
        cmap_gray = cm.get_cmap('gray')
        V = self._compute_value_function_grid(state_action_features, theta)
        im = plot_value_function(ax_bef_V, *V, cmap=cmap_plasma)
        plot_objects(ax_bef_V, env=self._sampler.env, alpha=.3, fill=True)
        ax_bef_V.set_title('value function, before reps, iteration {}'.format(self._it))
        fig.colorbar(im, cax=ax_bef_V_cb)

        # trajectories plot
        ax_bef_T = fig.add_subplot(gs[2, 0])
        ax_bef_T_cb = fig.add_subplot(gs[2, 1])
        ax_bef_T.set_title('trajectories, iteration {}'.format(self._it))
        plot_value_function(ax_bef_T, *V, cmap=cmap_gray)
        tr = plot_trajectories(ax_bef_T, it_sars['S']['light'])
        plot_objects(ax_bef_T, env=self._sampler.env)
        fig.colorbar(tr[0], cax=ax_bef_T_cb)

        # new policy plot
        ax_aft_P = fig.add_subplot(gs[3, 0])
        ax_aft_P_cb = fig.add_subplot(gs[3, 1])
        plot_value_function(ax_aft_P, *self._compute_value_function_grid(state_action_features, theta),
                            cmap=cmap_gray)
        qv = plot_policy(ax_aft_P, *self._compute_policy_quivers(), cmap=cmap_plasma)
        plot_objects(ax_aft_P, env=self._sampler.env)
        fig.colorbar(qv, cax=ax_aft_P_cb)
        ax_aft_P.set_title('policy, after reps, iteration {}'.format(self._it))

        # save and show plot
        show_plot_as_pdf(fig, path=self._log_path_rep, filename='plot_{:02d}.pdf'.format(self._it),
                         overwrite=True, save_only=self._no_gui)
        plt.close(fig)

    def _compute_value_function_grid(self, state_action_features, theta, steps_x=40, steps_y=40):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = -Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots + 1])

        # get mean actions
        actions = self._policy.get_mean_action(states)

        value_function = state_action_features(states, actions).dot(theta).reshape((steps_y, steps_x))

        return value_function, x_range, y_range

    def _compute_policy_quivers(self, steps_x=40, steps_y=40):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots + 1])

        # get mean actions
        mean_actions, sigma_actions = self._policy.get_mean_sigma_action(states)
        mean_actions = mean_actions.reshape((steps_y, steps_x, mean_actions.shape[1]))
        sigma_actions = sigma_actions.reshape((steps_y, steps_x))

        actions = mean_actions, sigma_actions

        return actions, x_range, y_range


class SampleWeightACRepsLearner(ACRepsLearner):
    def _init_kernels(self):
        kernel_params = self._params['kernel']
        if kernel_params['type'] == 'mean':
            from kb_learning.kernel import compute_mean_position
            self._state_preprocessor = compute_mean_position
            EnvKernel = MeanEnvKernelWithWeight
        elif kernel_params['type'] == 'mean-cov':
            from kb_learning.kernel import compute_mean_and_cov_position
            self._state_preprocessor = compute_mean_and_cov_position
            EnvKernel = MeanCovEnvKernelWithWeight
        else:
            EnvKernel = KilobotEnvKernelWithWeight

        self._state_kernel = EnvKernel(weight=kernel_params['weight'],
                                       bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                       bandwidth_factor_ed=kernel_params['bandwidth_factor_extra_dims'],
                                       bandwidth_factor_weigth=kernel_params['bandwidth_factor_weight'],
                                       weight_dim=-1,
                                       extra_dims=self._extra_kernel_dimensions,
                                       num_processes=kernel_params['num_processes'])
        self._state_action_kernel = EnvKernel(weight=kernel_params['weight'],
                                              bandwidth_factor_kb=kernel_params['bandwidth_factor_kb'],
                                              bandwidth_factor_ed=kernel_params['bandwidth_factor_extra_dims_action'],
                                              bandwidth_factor_weigth=kernel_params['bandwidth_factor_weight'],
                                              weight_dim=-3,
                                              extra_dims=self._extra_kernel_dimensions + self._action_dimensions,
                                              num_processes=kernel_params['num_processes'])

    def _init_SARS(self):
        super()._init_SARS()

        self.weight_columns = pd.MultiIndex.from_product([['S'], ['weight'], ['']])
        self.state_columns = self.kilobots_columns.append(self.light_columns).append(self.weight_columns)
        self.next_state_columns = self.state_columns.copy()
        self.next_state_columns.set_levels(['S_'], 0, inplace=True)
        self.extra_dim_columns = self.light_columns.append(self.weight_columns)

        self._SARS_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
            self.next_state_columns)

        self._SARS = pd.DataFrame(columns=self._SARS_columns)

    def _init_sampler(self):
        sampling_params = self._params['sampling']
        sampler = SampleWeightQuadEnvSampler(num_episodes=sampling_params['num_episodes'],
                                             num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                             num_kilobots=sampling_params['num_kilobots'],
                                             column_index=self._SARS_columns,
                                             w_factor=sampling_params['w_factor'],
                                             num_workers=sampling_params['num_workers'],
                                             seed=self._seed, mp_context=self._MP_CONTEXT)

        return sampler

    def _plot_iteration_results(self, it_sars, state_action_features, theta):
        # setup figure
        fig = plt.figure(figsize=(11, 20))
        gs = gridspec.GridSpec(nrows=4, ncols=4, width_ratios=[7, 7, 7, 1], height_ratios=[1, 3, 3, 3])
        w0 = .166
        w1 = .5
        w2 = .833

        # reward plot
        ax_R = fig.add_subplot(gs[0, :])
        plot_trajectory_reward_distribution(ax_R, it_sars['R'])

        # value function plot
        cmap_plasma = cm.get_cmap('plasma')
        ax_V_w0 = fig.add_subplot(gs[1, 0])
        ax_V_w1 = fig.add_subplot(gs[1, 1])
        ax_V_w2 = fig.add_subplot(gs[1, 2])
        ax_V_cb = fig.add_subplot(gs[1, 3])
        ax_V_w0.set_title('value function, w = {}'.format(w0))
        ax_V_w1.set_title('w = {}'.format(w1))
        ax_V_w2.set_title('w = {}, iteration {}'.format(w2, self._it))
        V_w0 = self._compute_value_function_grid(state_action_features, theta, weight=w0)
        plot_value_function(ax_V_w0, *V_w0, cmap=cmap_plasma)
        plot_objects(ax_V_w0, env=self._sampler.env, fill=False)
        V_w1 = self._compute_value_function_grid(state_action_features, theta, weight=w1)
        plot_value_function(ax_V_w1, *V_w1, cmap=cmap_plasma)
        plot_objects(ax_V_w1, env=self._sampler.env, fill=False)
        V_w2 = self._compute_value_function_grid(state_action_features, theta, weight=w2)
        im = plot_value_function(ax_V_w2, *V_w2, cmap=cmap_plasma)
        plot_objects(ax_V_w2, env=self._sampler.env, fill=False)
        fig.colorbar(im, cax=ax_V_cb)

        # trajectories plot
        cmap_gray = cm.get_cmap('gray')
        ax_T_w0 = fig.add_subplot(gs[2, 0])
        ax_T_w1 = fig.add_subplot(gs[2, 1])
        ax_T_w2 = fig.add_subplot(gs[2, 2])
        ax_T_cb = fig.add_subplot(gs[2, 3])
        ax_T_w0.set_title('trajectories, w <= 0.33')
        ax_T_w1.set_title('0.33 < w <= 0.66')
        ax_T_w2.set_title('0.66 < w, iteration {}'.format(self._it))
        plot_value_function(ax_T_w0, *V_w0, cmap=cmap_gray)
        plot_objects(ax_T_w0, env=self._sampler.env)
        small_w_index = it_sars[('S', 'weight')] <= .33
        medium_w_index = (it_sars[('S', 'weight')] <= .66) & ~small_w_index
        large_w_index = ~(small_w_index | medium_w_index)
        plot_trajectories(ax_T_w0, it_sars[('S', 'light')].loc[small_w_index])
        plot_value_function(ax_T_w1, *V_w1, cmap=cmap_gray)
        plot_objects(ax_T_w1, env=self._sampler.env)
        plot_trajectories(ax_T_w1, it_sars[('S', 'light')].loc[medium_w_index])
        plot_value_function(ax_T_w2, *V_w2, cmap=cmap_gray)
        plot_objects(ax_T_w2, env=self._sampler.env)
        tr = plot_trajectories(ax_T_w2, it_sars[('S', 'light')].loc[large_w_index])
        fig.colorbar(tr[0], cax=ax_T_cb)

        # new policy plot
        ax_P_w0 = fig.add_subplot(gs[3, 0])
        ax_P_w1 = fig.add_subplot(gs[3, 1])
        ax_P_w2 = fig.add_subplot(gs[3, 2])
        ax_P_cb = fig.add_subplot(gs[3, 3])
        ax_P_w0.set_title('learned policies, w = {}'.format(w0))
        ax_P_w1.set_title('w = {}'.format(w1))
        ax_P_w2.set_title('w = {}, iteration {}'.format(w2, self._it))
        plot_value_function(ax_P_w0, *V_w0, cmap=cmap_gray)
        plot_objects(ax_P_w0, env=self._sampler.env)
        plot_policy(ax_P_w0, *self._compute_policy_quivers(weight=w0), cmap=cmap_plasma)
        plot_value_function(ax_P_w1, *V_w1, cmap=cmap_gray)
        plot_objects(ax_P_w1, env=self._sampler.env)
        plot_policy(ax_P_w1, *self._compute_policy_quivers(weight=w1), cmap=cmap_plasma)
        plot_value_function(ax_P_w2, *V_w2, cmap=cmap_gray)
        plot_objects(ax_P_w2, env=self._sampler.env)
        qv = plot_policy(ax_P_w2, *self._compute_policy_quivers(weight=w2), cmap=cmap_plasma)
        fig.colorbar(qv, cax=ax_P_cb)

        # save and show plot
        show_plot_as_pdf(fig, path=self._log_path_rep, filename='plot_{:02d}.pdf'.format(self._it),
                         overwrite=True, save_only=self._no_gui)
        plt.close(fig)

    def _compute_value_function_grid(self, state_action_features, theta, weight, steps_x=40, steps_y=40):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = -Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots + 1])
        states = np.c_[states, np.ones((states.shape[0], 1)) * weight]

        # get mean actions
        actions = self._policy.get_mean_action(states)

        value_function = state_action_features(states, actions).dot(theta).reshape((steps_y, steps_x))

        return value_function, x_range, y_range

    def _compute_policy_quivers(self, weight, steps_x=20, steps_y=20):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots + 1])
        states = np.c_[states, np.ones((states.shape[0], 1)) * weight]

        # get mean actions
        mean_actions, sigma_actions = self._policy.get_mean_sigma_action(states)
        # mean_actions /= np.linalg.norm(mean_actions, axis=1, keepdims=True)
        mean_actions = mean_actions.reshape((steps_y, steps_x, 1))
        sigma_actions = sigma_actions.reshape((steps_y, steps_x))

        actions = mean_actions, sigma_actions

        return actions, x_range, y_range


class ComplexObjectACRepsLearner(ACRepsLearner):
    def _init_sampler(self):
        sampling_params = self._params['sampling']
        return ComplexObjectEnvSampler(object_shape=sampling_params['object_shape'],
                                       object_width=sampling_params['object_width'],
                                       object_height=sampling_params['object_height'],
                                       num_episodes=sampling_params['num_episodes'],
                                       num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                       num_kilobots=sampling_params['num_kilobots'],
                                       column_index=self._SARS_columns,
                                       w_factor=sampling_params['w_factor'],
                                       num_workers=sampling_params['num_workers'],
                                       seed=self._seed, mp_context=self._MP_CONTEXT)


class GradientLightComplexObjectACRepsLearner(ACRepsLearner):
    def __init__(self):
        super().__init__()

        self._light_dimensions = 0
        self._action_dimensions = 1

    def _init_policy(self, action_bounds):
        policy = super()._init_policy(action_bounds)
        policy.gp_prior_mean = angle_from_swarm_mean

        return policy

    def _init_SARS(self):
        kb_columns_level = ['kb_{}'.format(i) for i in range(self._params['sampling']['num_kilobots'])]
        self.kilobots_columns = pd.MultiIndex.from_product([['S'], kb_columns_level, ['x', 'y']])
        self.state_columns = self.kilobots_columns
        self.action_columns = pd.MultiIndex.from_product([['A'], [''], ['']])
        self.reward_columns = pd.MultiIndex.from_arrays([['R'], [''], ['']])
        self.next_state_columns = self.state_columns.copy()
        self.next_state_columns.set_levels(['S_'], 0, inplace=True)

        self._SARS_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
            self.next_state_columns)

        self._SARS = pd.DataFrame(columns=self._SARS_columns)

    def _init_sampler(self):
        sampling_params = self._params['sampling']
        return GradientLightComplexObjectEnvSampler(object_shape=sampling_params['object_shape'],
                                                    object_width=sampling_params['object_width'],
                                                    object_height=sampling_params['object_height'],
                                                    num_episodes=sampling_params['num_episodes'],
                                                    num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                                    num_kilobots=sampling_params['num_kilobots'],
                                                    column_index=self._SARS_columns,
                                                    w_factor=sampling_params['w_factor'],
                                                    num_workers=sampling_params['num_workers'],
                                                    seed=self._seed, mp_context=self._MP_CONTEXT)

    def _plot_iteration_results(self, it_sars, state_action_features, theta):
        # setup figure
        fig = plt.figure(figsize=(10, 20))
        gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[20, 1], height_ratios=[1, 3, 3, 3])

        # reward plot
        ax_R = fig.add_subplot(gs[0, :])
        plot_trajectory_reward_distribution(ax_R, it_sars['R'])

        # value function plot
        ax_bef_V = fig.add_subplot(gs[1, 0])
        ax_bef_V_cb = fig.add_subplot(gs[1, 1])
        cmap_plasma = cm.get_cmap('plasma')
        cmap_gray = cm.get_cmap('gray')
        V = self._compute_value_function_grid(state_action_features, theta)
        im = plot_value_function(ax_bef_V, *V, cmap=cmap_plasma)
        plot_objects(ax_bef_V, env=self._sampler.env, alpha=.3, fill=True)
        ax_bef_V.set_title('value function, before reps, iteration {}'.format(self._it))
        fig.colorbar(im, cax=ax_bef_V_cb)

        # trajectories plot
        ax_bef_T = fig.add_subplot(gs[2, 0])
        ax_bef_T_cb = fig.add_subplot(gs[2, 1])
        ax_bef_T.set_title('trajectories, iteration {}'.format(self._it))
        plot_value_function(ax_bef_T, *V, cmap=cmap_gray)
        tr = plot_trajectories(ax_bef_T, compute_mean_position_pandas(it_sars['S']))
        plot_objects(ax_bef_T, env=self._sampler.env)
        fig.colorbar(tr[0], cax=ax_bef_T_cb)

        # new policy plot
        ax_aft_P = fig.add_subplot(gs[3, 0])
        ax_aft_P_cb = fig.add_subplot(gs[3, 1])
        plot_value_function(ax_aft_P, *self._compute_value_function_grid(state_action_features, theta),
                            cmap=cmap_gray)
        qv = plot_policy(ax_aft_P, *self._compute_policy_quivers(), cmap=cmap_plasma)
        plot_objects(ax_aft_P, env=self._sampler.env)
        fig.colorbar(qv, cax=ax_aft_P_cb)
        ax_aft_P.set_title('policy, after reps, iteration {}'.format(self._it))

        # save and show plot
        show_plot_as_pdf(fig, path=self._log_path_rep, filename='plot_{:02d}.pdf'.format(self._it),
                         overwrite=True, save_only=self._no_gui)
        # plt.show(block=True)

        plt.close(fig)

    def _compute_value_function_grid(self, state_action_features, theta, steps_x=40, steps_y=40):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = -Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots])

        # get mean actions
        actions = self._policy.get_mean_action(states)

        value_function = state_action_features(states, actions).dot(theta).reshape((steps_y, steps_x))

        return value_function, x_range, y_range

    def _compute_policy_quivers(self, steps_x=40, steps_y=40):
        x_range = self._sampler.env.world_x_range
        y_range = self._sampler.env.world_y_range
        [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
        X = X.flatten()
        Y = Y.flatten()

        # kilobots at light position
        states = np.tile(np.c_[X, Y], [1, self._sampler.num_kilobots])

        # get mean actions
        mean_actions, sigma_actions = self._policy.get_mean_sigma_action(states)
        # mean_actions /= np.linalg.norm(mean_actions, axis=1, keepdims=True)
        mean_actions = mean_actions.reshape((steps_y, steps_x, mean_actions.shape[1]))
        sigma_actions = sigma_actions.reshape((steps_y, steps_x))

        actions = mean_actions, sigma_actions

        return actions, x_range, y_range


