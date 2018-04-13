import abc
import os
import gc

import logging

from typing import Generator

from cluster_work import ClusterWork, InvalidParameterArgument

from kb_learning.kernel import KilobotEnvKernel
from kb_learning.kernel import EmbeddedSwarmDistance, MahaDist, MeanSwarmDist, MeanCovSwarmDist, PeriodicDist
from kb_learning.kernel import compute_median_bandwidth, compute_median_bandwidth_kilobots, \
    select_reference_set_by_kernel_activation, compute_mean_position_pandas, angle_from_swarm_mean, step_towards_center
from kb_learning.ac_reps.lstd import LeastSquaresTemporalDifference
from kb_learning.ac_reps.reps import ActorCriticReps
from kb_learning.ac_reps.spwgp import SparseWeightedGP
from kb_learning.ac_reps.gpy_spwgp import SparseWeightedGPyWrapper

from kb_learning.envs.sampler import ParallelSARSSampler, SARSSampler
from kb_learning.envs import register_object_env

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
            'save_trajectories': True,
            'light_type': 'circular',
            'light_radius': .3
        },
        'kernel': {
            'kb_dist': 'embedded',
            'l_dist': 'maha',
            'a_dist': 'maha',
            'w_dist': 'maha',
            'bandwidth_factor_kb': .3,
            'bandwidth_factor_light': .55,
            'bandwidth_factor_action': .8,
            'bandwidth_factor_weight': .3,
            'rho': .5,
            'variance': 1.,
        },
        'learn_iterations': 1,
        'lstd': {
            'discount_factor': .99,
            'num_features': 1000,
            'num_policy_samples': 1,
        },
        'ac_reps': {
            'epsilon': .3
        },
        'gp': {
            'prior_variance': 5e-5,
            'noise_variance': 2e-5,
            'num_sparse_states': 1000,
            'kb_dist': 'embedded',
            'l_dist': 'maha',
            'w_dist': 'maha',
            'bandwidth_factor_kb': .3,
            'bandwidth_factor_light': .55,
            'bandwidth_factor_weight': .3,
            'rho': .5,
        },
        'eval': {
            'num_episodes': 50,
            'num_steps_per_episode': 100,
            'save_trajectories': True
        }
    }

    def __init__(self):
        super().__init__()

        self.state_preprocessor = None
        self.state_kernel = None
        self.state_action_kernel = None

        self.policy = None

        self.sars = None

        self.theta = None
        self.it_sars = None
        self.eval_sars = None
        self.it_info = None
        self.eval_info = None

        self.lstd_samples = None

        self.sampler = None

        self.kilobots_columns = None
        self.light_columns = None
        self.object_columns = None
        self.weight_columns = None
        self.action_columns = None
        self.reward_columns = None
        self.state_columns = None
        self.next_state_columns = None
        self.sars_columns = None
        self.state_object_columns = None

        self._light_dimensions = 2
        self._action_dimensions = 2

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        sampling_params = self._params['sampling']
        np.random.seed(self._seed)

        logger.info('sampling environment')
        self.sampler.seed = self._seed + 123
        self.it_sars, self.it_info = self.sampler(self.policy)

        sum_R = self.it_sars['R'].groupby(level=0).sum()

        mean_sum_R = sum_R.mean()
        sum_sum_R = sum_R.sum()
        std_sum_R = sum_R.std()
        max_sum_R = sum_R.max()
        min_sum_R = sum_R.min()

        logger.info('statistics on sum R -- mean: {:.6f} sum: {:.6f} std: {:.6f} max: {:.6f} min: {:.6f}'.format(
            mean_sum_R, sum_sum_R, std_sum_R, max_sum_R, min_sum_R))

        # add samples to data set and select subset
        _extended_sars = self.sars.append(self.it_sars, ignore_index=True)
        # _extended_sars.reset_index(drop=True)

        if n == 0:
            # compute kernel parameters
            logger.debug('computing kernel bandwidths.')
            if self.state_preprocessor:
                bandwidth_kb = compute_median_bandwidth(_extended_sars[self.kilobots_columns], sample_size=500,
                                                        preprocessor=self.state_preprocessor)
            else:
                bandwidth_kb = compute_median_bandwidth_kilobots(_extended_sars[self.kilobots_columns], sample_size=500)
            self.state_kernel.kilobots_bandwidth = bandwidth_kb * self._params['kernel']['bandwidth_factor_kb']
            self.state_action_kernel.kilobots_bandwidth = bandwidth_kb * self._params['kernel']['bandwidth_factor_kb']
            self.policy.kernel.kilobots_bandwidth = bandwidth_kb * self._params['gp']['bandwidth_factor_kb']

            if self.light_columns is not None:
                bandwidth_l = compute_median_bandwidth(_extended_sars[self.light_columns], sample_size=500)
                self.state_kernel.light_bandwidth = bandwidth_l * self._params['kernel']['bandwidth_factor_light']
                self.state_action_kernel.light_bandwidth = bandwidth_l * self._params['kernel']['bandwidth_factor_light']
                self.policy.kernel.light_bandwidth = bandwidth_l * self._params['gp']['bandwidth_factor_light']

            if self.weight_columns is not None:
                bandwidth_w = compute_median_bandwidth(_extended_sars[self.weight_columns], sample_size=500)
                self.state_kernel.weight_bandwidth = bandwidth_w * self._params['kernel']['bandwidth_factor_weight']
                self.state_action_kernel.weight_bandwidth = bandwidth_w * self._params['kernel']['bandwidth_factor_weight']
                self.policy.kernel.weight_bandwidth = bandwidth_w * self._params['gp']['bandwidth_factor_weight']

            bandwidth_a = compute_median_bandwidth(_extended_sars['A'], sample_size=500)
            bandwidth_a *= self._params['kernel']['bandwidth_factor_action']
            self.state_action_kernel.action_bandwidth = bandwidth_a

        logger.debug('selecting SARS samples.')
        if _extended_sars.shape[0] <= sampling_params['num_SARS_samples']:
            self.sars = _extended_sars
        else:
            self.sars = _extended_sars.sample(sampling_params['num_SARS_samples'])
        self.sars.reset_index(drop=True, inplace=True)
        del _extended_sars

        # compute feature matrices
        logger.debug('selecting lstd samples.')
        # lstd_reference = select_reference_set_by_kernel_activation(data=self.sars[['S', 'A']],
        #                                                            size=self._params['lstd']['num_features'],
        #                                                            kernel_function=self.state_kernel,
        #                                                            batch_size=10)
        #
        # self.lstd_samples = self.sars.loc[lstd_reference][['S', 'A']]
        self.lstd_samples = self.sars[['S', 'A']].sample(self._params['lstd']['num_features'])

        def state_features(state):
            if state.ndim == 1:
                state = state.reshape((1, -1))
            return self.state_kernel(state, self.lstd_samples['S'].values)

        def state_action_features(state, action):
            if state.ndim == 1:
                state = state.reshape((1, -1))
            if action.ndim == 1:
                action = action.reshape((1, -1))
            return self.state_action_kernel(np.c_[state, action], self.lstd_samples.values)

        for i in range(self._params['learn_iterations']):
            # compute features
            logger.info('compute state-action features')
            phi_SA = state_action_features(self.sars['S'].values, self.sars[['A']].values)
            next_actions = np.array([self.policy(self.sars['S_'].values) for _ in range(5)]).mean(axis=0)
            # next_actions = self.policy.get_mean(self.sars['S_'].values)
            phi_SA_next = state_action_features(self.sars['S_'].values, next_actions)

            # learn theta (parameters of Q-function) using lstd
            logger.info('learning theta [LSTD]')
            lstd = LeastSquaresTemporalDifference()
            lstd.discount_factor = self._params['lstd']['discount_factor']
            self.theta = lstd.learn_q_function(phi_SA, phi_SA_next, rewards=self.sars[['R']].values)

            # compute q-function
            logger.debug('compute q-function')
            q_fct = phi_SA.dot(self.theta)

            # compute state features
            logger.info('compute state features')
            phi_S = state_features(self.sars['S'].values)

            # compute sample weights using AC-REPS
            logger.info('learning weights [AC-REPS]')
            ac_reps = ActorCriticReps()
            ac_reps.epsilon_action = self._params['ac_reps']['epsilon']
            weights = ac_reps.compute_weights(q_fct, phi_S)

            # get subset for sparse GP
            logger.debug('select samples for GP')
            # weights_sort_index = np.argsort(weights)
            # gp_reference = pd.Index(weights_sort_index[-self._params['gp']['num_sparse_states']:])
            gp_reference = pd.Index(np.random.choice(self.sars.index.values,
                                                     size=self._params['gp']['num_sparse_states'],
                                                     replace=False, p=weights))
            # gp_reference = select_reference_set_by_kernel_activation(data=self._SARS[['S']],
            #                                                          size=self._params['gp']['num_sparse_states'],
            #                                                          kernel_function=self._state_kernel,
            #                                                          batch_size=10, start_from=lstd_reference)
            gp_samples = self.sars['S'].loc[gp_reference].values
            # gp_samples = self._SARS['S'].sample(self._params['gp']['num_sparse_states']).values

            # fit weighted GP to samples
            logger.info('fitting GP to policy')
            # self.policy = self._init_policy(self.state_kernel,
            #                                 (self.sampler.env.action_space.low, self.sampler.env.action_space.high))
            self.policy.train(inputs=self.sars['S'].values, outputs=self.sars['A'].values, weights=weights,
                              sparse_inputs=gp_samples, optimize=True)

            # evaluate policy
            logger.info('evaluating policy')
            self.sampler.seed = 5555
            self.policy.eval_mode = True
            self.eval_sars, self.eval_info = self.sampler(self.policy,
                                                          num_episodes=self._params['eval']['num_episodes'],
                                                          num_steps_per_episode=self._params['eval'][
                                                              'num_steps_per_episode'])
            self.policy.eval_mode = False

            sum_R = self.eval_sars['R'].groupby(level=0).sum()

            mean_sum_R = sum_R.mean()
            sum_sum_R = sum_R.sum()
            std_sum_R = sum_R.std()
            max_sum_R = sum_R.max()
            min_sum_R = sum_R.min()

            logger.info('statistics on sum R -- mean: {:.6f} sum: {:.6f} std: {:.6f} max: {:.6f} min: {:.6f}'.format(
                mean_sum_R, sum_sum_R, std_sum_R, max_sum_R, min_sum_R))

        return_dict = dict(mean_sum_R=mean_sum_R, sum_sum_R=sum_sum_R, std_sum_R=std_sum_R,
                           max_sum_R=max_sum_R, min_sum_R=min_sum_R)

        gc.collect()

        return return_dict

    def reset(self, config: dict, rep: int) -> None:
        self._init_SARS()

        self.state_kernel, self.state_action_kernel = self._init_kernels()

        self.sampler = self._init_sampler()

        self.policy = self._init_policy(output_bounds=(self.sampler.env.action_space.low,
                                                       self.sampler.env.action_space.high))

    def _init_kernels(self):
        kernel_params = self._params['kernel']
        sampling_params = self._params['sampling']

        if kernel_params['kb_dist'] == 'mean':
            from kb_learning.kernel import compute_mean_position
            self.state_preprocessor = compute_mean_position
        elif kernel_params['kb_dist'] == 'mean-cov':
            from kb_learning.kernel import compute_mean_and_cov_position
            self.state_preprocessor = compute_mean_and_cov_position

        state_kernel = KilobotEnvKernel(rho=kernel_params['rho'],
                                        variance=kernel_params['variance'],
                                        kilobots_dim=sampling_params['num_kilobots'] * 2,
                                        light_dim=self._light_dimensions,
                                        weight_dim=1 if sampling_params['w_factor'] is None else 0,
                                        kilobots_dist_class=kernel_params['kb_dist'],
                                        light_dist_class=kernel_params['l_dist'],
                                        weight_dist_class=kernel_params['w_dist'])

        state_action_kernel = KilobotEnvKernel(rho=kernel_params['rho'],
                                               variance=kernel_params['variance'],
                                               kilobots_dim=sampling_params['num_kilobots'] * 2,
                                               light_dim=self._light_dimensions,
                                               weight_dim=1 if sampling_params['w_factor'] is None else 0,
                                               action_dim=self._action_dimensions,
                                               kilobots_dist_class=kernel_params['kb_dist'],
                                               light_dist_class=kernel_params['l_dist'],
                                               weight_dist_class=kernel_params['w_dist'],
                                               action_dist_class=kernel_params['a_dist'])

        return state_kernel, state_action_kernel

    def _init_policy(self, output_bounds):
        kernel_params = self._params['gp']
        gp_params = self._params['gp']
        sampling_params = self._params['sampling']

        gp_kernel = KilobotEnvKernel(rho=gp_params['rho'],
                                     variance=gp_params['prior_variance'],
                                     kilobots_dim=sampling_params['num_kilobots'] * 2,
                                     light_dim=self._light_dimensions,
                                     weight_dim=1 if sampling_params['w_factor'] is None else 0,
                                     kilobots_dist_class=kernel_params['kb_dist'],
                                     light_dist_class=kernel_params['l_dist'],
                                     weight_dist_class=kernel_params['w_dist'])
        gp_kernel.variance.fix()

        self.PolicyClass = SparseWeightedGP
        # self.PolicyClass = SparseWeightedGPyWrapper
        policy = self.PolicyClass(kernel=gp_kernel, noise_variance=self._params['gp']['noise_variance'],
                                  mean_function=step_towards_center([-2, -1]), output_bounds=output_bounds,
                                  output_dim=output_bounds[0].shape[0])

        return policy

    def _init_SARS(self):
        kb_columns_level = ['kb_{}'.format(i) for i in range(self._params['sampling']['num_kilobots'])]
        self.kilobots_columns = pd.MultiIndex.from_product([['S'], kb_columns_level, ['x', 'y']])
        self.light_columns = pd.MultiIndex.from_product([['S'], ['light'], ['x', 'y']])
        self.object_columns = pd.MultiIndex.from_product([['S'], ['object'], ['x', 'y', 'theta']])
        self.state_columns = self.kilobots_columns.append(self.light_columns)
        self.state_object_columns = self.state_columns.append(self.object_columns)
        self.action_columns = pd.MultiIndex.from_product([['A'], ['x', 'y'], ['']])
        self.reward_columns = pd.MultiIndex.from_arrays([['R'], [''], ['']])
        self.next_state_columns = self.state_columns.copy()
        self.next_state_columns.set_levels(['S_'], 0, inplace=True)
        self.extra_dim_columns = self.light_columns

        self.sars_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
            self.next_state_columns)

        self._info_columns = self.state_columns.append(self.object_columns)

        self.sars = pd.DataFrame(columns=self.sars_columns)

    def _init_sampler(self):
        sampling_params = self._params['sampling']
        self.SamplerClass = ParallelSARSSampler
        return self.SamplerClass(object_shape=sampling_params['object_shape'],
                                 object_width=sampling_params['object_width'],
                                 object_height=sampling_params['object_height'],
                                 light_type=sampling_params['light_type'],
                                 light_radius=sampling_params['light_radius'],
                                 registration_function=register_object_env,
                                 num_episodes=sampling_params['num_episodes'],
                                 num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                 num_kilobots=sampling_params['num_kilobots'],
                                 sars_column_index=self.sars_columns,
                                 state_column_index=self.state_object_columns,
                                 w_factor=sampling_params['w_factor'],
                                 num_workers=sampling_params['num_workers'],
                                 seed=self._seed, mp_context=self._MP_CONTEXT)

    def save_state(self, config: dict, rep: int, n: int) -> None:
        # save policy
        logger.info('pickling policy...')
        policy_file_name = os.path.join(self._log_path_rep, 'policy_it{:02d}.pkl'.format(self._it))
        with open(policy_file_name, mode='w+b') as policy_file:
            pickle.dump(self.policy.to_dict(), policy_file)

        # save kernels
        logger.info('pickling kernel...')
        state_kernel_file_name = os.path.join(self._log_path_rep, 'state_kernel_it{:02d}.pkl'.format(self._it))
        with open(state_kernel_file_name, mode='w+b') as state_kernel_file:
            pickle.dump(self.state_kernel.to_dict(), state_kernel_file)
        state_action_kernel_file_name = os.path.join(self._log_path_rep,
                                                     'state_action_kernel_it{:02d}.pkl'.format(self._it))
        with open(state_action_kernel_file_name, mode='w+b') as state_action_file:
            pickle.dump(self.state_action_kernel.to_dict(), state_action_file)

        if self._params['sampling']['save_trajectories']:
            logger.info('pickling sampling trajectories...')
            it_sars_file_name = os.path.join(self._log_path_rep, 'it_sars_it{:02d}.pkl'.format(self._it))
            self.it_sars.to_pickle(it_sars_file_name, compression='gzip')
            it_info_file_name = os.path.join(self._log_path_rep, 'it_info_it{:02d}.pkl'.format(self._it))
            self.it_info.to_pickle(it_info_file_name, compression='gzip')

        if self._params['eval']['save_trajectories']:
            logger.info('pickling eval trajectories...')
            eval_sars_file_name = os.path.join(self._log_path_rep, 'eval_sars_it{:02d}.pkl'.format(self._it))
            self.eval_sars.to_pickle(eval_sars_file_name, compression='gzip')
            eval_info_file_name = os.path.join(self._log_path_rep, 'eval_info_it{:02d}.pkl'.format(self._it))
            self.eval_info.to_pickle(eval_info_file_name, compression='gzip')

        # save theta
        logger.info('pickling theta and lstd samples...')
        theta_file_name = os.path.join(self._log_path_rep, 'theta_it{:02d}.npy'.format(self._it))
        np.save(theta_file_name, self.theta)
        lstd_samples_file_name = os.path.join(self._log_path_rep, 'lstd_samples_it{:02d}.pkl'.format(self._it))
        self.lstd_samples.to_pickle(lstd_samples_file_name, compression='gzip')

        # save SARS data
        logger.info('pickling SARS...')
        SARS_file_name = os.path.join(self._log_path_rep, 'last_SARS.pkl.gz')
        self.sars.to_pickle(SARS_file_name, compression='gzip')

    def restore_state(self, config: dict, rep: int, n: int) -> bool:
        # restore policy
        logger.info('restoring policy and kernel...')
        policy_file_name = os.path.join(self._log_path_rep, 'policy_it{:02d}.pkl'.format(n))
        with open(policy_file_name, mode='r+b') as policy_file:
            policy_dict = pickle.load(policy_file)
        self.policy = self.PolicyClass.from_dict(policy_dict)
        self.sampler.policy = self.policy

        # restore kernel
        logger.info('restoring policy and kernel...')
        state_kernel_file_name = os.path.join(self._log_path_rep, 'state_kernel_it{:02d}.pkl'.format(n))
        with open(state_kernel_file_name, mode='r+b') as state_kernel_file:
            state_kernel_dict = pickle.load(state_kernel_file)
        self.state_kernel = KilobotEnvKernel.from_dict(state_kernel_dict)
        state_action_kernel_file_name = os.path.join(self._log_path_rep, 'state_action_kernel_it{:02d}.pkl'.format(n))
        with open(state_action_kernel_file_name, mode='r+b') as state_action_kernel_file:
            state_action_kernel_dict = pickle.load(state_action_kernel_file)
        self.state_action_kernel = KilobotEnvKernel.from_dict(state_action_kernel_dict)

        if self._params['sampling']['save_trajectories']:
            logger.info('restoring sampling trajectories...')
            it_sars_file_name = os.path.join(self._log_path_rep, 'it_sars_it{:02d}.pkl'.format(n))
            self.it_sars = pd.read_pickle(it_sars_file_name, compression='gzip')
            it_info_file_name = os.path.join(self._log_path_rep, 'it_info_it{:02d}.pkl'.format(n))
            self.it_info = pd.read_pickle(it_info_file_name, compression='gzip')

        if self._params['eval']['save_trajectories']:
            logger.info('restoring eval trajectories...')
            eval_sars_file_name = os.path.join(self._log_path_rep, 'eval_sars_it{:02d}.pkl'.format(n))
            self.eval_sars = pd.read_pickle(eval_sars_file_name, compression='gzip')
            eval_info_file_name = os.path.join(self._log_path_rep, 'eval_info_it{:02d}.pkl'.format(n))
            self.eval_info = pd.read_pickle(eval_info_file_name, compression='gzip')

        # save theta
        logger.info('pickling theta and lstd samples...')
        theta_file_name = os.path.join(self._log_path_rep, 'theta_it{:02d}.npy'.format(n))
        self.theta = np.load(theta_file_name, self.theta)
        lstd_samples_file_name = os.path.join(self._log_path_rep, 'lstd_samples_it{:02d}.pkl'.format(n))
        self.lstd_samples = pd.read_pickle(lstd_samples_file_name, compression='gzip')

        # restore SARS data
        SARS_file_name = os.path.join(self._log_path_rep, 'last_SARS.pkl.gz')
        if not os.path.exists(SARS_file_name):
            return False
        self.sars = pd.read_pickle(SARS_file_name, compression='gzip')
        return True

    # TODO move to notebooks
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

# class SampledWeightACRepsLearner(ACRepsLearner):
#     def _init_policy(self, output_bounds):
#         policy = super()._init_policy(output_bounds)
#         policy.gp_prior_mean = step_towards_center([-3, -2])
#
#         return policy
#
#     def _init_SARS(self):
#         super()._init_SARS()
#
#         self.weight_columns = pd.MultiIndex.from_product([['S'], ['weight'], ['']])
#         self.state_columns = self.kilobots_columns.append(self.light_columns).append(self.weight_columns)
#         self.next_state_columns = self.state_columns.copy()
#         self.next_state_columns.set_levels(['S_'], 0, inplace=True)
#         self.extra_dim_columns = self.light_columns.append(self.weight_columns)
#
#         self.sars_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
#             self.next_state_columns)
#
#         self.sars = pd.DataFrame(columns=self.sars_columns)
#
#     def _plot_iteration_results(self, state_action_features):
#         # setup figure
#         fig = plt.figure(figsize=(11, 20))
#         gs = gridspec.GridSpec(nrows=4, ncols=4, width_ratios=[7, 7, 7, 1], height_ratios=[1, 3, 3, 3])
#         w0 = .166
#         w1 = .5
#         w2 = .833
#
#         # reward plot
#         ax_R = fig.add_subplot(gs[0, :])
#         plot_trajectory_reward_distribution(ax_R, self.it_sars['R'])
#
#         # value function plot
#         cmap_plasma = cm.get_cmap('plasma')
#         ax_V_w0 = fig.add_subplot(gs[1, 0])
#         ax_V_w1 = fig.add_subplot(gs[1, 1])
#         ax_V_w2 = fig.add_subplot(gs[1, 2])
#         ax_V_cb = fig.add_subplot(gs[1, 3])
#         ax_V_w0.set_title('value function, w = {}'.format(w0))
#         ax_V_w1.set_title('w = {}'.format(w1))
#         ax_V_w2.set_title('w = {}, iteration {}'.format(w2, self._it))
#         V_w0 = self._compute_value_function_grid(state_action_features, weight=w0)
#         plot_value_function(ax_V_w0, *V_w0, cmap=cmap_plasma)
#         plot_objects(ax_V_w0, env=self.sampler.env, fill=False)
#         V_w1 = self._compute_value_function_grid(state_action_features, weight=w1)
#         plot_value_function(ax_V_w1, *V_w1, cmap=cmap_plasma)
#         plot_objects(ax_V_w1, env=self.sampler.env, fill=False)
#         V_w2 = self._compute_value_function_grid(state_action_features, weight=w2)
#         im = plot_value_function(ax_V_w2, *V_w2, cmap=cmap_plasma)
#         plot_objects(ax_V_w2, env=self.sampler.env, fill=False)
#         fig.colorbar(im, cax=ax_V_cb)
#
#         # trajectories plot
#         cmap_gray = cm.get_cmap('gray')
#         ax_T_w0 = fig.add_subplot(gs[2, 0])
#         ax_T_w1 = fig.add_subplot(gs[2, 1])
#         ax_T_w2 = fig.add_subplot(gs[2, 2])
#         ax_T_cb = fig.add_subplot(gs[2, 3])
#         ax_T_w0.set_title('trajectories, w <= 0.33')
#         ax_T_w1.set_title('0.33 < w <= 0.66')
#         ax_T_w2.set_title('0.66 < w, iteration {}'.format(self._it))
#         plot_value_function(ax_T_w0, *V_w0, cmap=cmap_gray)
#         plot_objects(ax_T_w0, env=self.sampler.env)
#         small_w_index = self.it_sars['S']['weight'] <= .33
#         medium_w_index = (self.it_sars['S']['weight'] <= .66) & ~small_w_index
#         large_w_index = ~(small_w_index | medium_w_index)
#         plot_trajectories(ax_T_w0, self.it_sars['S']['light'].loc[small_w_index])
#         plot_value_function(ax_T_w1, *V_w1, cmap=cmap_gray)
#         plot_objects(ax_T_w1, env=self.sampler.env)
#         plot_trajectories(ax_T_w1, self.it_sars['S']['light'].loc[medium_w_index])
#         plot_value_function(ax_T_w2, *V_w2, cmap=cmap_gray)
#         plot_objects(ax_T_w2, env=self.sampler.env)
#         tr = plot_trajectories(ax_T_w2, self.it_sars['S']['light'].loc[large_w_index])
#         fig.colorbar(tr[0], cax=ax_T_cb)
#
#         # new policy plot
#         ax_P_w0 = fig.add_subplot(gs[3, 0])
#         ax_P_w1 = fig.add_subplot(gs[3, 1])
#         ax_P_w2 = fig.add_subplot(gs[3, 2])
#         ax_P_cb = fig.add_subplot(gs[3, 3])
#         ax_P_w0.set_title('learned policies, w = {}'.format(w0))
#         ax_P_w1.set_title('w = {}'.format(w1))
#         ax_P_w2.set_title('w = {}, iteration {}'.format(w2, self._it))
#         plot_value_function(ax_P_w0, *V_w0, cmap=cmap_gray)
#         plot_objects(ax_P_w0, env=self.sampler.env)
#         plot_policy(ax_P_w0, *self._compute_policy_quivers(weight=w0), cmap=cmap_plasma)
#         plot_value_function(ax_P_w1, *V_w1, cmap=cmap_gray)
#         plot_objects(ax_P_w1, env=self.sampler.env)
#         plot_policy(ax_P_w1, *self._compute_policy_quivers(weight=w1), cmap=cmap_plasma)
#         plot_value_function(ax_P_w2, *V_w2, cmap=cmap_gray)
#         plot_objects(ax_P_w2, env=self.sampler.env)
#         qv = plot_policy(ax_P_w2, *self._compute_policy_quivers(weight=w2), cmap=cmap_plasma)
#         fig.colorbar(qv, cax=ax_P_cb)
#
#         # save and show plot
#         show_plot_as_pdf(fig, path=self._log_path_rep, filename='plot_{:02d}.pdf'.format(self._it),
#                          overwrite=True, save_only=self._no_gui)
#         plt.close(fig)
#
#     def _compute_value_function_grid(self, state_action_features, weight, steps_x=40, steps_y=40):
#         x_range = self.sampler.env.world_x_range
#         y_range = self.sampler.env.world_y_range
#         [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
#         X = X.flatten()
#         Y = -Y.flatten()
#
#         # kilobots at light position
#         states = np.tile(np.c_[X, Y], [1, self.sampler.num_kilobots + 1])
#         states = np.c_[states, np.ones((states.shape[0], 1)) * weight]
#
#         # get mean actions
#         actions = self.policy.get_mean(states)
#
#         value_function = state_action_features(states, actions).dot(self.theta).reshape((steps_y, steps_x))
#
#         return value_function, x_range, y_range
#
#     def _compute_policy_quivers(self, weight, steps_x=20, steps_y=20):
#         x_range = self.sampler.env.world_x_range
#         y_range = self.sampler.env.world_y_range
#         [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
#         X = X.flatten()
#         Y = Y.flatten()
#
#         # kilobots at light position
#         states = np.tile(np.c_[X, Y], [1, self.sampler.num_kilobots + 1])
#         states = np.c_[states, np.ones((states.shape[0], 1)) * weight]
#
#         # get mean actions
#         mean_actions, sigma_actions = self.policy.get_mean_sigma(states)
#         # mean_actions /= np.linalg.norm(mean_actions, axis=1, keepdims=True)
#         mean_actions = mean_actions.reshape((steps_y, steps_x, mean_actions.shape[1]))
#         sigma_actions = sigma_actions.reshape((steps_y, steps_x))
#
#         actions = mean_actions, sigma_actions
#
#         return actions, x_range, y_range
#
#
# class GradientLightObjectACRepsLearner(ACRepsLearner):
#     def __init__(self):
#         super().__init__()
#
#         self._light_dimensions = 0
#         self._action_dimensions = 1
#
#     def _init_policy(self, output_bounds):
#         policy = super()._init_policy(output_bounds)
#         policy.gp_prior_mean = angle_from_swarm_mean(range(self._params['sampling']['num_kilobots'] * 2))
#
#         return policy
#
#     def _init_SARS(self):
#         super()._init_SARS()
#
#         self.light_columns = None
#         self.state_columns = self.kilobots_columns
#         self.action_columns = pd.MultiIndex.from_product([['A'], [''], ['']])
#         self.next_state_columns = self.state_columns.copy()
#         self.next_state_columns.set_levels(['S_'], 0, inplace=True)
#
#         self.sars_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
#             self.next_state_columns)
#
#         self.sars = pd.DataFrame(columns=self.sars_columns)
#
#     def _init_sampler(self):
#         sampling_params = self._params['sampling']
#         return ParallelSARSSampler(object_shape=sampling_params['object_shape'],
#                                    object_width=sampling_params['object_width'],
#                                    object_height=sampling_params['object_height'],
#                                    registration_function=register_gradient_light_object_env,
#                                    num_episodes=sampling_params['num_episodes'],
#                                    num_steps_per_episode=sampling_params['num_steps_per_episode'],
#                                    num_kilobots=sampling_params['num_kilobots'],
#                                    column_index=self.sars_columns,
#                                    w_factor=sampling_params['w_factor'],
#                                    num_workers=sampling_params['num_workers'],
#                                    seed=self._seed, mp_context=self._MP_CONTEXT)
#
#     def _plot_iteration_results(self, state_action_features):
#         # setup figure
#         fig = plt.figure(figsize=(10, 20))
#         gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[20, 1], height_ratios=[1, 3, 3, 3])
#
#         # reward plot
#         ax_R = fig.add_subplot(gs[0, :])
#         plot_trajectory_reward_distribution(ax_R, self.it_sars['R'])
#
#         # value function plot
#         ax_bef_V = fig.add_subplot(gs[1, 0])
#         ax_bef_V_cb = fig.add_subplot(gs[1, 1])
#         cmap_plasma = cm.get_cmap('plasma')
#         cmap_gray = cm.get_cmap('gray')
#         V = self._compute_value_function_grid(state_action_features)
#         im = plot_value_function(ax_bef_V, *V, cmap=cmap_plasma)
#         plot_objects(ax_bef_V, env=self.sampler.env, alpha=.3, fill=True)
#         ax_bef_V.set_title('value function, before ac_reps, iteration {}'.format(self._it))
#         fig.colorbar(im, cax=ax_bef_V_cb)
#
#         # trajectories plot
#         ax_bef_T = fig.add_subplot(gs[2, 0])
#         ax_bef_T_cb = fig.add_subplot(gs[2, 1])
#         ax_bef_T.set_title('trajectories, iteration {}'.format(self._it))
#         plot_value_function(ax_bef_T, *V, cmap=cmap_gray)
#         tr = plot_trajectories(ax_bef_T, compute_mean_position_pandas(self.it_sars['S']))
#         plot_objects(ax_bef_T, env=self.sampler.env)
#         fig.colorbar(tr[0], cax=ax_bef_T_cb)
#
#         # new policy plot
#         ax_aft_P = fig.add_subplot(gs[3, 0])
#         ax_aft_P_cb = fig.add_subplot(gs[3, 1])
#         plot_value_function(ax_aft_P, *self._compute_value_function_grid(state_action_features),
#                             cmap=cmap_gray)
#         qv = plot_policy(ax_aft_P, *self._compute_policy_quivers(), cmap=cmap_plasma)
#         plot_objects(ax_aft_P, env=self.sampler.env)
#         fig.colorbar(qv, cax=ax_aft_P_cb)
#         ax_aft_P.set_title('policy, after ac_reps, iteration {}'.format(self._it))
#
#         # save and show plot
#         show_plot_as_pdf(fig, path=self._log_path_rep, filename='plot_{:02d}.pdf'.format(self._it),
#                          overwrite=True, save_only=self._no_gui)
#         # plt.show(block=True)
#
#         plt.close(fig)
#
#     def _compute_value_function_grid(self, state_action_features, steps_x=40, steps_y=40):
#         x_range = self.sampler.env.world_x_range
#         y_range = self.sampler.env.world_y_range
#         [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
#         X = X.flatten()
#         Y = -Y.flatten()
#
#         # kilobots at light position
#         states = np.tile(np.c_[X, Y], [1, self.sampler.num_kilobots])
#
#         # get mean actions
#         actions = self.policy.get_mean(states)
#
#         value_function = state_action_features(states, actions).dot(self.theta).reshape((steps_y, steps_x))
#
#         return value_function, x_range, y_range
#
#     def _compute_policy_quivers(self, steps_x=40, steps_y=40):
#         x_range = self.sampler.env.world_x_range
#         y_range = self.sampler.env.world_y_range
#         [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
#         X = X.flatten()
#         Y = Y.flatten()
#
#         # kilobots at light position
#         states = np.tile(np.c_[X, Y], [1, self.sampler.num_kilobots])
#
#         # get mean actions
#         mean_actions, sigma_actions = self.policy.get_mean_sigma(states)
#         # mean_actions /= np.linalg.norm(mean_actions, axis=1, keepdims=True)
#         mean_actions = mean_actions.reshape((steps_y, steps_x, mean_actions.shape[1]))
#         sigma_actions = sigma_actions.reshape((steps_y, steps_x))
#
#         actions = mean_actions, sigma_actions
#
#         return actions, x_range, y_range
#
#
# class DualLightComplexObjectACRepsLearner(ACRepsLearner):
#     def __init__(self):
#         super().__init__()
#
#         self._light_dimensions = 2
#         self._action_dimensions = 2
#
#     def _init_policy(self, output_bounds):
#         policy = super()._init_policy(output_bounds)
#         policy.gp_prior_mean = angle_from_swarm_mean(range(self._params['sampling']['num_kilobots'] * 2), 2)
#
#         return policy
#
#     def _init_SARS(self):
#         self.light_columns = pd.MultiIndex.from_product([['S'], ['light_0', 'light_1'], ['theta']])
#         self.state_columns = self.kilobots_columns.append(self.light_columns)
#         self.action_columns = pd.MultiIndex.from_product([['A'], ['light_0', 'light_1'], ['des_theta']])
#         self.next_state_columns = self.state_columns.copy()
#         self.next_state_columns.set_levels(['S_'], 0, inplace=True)
#
#         self.sars_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
#             self.next_state_columns)
#
#         self.sars = pd.DataFrame(columns=self.sars_columns)
#
#     def _init_sampler(self):
#         sampling_params = self._params['sampling']
#         return ParallelSARSSampler(object_shape=sampling_params['object_shape'],
#                                    object_width=sampling_params['object_width'],
#                                    object_height=sampling_params['object_height'],
#                                    registration_function=register_dual_light_complex_object_env,
#                                    num_episodes=sampling_params['num_episodes'],
#                                    num_steps_per_episode=sampling_params['num_steps_per_episode'],
#                                    num_kilobots=sampling_params['num_kilobots'],
#                                    column_index=self.sars_columns,
#                                    w_factor=sampling_params['w_factor'],
#                                    num_workers=sampling_params['num_workers'],
#                                    seed=self._seed, mp_context=self._MP_CONTEXT)
#
#     def _plot_iteration_results(self, state_action_features):
#         # setup figure
#         fig = plt.figure(figsize=(10, 20))
#         gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[20, 1], height_ratios=[1, 3, 3, 3])
#
#         # reward plot
#         ax_R = fig.add_subplot(gs[0, :])
#         plot_trajectory_reward_distribution(ax_R, self.it_sars['R'])
#
#         # value function plot
#         ax_bef_V = fig.add_subplot(gs[1, 0])
#         ax_bef_V_cb = fig.add_subplot(gs[1, 1])
#         cmap_plasma = cm.get_cmap('plasma')
#         cmap_gray = cm.get_cmap('gray')
#         V = self._compute_value_function_grid(state_action_features)
#         im = plot_value_function(ax_bef_V, *V, cmap=cmap_plasma)
#         plot_objects(ax_bef_V, env=self.sampler.env, alpha=.3, fill=True)
#         ax_bef_V.set_title('value function, before ac_reps, iteration {}'.format(self._it))
#         fig.colorbar(im, cax=ax_bef_V_cb)
#
#         # trajectories plot
#         ax_bef_T = fig.add_subplot(gs[2, 0])
#         ax_bef_T_cb = fig.add_subplot(gs[2, 1])
#         ax_bef_T.set_title('trajectories, iteration {}'.format(self._it))
#         plot_value_function(ax_bef_T, *V, cmap=cmap_gray)
#         tr = plot_trajectories(ax_bef_T, compute_mean_position_pandas(self.it_sars['S']))
#         plot_objects(ax_bef_T, env=self.sampler.env)
#         fig.colorbar(tr[0], cax=ax_bef_T_cb)
#
#         # new policy plot
#         ax_aft_P = fig.add_subplot(gs[3, 0])
#         ax_aft_P_cb = fig.add_subplot(gs[3, 1])
#         plot_value_function(ax_aft_P, *self._compute_value_function_grid(state_action_features),
#                             cmap=cmap_gray)
#         (mean_actions, sigma_actions), x_range, y_range = self._compute_policy_quivers()
#         _ = plot_policy(ax_aft_P, (mean_actions[..., 0, None], sigma_actions), x_range, y_range, cmap=cmap_plasma)
#         qv = plot_policy(ax_aft_P, (mean_actions[..., 1, None], sigma_actions), x_range, y_range, cmap=cmap_plasma)
#         plot_objects(ax_aft_P, env=self.sampler.env)
#         fig.colorbar(qv, cax=ax_aft_P_cb)
#         ax_aft_P.set_title('policy, after ac_reps, iteration {}'.format(self._it))
#
#         # save and show plot
#         show_plot_as_pdf(fig, path=self._log_path_rep, filename='plot_{:02d}.pdf'.format(self._it),
#                          overwrite=True, save_only=self._no_gui)
#         # plt.show(block=True)
#
#         plt.close(fig)
#
#     def _compute_value_function_grid(self, state_action_features, steps_x=40, steps_y=40):
#         x_range = self.sampler.env.world_x_range
#         y_range = self.sampler.env.world_y_range
#         [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
#         X = X.flatten()
#         Y = -Y.flatten()
#
#         # kilobots at light position
#         states = np.tile(np.c_[X, Y], [1, self.sampler.num_kilobots])
#
#         # get mean actions
#         actions = self.policy.get_mean(states)
#
#         value_function = state_action_features(states, actions).dot(self.theta).reshape((steps_y, steps_x))
#
#         return value_function, x_range, y_range
#
#     def _compute_policy_quivers(self, steps_x=40, steps_y=40):
#         x_range = self.sampler.env.world_x_range
#         y_range = self.sampler.env.world_y_range
#         [X, Y] = np.meshgrid(np.linspace(*x_range, steps_x), np.linspace(*y_range, steps_y))
#         X = X.flatten()
#         Y = Y.flatten()
#
#         # kilobots at light position
#         states = np.tile(np.c_[X, Y], [1, self.sampler.num_kilobots])
#
#         # get mean actions
#         mean_actions, sigma_actions = self.policy.get_mean_sigma(states)
#         # mean_actions /= np.linalg.norm(mean_actions, axis=1, keepdims=True)
#         mean_actions = mean_actions.reshape((steps_y, steps_x, mean_actions.shape[1]))
#         sigma_actions = sigma_actions.reshape((steps_y, steps_x))
#
#         actions = mean_actions, sigma_actions
#
#         return actions, x_range, y_range
