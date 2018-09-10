from ._learner import KilobotLearner

import os
import gc

import logging

from typing import Generator

from cluster_work import InvalidParameterArgument

from kb_learning.kernel import KilobotEnvKernel
from kb_learning.kernel import compute_median_bandwidth, compute_median_bandwidth_kilobots, angle_from_swarm_mean, \
    step_towards_center
from kb_learning.ac_reps.lstd import LeastSquaresTemporalDifference
from kb_learning.ac_reps.reps import ActorCriticReps
from kb_learning.ac_reps.spwgp import SparseWeightedGP

from kb_learning.envs.sampler import ParallelSARSSampler
from kb_learning.envs import register_object_relative_env

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import pickle

logger = logging.getLogger('kb_learning')


class ACRepsLearner(KilobotLearner):
    _restore_supported = True
    _default_params = {
        'sampling':         {
            'num_kilobots':          15,
            'w_factor':              .0,
            'num_episodes':          100,
            'num_steps_per_episode': 125,
            'num_SARS_samples':      10000,
            'num_workers':           None,
            'object_shape':          'quad',
            'object_width':          .15,
            'object_height':         .15,
            'save_trajectories':     True,
            'light_type':            'circular',
            'light_radius':          .2
        },
        'kernel':           {
            'kb_dist':                 'embedded',
            'l_dist':                  'maha',
            'a_dist':                  'maha',
            'w_dist':                  'maha',
            'bandwidth_factor_kb':     .3,
            'bandwidth_factor_light':  .55,
            'bandwidth_factor_action': .8,
            'bandwidth_factor_weight': .3,
            'rho':                     .5,
            'variance':                1.,
        },
        'learn_iterations': 1,
        'lstd':             {
            'discount_factor':    .99,
            'num_features':       1000,
            'num_policy_samples': 5,
        },
        'ac_reps':          {
            'epsilon': .3,
            'alpha': .0
        },
        'gp':               {
            'prior_variance':          7e-5,
            'noise_variance':          1e-5,
            'num_sparse_states':       1000,
            'kb_dist':                 'embedded',
            'l_dist':                  'maha',
            'w_dist':                  'maha',
            'bandwidth_factor_kb':     .3,
            'bandwidth_factor_light':  .55,
            'bandwidth_factor_weight': .3,
            'rho':                     .5,
            'use_prior_mean':          True
        },
        'eval':             {
            'num_episodes':          50,
            'num_steps_per_episode': 100,
            'save_trajectories':     True
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

        self._light_dimensions = None
        self._action_dimensions = None

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

            if self._light_dimensions:
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

        # compute state-action features
        logger.info('compute state-action features')
        phi_SA = state_action_features(self.sars['S'].values, self.sars[['A']].values)

        # compute state features
        logger.info('compute state features')
        phi_S = state_features(self.sars['S'].values)

        for i in range(self._params['learn_iterations']):
            # compute next state-action features
            logger.info('compute next state-action features')
            policy_samples = self._params['lstd']['num_policy_samples']
            next_actions = np.array([self.policy(self.sars['S_'].values) for _ in range(policy_samples)]).mean(axis=0)
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

            # compute sample weights using AC-REPS
            logger.info('learning weights [AC-REPS]')
            ac_reps = ActorCriticReps()
            ac_reps.epsilon = self._params['ac_reps']['epsilon']
            ac_reps.alpha = self._params['ac_reps']['alpha']
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
        # self.policy.eval_mode = True
        self.eval_sars, self.eval_info = self.sampler(self.policy,
                                                      num_episodes=self._params['eval']['num_episodes'],
                                                      num_steps_per_episode=self._params['eval'][
                                                          'num_steps_per_episode'])
        # self.policy.eval_mode = False

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

    def finalize(self):
        del self.sampler
        self.sampler = None

    def _init_kernels(self):
        kernel_params = self._params['kernel']
        sampling_params = self._params['sampling']

        if kernel_params['kb_dist'] == 'mean':
            from kb_learning.kernel import compute_mean_position
            self.state_preprocessor = compute_mean_position
        elif kernel_params['kb_dist'] == 'mean-cov':
            from kb_learning.kernel import compute_mean_and_cov_position
            self.state_preprocessor = compute_mean_and_cov_position

        if sampling_params['light_type'] == 'circular':
            self._light_dimensions = 2
            self._action_dimensions = 2
        elif sampling_params['light_type'] == 'linear':
            self._light_dimensions = 0
            self._action_dimensions = 1

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

        if self._params['gp']['use_prior_mean']:
            if sampling_params['light_type'] == 'circular':
                mean_function = step_towards_center([-2, -1])
            elif sampling_params['light_type'] == 'linear':
                mean_function = angle_from_swarm_mean(range(sampling_params['num_kilobots'] * 2))
            else:
                raise InvalidParameterArgument('Unknown argument for parameter \'sampling.light_type\': \'{}\''.format(
                    sampling_params['light_type']))
        else:
            mean_function = None

        policy = SparseWeightedGP(kernel=gp_kernel, noise_variance=self._params['gp']['noise_variance'],
                                  mean_function=mean_function, output_bounds=output_bounds,
                                  output_dim=output_bounds[0].shape[0])

        return policy

    def _init_SARS(self):
        sampling_params = self._params['sampling']

        kb_columns_level = ['kb_{}'.format(i) for i in range(self._params['sampling']['num_kilobots'])]
        self.kilobots_columns = pd.MultiIndex.from_product([['S'], kb_columns_level, ['x', 'y']])
        self.state_columns = self.kilobots_columns.copy()
        self.state_object_columns = self.state_columns.copy()
        self.object_columns = pd.MultiIndex.from_product([['S'], ['object'], ['x', 'y', 'theta']])

        if sampling_params['light_type'] == 'circular':
            self.light_columns = pd.MultiIndex.from_product([['S'], ['light'], ['x', 'y']])
            self.action_columns = pd.MultiIndex.from_product([['A'], ['x', 'y'], ['']])
            self.state_columns = self.state_columns.append(self.light_columns)
        else:
            self.light_columns = pd.MultiIndex.from_product([['S'], ['light'], ['theta']])
            self.action_columns = pd.MultiIndex.from_product([['A'], ['theta'], ['']])

        self.state_object_columns = self.state_object_columns.append(self.light_columns)

        if sampling_params['w_factor'] is None:
            self.weight_columns = pd.MultiIndex.from_product([['S'], ['weight'], ['']])
            self.state_columns = self.state_columns.append(self.weight_columns)
            self.state_object_columns = self.state_object_columns.append(self.weight_columns)

        self.state_object_columns = self.state_object_columns.append(self.object_columns)

        self.reward_columns = pd.MultiIndex.from_arrays([['R'], [''], ['']])

        self.next_state_columns = self.state_columns.copy()
        self.next_state_columns.set_levels(['S_'], 0, inplace=True)

        self.sars_columns = self.state_columns.append(self.action_columns).append(self.reward_columns).append(
            self.next_state_columns)

        self.sars = pd.DataFrame(columns=self.sars_columns)

    def _init_sampler(self):
        sampling_params = self._params['sampling']
        self.SamplerClass = ParallelSARSSampler
        # self.SamplerClass = SARSSampler
        return self.SamplerClass(object_shape=sampling_params['object_shape'],
                                 object_width=sampling_params['object_width'],
                                 object_height=sampling_params['object_height'],
                                 light_type=sampling_params['light_type'],
                                 light_radius=sampling_params['light_radius'],
                                 registration_function=register_object_relative_env,
                                 num_episodes=sampling_params['num_episodes'],
                                 num_steps_per_episode=sampling_params['num_steps_per_episode'],
                                 num_kilobots=sampling_params['num_kilobots'],
                                 sars_column_index=self.sars_columns,
                                 state_column_index=self.state_object_columns,
                                 w_factor=sampling_params['w_factor'],
                                 num_workers=sampling_params['num_workers'],
                                 seed=self._seed,
                                 # mp_context='spawn')
                                 mp_context=self._MP_CONTEXT)

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
        self.policy = SparseWeightedGP.from_dict(policy_dict)
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
        logger.info('restoring theta and lstd samples...')
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
        from pathlib import Path

        figure_dict = dict()
        path = None

        for config, results in results_config:
            if not path:
                path = Path(config['_config_path']).relative_to('data')
                path = Path('plots').joinpath(path)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

            if config['experiment_name'] not in figure_dict:
                _f = plt.figure()
                _a = _f.add_subplot(111)
                figure_dict[config['experiment_name']] = (_f, _a)
            else:
                _f, _a = figure_dict[config['experiment_name']]

            mean_sum_R = results['mean_sum_R']

            mean = mean_sum_R.groupby(level=1).mean()
            std = mean_sum_R.groupby(level=1).std()

            _a.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=.5)
            _a.plot(mean.index, mean, label=config['name'])

        for name, (_f, _a) in figure_dict.items():
            _a.legend()
            save_path = path.joinpath(name + '_plot.png')
            _f.savefig(save_path)
            # plt.savefig('plot.png')

            # plt.show(block=True)
