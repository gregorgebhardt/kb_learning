import logging
import os
import random
import time
from collections import deque
from typing import Generator
import multiprocessing

import tensorflow as tf
import cloudpickle
import gym
import numpy as np
from baselines.bench.monitor import Monitor
from baselines.common import explained_variance
from baselines.common.policies import build_policy
from baselines.common.tf_util import make_session
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

from cluster_work import ClusterWork, InvalidParameterArgument

from kb_learning.envs import register_object_env, NormalizeActionWrapper
from kb_learning.policy_networks import mlp, swarm_policy_network
from kb_learning.tools import const_fn, safe_mean
from kb_learning.tools.ppo_tools import model_lambda, Runner

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

logger = logging.getLogger('ppo')


class PPOLearner(ClusterWork):
    _restore_supported = True
    _default_params = {
        'sampling':              {
            'num_kilobots':     15,
            'w_factor':         .0,
            'num_workers':      10,
            'num_worker_steps': 500,
            'num_minibatches':  4,
            'object_shape':     'quad',
            'object_width':     .15,
            'object_height':    .15,
            'light_type':       'circular',
            'light_radius':     .2,
            'environment':      'absolute',
            'done_after_steps': 125
        },
        'ppo':                   {
            'learning_rate':        3e-4,
            'clip_range':           0.2,
            'gamma':                0.99,
            'lambda':               0.95,
            'entropy_coefficient':  0.0,
            'value_fn_coefficient': 0.5,
            'max_grad_norm':        0.5,
            'num_train_epochs':     4,
            'num_threads':          0,
        },
        'policy': {
            'type': 'me_mlp',
            'swarm_network_type':   'me',
            'swarm_network_size':   (64,),
            'object_dims':          4,
            'objects_network_type': 'mlp',
            'objects_network_size': (64,),
            'extra_dims':           2,
            'extra_network_size':   (64,),
            'concat_network_size':  (64,)
        },
        'updates_per_iteration': 5,
        'episode_info_length':   100,
    }

    def __init__(self):
        super().__init__()
        self.session: tf.Session = None
        self.graph: tf.Graph = None
        self.file_writer = None
        self.merged_summary = None

        self.learning_rate = None
        self.clip_range = None

        self.make_model = None

        self.env = None
        self.runner = None
        self.model = None

        self.num_workers = None
        self.steps_per_worker = None
        self.steps_per_batch = None
        self.num_minibatches = None
        self.steps_per_minibatch = None
        self.num_updates = None
        self.updates_per_iteration = None
        self.num_train_epochs = None

        self.ep_info_buffer = None

        self.timer = .0

    def reset(self, config: dict, rep: int):
        if self._params['ppo']['num_threads'] == 0:
            self._params['ppo']['num_threads'] = multiprocessing.cpu_count()
        os.environ["OMP_NUM_THREADS"] = str(self._params['ppo']['num_threads'])

        # get seed from cluster work
        tf.set_random_seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)

        # learning rate could change over time...
        self.learning_rate = const_fn(self._params['ppo']['learning_rate'])

        # clip range could change over time...
        self.clip_range = const_fn(self._params['ppo']['clip_range'])

        self.num_workers = self._params['sampling']['num_workers']
        self.steps_per_worker = int(self._params['sampling']['num_worker_steps'])
        self.steps_per_batch = self.num_workers * self.steps_per_worker
        self.num_minibatches = int(self._params['sampling']['num_minibatches'])
        self.steps_per_minibatch = self.steps_per_batch // self.num_minibatches
        self.num_train_epochs = self._params['ppo']['num_train_epochs']

        self.updates_per_iteration = self._params['updates_per_iteration']
        self.num_updates = config['iterations'] * self.updates_per_iteration

        # create environment
        if self._params['sampling']['environment'] == 'relative':
            logger.info('using object relative environment')
            env_class = 'ObjectRelativeEnv'
            env_params = dict(weight=self._params['sampling']['w_factor'])
        elif self._params['sampling']['environment'] == 'absolute':
            logger.info('using object absolute environment')
            env_class = 'ObjectAbsoluteEnv'
            env_params = dict()
        else:
            raise InvalidParameterArgument("Invalid argument for parameter 'sampling.environment: {}'".format(
                self._params['sampling']['environment']))

        env_params.update(dict(entry_point='kb_learning.envs:' + env_class,
                               num_kilobots=self._params['sampling']['num_kilobots'],
                               object_shape=self._params['sampling']['object_shape'],
                               object_width=self._params['sampling']['object_width'],
                               object_height=self._params['sampling']['object_height'],
                               light_type=self._params['sampling']['light_type'],
                               light_radius=self._params['sampling']['light_radius'],
                               done_after_steps=self._params['sampling']['done_after_steps']))

        env_id = register_object_env(**env_params)
        proto_env = gym.make(env_id)

        def _make_env(i, seed):
            def _make_env_i():
                np.random.seed(seed + 10000 * i)
                _env_id = register_object_env(**env_params)
                env = Monitor(NormalizeActionWrapper(gym.make(_env_id)), None, True)
                return env

            return _make_env_i

        envs = [_make_env(i, self._seed) for i in range(self._params['sampling']['num_workers'])]

        # wrap env in SubprocVecEnv
        # self.env = SubprocVecEnv(envs, context=self._MP_CONTEXT)
        self.env = ShmemVecEnv(envs, context=self._MP_CONTEXT)

        ob_space = self.env.observation_space
        ac_space = self.env.action_space

        # create network
        if self._params['policy']['type'] == 'me_mlp':
            logger.info('using mean embedding policy')
            network = swarm_policy_network(num_agents=proto_env.num_kilobots,
                                           swarm_network_size=self._params['policy']['swarm_network_size'],
                                           swarm_network_type=self._params['policy']['swarm_network_type'],
                                           num_objects=len(proto_env.objects),
                                           object_dims=self._params['policy']['object_dims'],
                                           objects_network_size=self._params['policy']['objects_network_size'],
                                           objects_network_type=self._params['policy']['objects_network_type'],
                                           extra_dims=self._params['policy']['extra_dims'],
                                           exta_network_size=self._params['policy']['extra_network_size'],
                                           concat_network_size=self._params['policy']['concat_network_size'])
        elif self._params['policy']['type'] == 'mlp':
            logger.info('using plain mlp policy')
            network = mlp(size=self._params['policy']['mlp_size'])
        elif self._params['policy']['type'] == 'mean_mlp':
            logger.info('using plain mlp policy with mean wrapper')
            from kb_learning.policy_networks.policy_decorator import compute_mean
            network = compute_mean(mlp(size=self._params['policy']['mlp_size']),
                                   num_kilobots=self._params['sampling']['num_kilobots'])
        elif self._params['policy']['type'] == 'mean_var_mlp':
            logger.info('using plain mlp policy with mean-variance wrapper')
            from kb_learning.policy_networks.policy_decorator import compute_mean_var
            network = compute_mean_var(mlp(size=self._params['policy']['mlp_size']),
                                       num_kilobots=self._params['sampling']['num_kilobots'])
        else:
            raise InvalidParameterArgument("Invalid argument for parameter 'policy.type': {}".format(
                self._params['policy']['type']))

        policy = build_policy(ob_space, ac_space, network)

        self.make_model = model_lambda(policy=policy, ob_space=ob_space, ac_space=ac_space,
                                       num_workers=self.num_workers, steps_per_minibatch=self.steps_per_minibatch,
                                       steps_per_worker=self.steps_per_worker,
                                       entropy_coefficient=self._params['ppo']['entropy_coefficient'],
                                       value_fn_coefficient=self._params['ppo']['value_fn_coefficient'],
                                       max_grad_norm=self._params['ppo']['max_grad_norm'])

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = make_session(num_cpu=self._params['ppo']['num_threads'], graph=self.graph)
            self.session.__enter__()
            # with tf.variable_scope(self._name + '_rep_{}'.format(rep)) as base_scope:
            self.model = self.make_model()

            # self.merged_summary = tf.summary.merge_all()

            self.runner = Runner(env=self.env, model=self.model, nsteps=self.steps_per_worker,
                                 gamma=self._params['ppo']['gamma'], lam=self._params['ppo']['lambda'])

            self.model.act_model.state = tf.constant(0)

        self.ep_info_buffer = deque(maxlen=self._params['episode_info_length'])

    def iterate(self, config: dict, rep: int, n: int):
        # set random seed for repetition and iteration
        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        if not self.graph.finalized:
            self.graph.finalize()

        values = None
        returns = None
        # loss_values = np.zeros(self.updates_per_iteration)
        time_update_start = time.time()
        # frames_per_second = None
        # time_elapsed = .0

        for i_update in range(self.updates_per_iteration):
            _finished_updates = n * self._params['updates_per_iteration'] + i_update
            frac = 1.0 - _finished_updates / self.num_updates
            _learning_rate = self.learning_rate(frac)
            _clip_range = self.clip_range(frac)

            # sample environments
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.runner.run()
            logger.info('iteration {}: mean return/step: {}  mean reward/step: {}'.format(i_update, safe_mean(returns),
                                                                                  safe_mean([e['r'] / e['l'] for e in
                                                                                             epinfos])))

            # todo save trajectories from obs...

            self.ep_info_buffer.extend(epinfos)
            mb_loss_values = []

            # train the model in mini-batches
            indices = np.arange(self.steps_per_batch)
            for i_epoch in range(self.num_train_epochs):
                np.random.shuffle(indices)
                for start in range(0, self.steps_per_batch, self.steps_per_minibatch):
                    end = start + self.steps_per_minibatch
                    mb_indices = indices[start:end]
                    slices = (arr[mb_indices] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mb_loss_values.append(self.model.train(_learning_rate, _clip_range, *slices))

            # summary = self.session.run([self.merged_summary, self.model])
            # self.file_writer.add_summary(summary, i_update + n * self.updates_per_iteration)

            # TODO fix this
            # loss_values[i_update] = np.mean(mb_loss_values, axis=0)

        time_elapsed = time.time() - time_update_start
        frames_per_second = int(self.steps_per_batch * self.updates_per_iteration / time_elapsed)
        self.timer += time_elapsed

        return_values = {
            # compute elapsed number of steps
            'serial_steps':        (n + 1) * self.updates_per_iteration * self.steps_per_worker,
            # compute total number of steps, i.e., the number of simulated steps
            'total_steps':         (n + 1) * self.updates_per_iteration * self.steps_per_batch,
            # compute performed update iterations
            'num_updates':         (n + 1) * self.updates_per_iteration,
            # fps in this iteration
            'frames_per_second':   frames_per_second,
            'explained_variance':  float(explained_variance(values, returns)),
            # compute mean over episode rewards
            'episode_reward_mean': safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]),
            # compute mean over episode lengths
            'episode_length_mean': safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]),
            # elapsed time during iteration
            'time_elapsed':        time_elapsed,
            # elapsed time since starting the experiment
            'total_time_elapsed':  self.timer
        }

        for k, v in return_values.items():
            logger.info("{}: {}".format(k, v))

        return return_values

    def finalize(self):
        if self.env:
            self.env.close()
        self.session.close()

    def save_state(self, config: dict, rep: int, n: int):
        # dump model only after the first iteration
        if n == 0:
            model_path = os.path.join(self._log_path_rep, 'make_model.pkl')
            logger.info('Saving model to to {}'.format(model_path))
            with open(model_path, 'wb') as fh:
                fh.write(cloudpickle.dumps(self.make_model))

        parameters_path = os.path.join(self._log_path_it, 'model_parameters')
        logger.info('Saving parameters to {}'.format(parameters_path))
        self.model.save(parameters_path)

        # save seed and time information TODO remove
        seed_time_path = os.path.join(self._log_path_it, 'seed_time')
        logger.info('Saving seed and timer information to {}'.format(seed_time_path))
        np_random_state = np.random.get_state()
        with open(seed_time_path, 'wb') as fh:
            cloudpickle.dump(dict(np_random_state=np_random_state, timer=self.timer), fh)

    def restore_state(self, config: dict, rep: int, n: int):
        # we do not need to restore the model as it is reconstructed from the parameters

        # restore model parameters
        parameters_path = os.path.join(self._log_path_it, 'model_parameters')
        logger.info('Restoring parameters from {}'.format(parameters_path))
        self.model.load(parameters_path)

        # restore seed and time TODO remove
        seed_time_path = os.path.join(self._log_path_it, 'seed_time')
        logger.info('Restoring seed and timer information to {}'.format(seed_time_path))
        with open(seed_time_path, 'rb') as fh:
            d = cloudpickle.load(fh)
            if 'timer' not in d or 'np_random_state' not in d:
                return False
            self.timer = d['timer']
            np.random.set_state(d['np_random_state'])

        return True

    @classmethod
    def plot_results(cls, configs_results: Generator):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.add_subplot(111)

        for config, results in configs_results:
            if 'w_factor0.5' in config['name'] or 'w_factor1.0' in config['name']:
                continue
            mean_ep_reward = results.groupby(level=1).episode_reward_mean.mean()
            std_ep_reward = results.groupby(level=1).episode_reward_mean.std()

            axes.fill_between(mean_ep_reward.index, mean_ep_reward - 2 * std_ep_reward,
                              mean_ep_reward + 2 * std_ep_reward, alpha=.5)
            axes.plot(mean_ep_reward.index, mean_ep_reward, label=config['name'])

        axes.legend()
        plt.show(block=True)
