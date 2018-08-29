import functools
import os
import random
import time
from collections import deque

import cloudpickle
import gym
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from baselines.bench.monitor import Monitor
from baselines.common import explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2.ppo2 import learn
from cluster_work import ClusterWork

from kb_learning.envs import register_object_relative_env, NormalizeActionWrapper
from kb_learning.policy_networks import mlp, me_mlp
from kb_learning.tools import swap_flatten_01, const_fn, safe_mean

import logging
logger = logging.getLogger('ppo')


class PPOLearner(ClusterWork):
    # TODO set this to true and test restoring of repetitions
    _restore_supported = False
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
            'light_radius':     .2
        },
        'ppo':                   {
            'learning_rate':        3e-4,
            'clip_range':           0.2,
            'gamma':                0.99,
            'lambda':               0.95,
            'entropy_coefficient':  0.0,
            'value_fn_coefficient': 0.5,
            'max_grad_norm':        0.5,
            'num_train_epochs':     4
        },
        'policy': {
            'use_mean_embedding': True,
            'me_size': (128, 128),
            'mlp_size': (128, 128)
        },
        'updates_per_iteration': 5,
        'episode_info_length':   100,
    }
    pass

    def __init__(self):
        super().__init__()
        self.session: tf.Session = None
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

        self._default_tf_graph = None

        self.timer = .0

    def reset(self, config: dict, rep: int):
        self.session = tf_util.single_threaded_session()
        self.session.__enter__()

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
        env_id = register_object_relative_env(weight=self._params['sampling']['w_factor'],
                                              num_kilobots=self._params['sampling']['num_kilobots'],
                                              object_shape=self._params['sampling']['object_shape'],
                                              object_width=self._params['sampling']['object_width'],
                                              object_height=self._params['sampling']['object_height'],
                                              light_type=self._params['sampling']['light_type'],
                                              light_radius=self._params['sampling']['light_radius'])

        def _make_env(i):
            def _make_env_i():
                np.random.seed(self._seed + 10000 * i)
                env = Monitor(NormalizeActionWrapper(gym.make(env_id)), None, True)
                return env

            return _make_env_i

        envs = [_make_env(i) for i in range(self._params['sampling']['num_workers'])]

        # wrap env in SubprocVecEnv
        self.env = SubprocVecEnv(envs)

        ob_space = self.env.observation_space
        ac_space = self.env.action_space

        # create network
        if self._params['policy']['use_mean_embedding']:
            logger.debug('using mean embedding policy')
            network = me_mlp(num_me_inputs=self._params['sampling']['num_kilobots'], dim_me_inputs=2,
                             me_size=self._params['policy']['me_size'], mlp_size=self._params['policy']['mlp_size'])
        else:
            logger.debug('using plain mlp policy')
            network = mlp(size=self._params['policy']['mlp_size'])

        policy = build_policy(ob_space, ac_space, network)

        self.make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                                        nbatch_act=self.num_workers,
                                        nbatch_train=self.steps_per_minibatch,
                                        nsteps=self.steps_per_worker,
                                        ent_coef=self._params['ppo']['entropy_coefficient'],
                                        vf_coef=self._params['ppo']['value_fn_coefficient'],
                                        max_grad_norm=self._params['ppo']['max_grad_norm'])

        with tf.variable_scope(config['name'] + '_rep_{}'.format(rep)) as base_scope:
            self.model = self.make_model()

            # self.merged_summary = tf.summary.merge_all()

        self.runner = Runner(env=self.env, model=self.model, nsteps=self.steps_per_worker,
                             gamma=self._params['ppo']['gamma'], lam=self._params['ppo']['lambda'])

        self.ep_info_buffer = deque(maxlen=self._params['episode_info_length'])

    def iterate(self, config: dict, rep: int, n: int):
        values = None
        returns = None
        loss_values = np.zeros(self.updates_per_iteration)
        time_update_start = time.time()

        for i_update in range(self.updates_per_iteration):
            _finished_updates = n * self._params['updates_per_iteration'] + i_update
            frac = 1.0 - _finished_updates / self.num_updates
            _learning_rate = self.learning_rate(frac)
            _clip_range = self.clip_range(frac)

            # sample environments
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.runner.run()

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
            with open(os.path.join(self._log_path_rep, 'make_model.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(self.make_model))

        save_path = os.path.join(self._log_path_it, 'model_parameters')
        logger.info('Saving to {}'.format(save_path))
        self.model.save(save_path)

        # save seed and time information
        np_seed = np.random.seed()
        with open(os.path.join(self._log_path_it, 'seed_time'), 'wb') as fh:
            cloudpickle.dump(dict(np_seed=np_seed, timer=self.timer), fh)

    def restore_state(self, config: dict, rep: int, n: int):
        # we do not need to restore the model as it is reconstructed from the parameters

        # restore model parameters
        save_path = os.path.join(self._log_path_it, 'model_parameters')
        logger.info('Saving to {}'.format(save_path))
        self.model.load(save_path)

        # restore seed and time
        with open(os.path.join(self._log_path_it, 'seed_time'), 'rb') as fh:
            d = cloudpickle.load(fh)
            self.timer = d['timer']
            np.random.seed(d['np_seed'])


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE) as ppo_scope:
            act_model = policy(nbatch_act, 1, sess)
            train_model = policy(nbatch_train, nsteps, sess)
            eval_model = policy(1, 1, sess)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        # tf.summary.scalar('ppo loss', loss)

        params = tf.trainable_variables(ppo_scope.name)
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
        # from mpi4py import MPI
        # trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        _train = trainer.apply_gradients(grads_and_var)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE:     cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.eval_model = eval_model
        self.step = act_model.step
        self.eval_step = eval_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        # global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        # sync_from_root(sess, global_variables)  # pylint: disable=E1101


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(swap_flatten_01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


def train(num_envs):
    sess = tf_util.single_threaded_session()

    with sess:
        # create env here
        # TODO add wrapper for gym to reset in case object close to env boundaries or num time steps too large
        env_id = register_object_relative_env(1., 10, 'quad', .15, .15, 'circular', .20)

        def _make_env(i):
            def _make_env_i():
                import numpy as np
                np.random.seed(1234 * i + 7809)
                env = Monitor(NormalizeActionWrapper(gym.make(env_id)), 'env_{}_monitor.csv'.format(i), True)
                return env

            return _make_env_i

        envs = [_make_env(i) for i in range(num_envs)]

        # wrap env in SubprocVecEnv
        vec_env = SubprocVecEnv(envs)

        network = me_mlp(num_me_inputs=10, dim_me_inputs=2)

        # def policy_fn(name):
        #     return mlp_policy.MlpPolicy(name=name, ob_space=vec_env.observation_space, ac_space=vec_env.action_space,
        #         hid_size=256, num_hid_layers=3)

        model = learn(network=network, env=vec_env, total_timesteps=2000000, nsteps=500, seed=1234, log_interval=5,
                      save_interval=5)

        vec_env.close()


if __name__ == '__main__':
    train(num_envs=10)
