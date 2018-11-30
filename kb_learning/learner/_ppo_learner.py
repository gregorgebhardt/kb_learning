import logging
import os
import random
import tempfile
import time
import zipfile
from collections import deque
from contextlib import contextmanager
from typing import Generator

import cloudpickle
import numpy as np
import tensorflow as tf
from baselines.common import Dataset, colorize, explained_variance, tf_util, zipsame
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from cluster_work import ClusterWork
from mpi4py import MPI

from kb_learning.envs import MultiObjectTargetPoseEnv, NormalizeActionWrapper
from kb_learning.policy_networks.mlp_policy import MlpPolicyNetwork
from kb_learning.policy_networks.swarm_policy import SwarmPolicyNetwork
from kb_learning.tools.ppo_tools import add_vtarg_and_adv, traj_segment_generator
from kb_learning.tools.trpo_tools import ActWrapper, flatten_lists

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

logger = logging.getLogger('ppo')


class PPOLearner(ClusterWork):
    _restore_supported = True
    _default_params = {
        'sampling': {
            'timesteps_per_batch': 2048,
            'done_after_steps':    1024,
            'num_objects':         None,
            'reward_function':     None,
            'swarm_reward':        True,
            'agent_reward':        True,
            'schedule':            'linear'
        },
        'ppo':      {
            'clip_range':          0.2,
            'entropy_coefficient': 0.0,
            'value_coefficient':   0.5,
            'gamma':               0.99,
            'lambda':              0.95,
            'schedule':            'linear',
            'epochs':              5,
            'step_size':           3e-4,
            'batch_size':          256,
            'max_gradient_norm':   .5,
        },
        'policy':   {
            'type':             'swarm',
            'swarm_net_size':   (64,),
            'swarm_net_type':   'mean',
            'objects_net_size': (64,),
            'objects_net_type': 'mean',
            'extra_net_size':   (64,),
            'concat_net_size':  (64,),
            'pd_bias_init':     None
        },
        'buffer_len': 300
    }

    def __init__(self):
        super().__init__()
        self.num_workers = None
        self.rank = None

        self.session = None
        self.graph = None
        self.env = None

        self.policy_fn = None

        self.timesteps_per_batch = None
        self.seg_gen = None

        self.loss_names = None

        self.pi = None

        self.adam = None

        self.assign_old_eq_new = None
        self.losses = None
        self.loss_gradient = None
        self.vf_loss = None

        self.sampling_schedule = None
        self.learning_schedule = None
        self.gamma = None
        self.lam = None
        self.epochs = None
        self.step_size = None
        self.batch_size = None

        self.episodes_so_far = 0
        self.timesteps_so_far = 0
        self.iters_so_far = 0
        self.tstart = time.time()
        self.ep_length_buffer = None  # rolling buffer for episode lengths
        self.ep_return_buffer = None  # rolling buffer for episode rewards

    @contextmanager
    def timed(self, msg):
        if self.rank == 0:
            logger.info(colorize(msg, color='magenta'))
            t_start = time.time()
            yield
            logger.info(colorize("done in %.3f seconds" % (time.time() - t_start), color='magenta'))
        else:
            yield

    def allmean(self, x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        self._COMM.Allreduce(x, out, op=MPI.SUM)
        out /= self.num_workers
        return out

    def reset(self, config: dict, rep: int):
        os.environ["OMP_NUM_THREADS"] = '1'

        sess_conf = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=1,
                                   intra_op_parallelism_threads=1)
        self.graph = tf.Graph()
        self.session = tf_util.make_session(config=sess_conf, make_default=True, graph=self.graph)

        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        if not self._COMM:
            logger.warning('setting self._COMM to MPI.COMM_WORLD')
            self._COMM = MPI.COMM_WORLD

        # create env
        self.env = MultiObjectTargetPoseEnv(configuration=config['env_config'],
                                            done_after_steps=self._params['sampling']['done_after_steps'])

        wrapped_env = NormalizeActionWrapper(self.env)

        # create policy function
        def policy_fn(name, **_policy_params):
            if self._params['policy']['type'] == 'swarm':
                return SwarmPolicyNetwork(name=name, **_policy_params)
            elif self._params['policy']['type'] == 'mlp':
                return MlpPolicyNetwork(name=name, **_policy_params)
            else:
                raise NotImplementedError

        _pd_bias_init = self._params['policy']['pd_bias_init'] or np.array([.0] * self.env.action_space.shape[0] * 2)
        policy_params = dict(ob_space=self.env.observation_space,
                             ac_space=self.env.action_space,  # only the type and shape of the action space is important
                             # we observe all other agents with (r, sin(a), cos(a), sin(th), cos(th), lin_vel, rot_vel)
                             agent_dims=self.env.kilobots_observation_space.shape[0] // self.env.num_kilobots,
                             num_agent_observations=self.env.num_kilobots,
                             # we observe all objects with (r, sin(a), cos(a), sin(th), cos(th), valid_indicator)
                             object_dims=self.env.object_observation_space.shape[0],
                             num_object_observations=len(config['env_config'].objects),
                             swarm_net_size=self._params['policy']['swarm_net_size'],
                             swarm_net_type=self._params['policy']['swarm_net_type'],
                             objects_net_size=self._params['policy']['objects_net_size'],
                             objects_net_type=self._params['policy']['objects_net_type'],
                             extra_net_size=self._params['policy']['extra_net_size'],
                             concat_net_size=self._params['policy']['concat_net_size'],
                             pd_bias_init=_pd_bias_init,
                             weight_sharing=True)

        self.policy_fn = policy_fn

        self.num_workers = self._COMM.Get_size()
        self.rank = self._COMM.Get_rank()
        logger.debug('initializing with rank {} of {}'.format(self.rank, self.num_workers))
        np.set_printoptions(precision=3)
        # Setup losses and stuff
        # ----------------------------------------
        pi = policy_fn("pi", **policy_params)
        oldpi = policy_fn("oldpi", **policy_params)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        # learning rate multiplier, updated with schedule
        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
        clip_param = self._params['ppo']['clip_range'] * lrmult

        ob = tf_util.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kl_old_new = oldpi.pd.kl(pi.pd)
        entropy = pi.pd.entropy()
        mean_kl = tf.reduce_mean(kl_old_new)
        mean_entropy = tf.reduce_mean(entropy)
        entropy_penalty = -1 * self._params['ppo']['entropy_coefficient'] * mean_entropy

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

        vpred_clipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
        vf_loss1 = tf.square(pi.vpred - ret)
        vf_loss2 = tf.square(vpred_clipped - ret)
        vf_loss = self._params['ppo']['value_coefficient'] * .5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
        # vf_loss = self._params['ppo']['value_coefficient'] * tf.reduce_mean(tf.square(pi.vpred - ret))

        total_loss = pol_surr + entropy_penalty + vf_loss
        losses = [pol_surr, entropy_penalty, vf_loss, mean_kl, mean_entropy]
        self.loss_names = ["pol_surr", "entropy_penalty", "vf_loss", "mean_kl", "mean_entropy"]

        var_list = pi.get_trainable_variables()

        max_grad_norm = self._params['ppo']['max_gradient_norm']
        self.loss_gradient = tf_util.function([ob, ac, atarg, ret, lrmult],
                                              losses + [tf_util.flatgrad(total_loss, var_list, max_grad_norm)])
        self.adam = MpiAdam(var_list, comm=self._COMM)

        self.assign_old_eq_new = tf_util.function([], [], updates=[tf.assign(oldv, newv)
                                                                   for (oldv, newv) in
                                                                   zipsame(oldpi.get_variables(), pi.get_variables())])
        self.losses = tf_util.function([ob, ac, atarg, ret, lrmult], losses)
        self.vf_loss = tf_util.function([ob, ret, lrmult], vf_loss)

        self.pi = ActWrapper(pi, policy_params)

        tf_util.initialize()
        self.adam.sync()

        self.seg_gen = traj_segment_generator(pi, wrapped_env, self._params['sampling']['timesteps_per_batch'],
                                              stochastic=True)

        self.gamma = self._params['ppo']['gamma']
        self.lam = self._params['ppo']['lambda']
        self.sampling_schedule = self._params['sampling']['schedule']
        self.learning_schedule = self._params['ppo']['schedule']
        self.epochs = self._params['ppo']['epochs']
        self.step_size = self._params['ppo']['step_size']
        self.batch_size = self._params['ppo']['batch_size']

        self.ep_length_buffer = deque(maxlen=self._params['buffer_len'])
        self.ep_return_buffer = deque(maxlen=self._params['buffer_len'])

    def _get_schedule_factor(self, schedule):
        if schedule == 'constant':
            factor = 1.0
        elif schedule == 'linear':
            factor = max(1.0 - float(self._it) / self._iterations, 0)
        elif schedule == 'sqrt':
            factor = np.sqrt(1.0 - float(self._it) / self._iterations)
        else:
            raise NotImplementedError
        return factor

    def iterate(self, config: dict, rep: int, n: int):
        # set random seed for repetition and iteration
        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        time_update_start = time.time()

        curr_learn_rate_factor = self._get_schedule_factor(self.learning_schedule)
        curr_sampling_factor = self._get_schedule_factor(self.sampling_schedule)

        self.env.progress_factor = curr_sampling_factor
        with self.timed("sampling"):
            seg = self.seg_gen.__next__()

        # rew_std = seg['rew'].std()
        # rew_std = self._COMM.allreduce(rew_std, MPI.SUM) / self.num_workers
        # seg['rew'] /= rew_std

        add_vtarg_and_adv(seg, self.gamma, self.lam)

        # ob, ac, a_target, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, a_target, td_lambda_return = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        v_prediction_before = seg["vpred"]  # predicted value function before udpate
        # standardized advantage function estimate
        a_target_mean = a_target.mean()
        # a_target_mean = self._COMM.allreduce(a_target_mean, MPI.SUM) / self.num_workers
        a_target_std = (a_target - a_target_mean).std()
        # a_target_std = self._COMM.allreduce(a_target_std, MPI.SUM) / self.num_workers
        a_target = (a_target - a_target_mean) / (a_target_std + 1e-8)

        d = Dataset(dict(ob=ob, ac=ac, atarg=a_target, vtarg=td_lambda_return), shuffle=True)
        optim_batchsize = self.batch_size or ob.shape[0]

        self.assign_old_eq_new()  # set old parameter values to new parameter values

        for ep_i in range(self.epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *new_losses, g = self.loss_gradient(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                    curr_learn_rate_factor)
                self.adam.update(g, self.step_size * curr_learn_rate_factor)
                losses.append(new_losses)

        logger.info("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            new_losses = self.losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                     curr_learn_rate_factor)
            losses.append(new_losses)
        mean_losses, _, _ = mpi_moments(losses, axis=0, comm=self._COMM)

        vf_loss = self.vf_loss(ob, td_lambda_return, curr_learn_rate_factor)

        vf_loss_mean = np.mean(self._COMM.allgather(vf_loss))
        len_ret_local = (seg["ep_lens"], seg["ep_rets"])  # local values
        len_ret_pairs = self._COMM.allgather(len_ret_local)  # list of tuples
        ep_lengths, ep_returns = map(flatten_lists, zip(*len_ret_pairs))
        self.ep_length_buffer.extend(ep_lengths)
        self.ep_return_buffer.extend(ep_returns)

        self.episodes_so_far += len(ep_lengths)
        self.timesteps_so_far += sum(ep_lengths)
        self.iters_so_far += 1

        if self._COMM.rank == 0:
            time_elapsed = time.time() - time_update_start

            return_values = dict(zip(self.loss_names, mean_losses))

            return_values.update({
                "ev_tdlam_before":         explained_variance(v_prediction_before, td_lambda_return),
                "vf_loss":                 vf_loss_mean,
                "lr_factor":               curr_learn_rate_factor,
                "episode_return_mean":     np.mean(self.ep_return_buffer),
                "episode_return_min":      np.min(self.ep_return_buffer),
                "episode_return_max":      np.max(self.ep_return_buffer),
                "episode_length_mean":     np.mean(self.ep_length_buffer),
                "episodes_this_iteration": len(ep_lengths),
                "episodes":                self.episodes_so_far,
                "timesteps":               self.timesteps_so_far,
                "frames_per_second":       int(np.sum(ep_lengths) / time_elapsed)
            })

            important_values = ["episode_return_mean", "ev_tdlam_before"]

            for k, v in return_values.items():
                if k in important_values:
                    logger.info(colorize("{}: {}".format(k, v), color='green', bold=True))
                else:
                    logger.info("{}: {}".format(k, v))

            return return_values

    def finalize(self):
        if self.env:
            self.env.close()
        self.session.close()
        del self.graph

    def save_state(self, config: dict, rep: int, n: int):
        # save parameters at every iteration
        policy_path = os.path.join(self._log_path_it, 'policy.pkl')
        self.pi.save(policy_path)

    def restore_state(self, config: dict, rep: int, n: int):
        # resore parameters
        policy_path = os.path.join(self._log_path_it, 'policy.pkl')
        logger.info('Restoring parameters from {}'.format(policy_path))
        # self.pi = ActWrapper.load(policy_path, self.policy_fn)

        with open(policy_path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)

        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            self.pi.load_state(os.path.join(td, "model"))

        return True

    @classmethod
    def plot_results(cls, configs_results: Generator):
        pass
