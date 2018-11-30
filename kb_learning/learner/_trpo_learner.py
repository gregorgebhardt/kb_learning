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
from baselines.common import colorize, dataset, explained_variance, tf_util, zipsame
from baselines.common.cg import cg
from baselines.common.mpi_adam import MpiAdam
from cluster_work import ClusterWork
from mpi4py import MPI

from kb_learning.envs import MultiObjectTargetPoseEnv, NormalizeActionWrapper
from kb_learning.policy_networks.mlp_policy import MlpPolicyNetwork
from kb_learning.policy_networks.swarm_policy import SwarmPolicyNetwork
from kb_learning.tools.trpo_tools import ActWrapper, add_vtarg_and_adv, flatten_lists, traj_segment_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

logger = logging.getLogger('trpo')


class TRPOLearner(ClusterWork):
    _restore_supported = True
    _default_params = {
        'sampling': {
            'timesteps_per_batch': 2048,
            'done_after_steps':    1024,
            'num_objects':         None,
            'reward_function':     None,
            'swarm_reward':        True,
            'agent_reward':        True,
            'schedule': 'linear'
        },
        'trpo':     {
            'gamma':         0.99,
            'lambda':        0.98,
            'max_kl':        0.01,
            'entcoeff':      0.0,
            'cg_iterations': 10,
            'cg_damping':    1e-2,
            'vf_step_size':  1e-3,
            'vf_iterations': 5,
        },
        'policy':   {
            'type':             'swarm',
            'swarm_net_size':   (64,),
            'swarm_net_type':   'mean',
            'objects_net_size': (64,),
            'objects_net_type': 'mean',
            'extra_net_size':   (64,),
            'concat_net_size':  (64,),
            'feature_net_size': (100, 50, 25),
            'pd_bias_init':     None
        },
        'buffer_len': 500
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

        self.vfadam = None

        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.vf_loss = None

        self.get_flat = None
        self.set_from_flat = None

        self.sampling_schedule = None
        self.gamma = None
        self.lam = None
        self.max_kl = None
        self.cg_iters = None
        self.cg_damping = None

        self.vf_iters = None
        self.vf_stepsize = None

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
            tstart = time.time()
            yield
            logger.info(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
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
                             feature_net_size=self._params['policy']['feature_net_size'],
                             pd_bias_init=_pd_bias_init)

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

        ob = tf_util.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kl_old_new = oldpi.pd.kl(pi.pd)
        entropy = pi.pd.entropy()
        mean_kl = tf.reduce_mean(kl_old_new)
        mean_entropy = tf.reduce_mean(entropy)
        entropy_bonus = self._params['trpo']['entcoeff'] * mean_entropy

        vf_error = tf.reduce_mean(tf.square(pi.vpred - ret))

        log_likelihood_ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        surrogate_gain = tf.reduce_mean(log_likelihood_ratio * atarg)

        optimgain = surrogate_gain + entropy_bonus
        losses = [optimgain, mean_kl, entropy_bonus, surrogate_gain, mean_entropy]
        self.loss_names = ["optimizer_gain", "mean_kl", "entropy_loss", "surrogate_gain", "mean_entropy"]

        dist = mean_kl

        all_var_list = pi.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]

        self.vfadam = MpiAdam(vf_var_list, comm=self._COMM)

        self.get_flat = tf_util.GetFlat(var_list)
        self.set_from_flat = tf_util.SetFromFlat(var_list)
        kl_gradients = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = tf_util.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(kl_gradients, tangents)])
        fvp = tf_util.flatgrad(gvp, var_list)

        self.assign_old_eq_new = tf_util.function([], [], updates=[tf.assign(oldv, newv)
                                                                   for (oldv, newv) in
                                                                   zipsame(oldpi.get_variables(), pi.get_variables())])

        self.compute_losses = tf_util.function([ob, ac, atarg], losses)
        self.compute_lossandgrad = tf_util.function([ob, ac, atarg], losses + [tf_util.flatgrad(optimgain, var_list)])
        self.compute_fvp = tf_util.function([flat_tangent, ob, ac, atarg], fvp)
        self.compute_vflossandgrad = tf_util.function([ob, ret], tf_util.flatgrad(vf_error, vf_var_list))
        self.vf_loss = tf_util.function([ob, ret], vf_error)

        self.pi = ActWrapper(pi, policy_params)

        tf_util.initialize()
        th_init = self.get_flat()
        self._COMM.Bcast(th_init, root=0)
        self.set_from_flat(th_init)
        self.vfadam.sync()
        logger.info("Init param sum {}".format(th_init.sum()))

        self.seg_gen = traj_segment_generator(pi, wrapped_env, self._params['sampling']['timesteps_per_batch'],
                                              stochastic=True)

        self.gamma = self._params['trpo']['gamma']
        self.lam = self._params['trpo']['lambda']
        self.max_kl = self._params['trpo']['max_kl']
        self.sampling_schedule = self._params['sampling']['schedule']
        self.vf_iters = self._params['trpo']['vf_iterations']
        self.vf_stepsize = self._params['trpo']['vf_step_size']

        self.cg_iters = self._params['trpo']['cg_iterations']
        self.cg_damping = self._params['trpo']['cg_damping']

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

        theta_new = None
        mean_losses = None

        curr_sampling_factor = self._get_schedule_factor(self.sampling_schedule)
        self.env.progress_factor = curr_sampling_factor
        with self.timed("sampling"):
            seg = self.seg_gen.__next__()

        add_vtarg_and_adv(seg, self.gamma, self.lam)

        # ob, ac, a_target, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, a_target, td_lambda_return = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        v_prediction_before = seg["vpred"]  # predicted value function before udpate
        # standardized advantage function estimate
        a_target_mean = a_target.mean()
        a_target_mean = self._COMM.allreduce(a_target_mean, MPI.SUM) / self.num_workers
        a_target_std = (a_target - a_target_mean).std()
        a_target_std = self._COMM.allreduce(a_target_std, MPI.SUM) / self.num_workers
        a_target = (a_target - a_target_mean) / a_target_std

        # if hasattr(self.pi, "ret_rms"): self.pi.ret_rms.update(td_lambda_return)
        # if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(ob)  # update running mean/std for policy

        # subsample data to take only 20% for computing the Fisher-vector products
        args = seg["ob"], seg["ac"], a_target
        fvp_arguments = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return self.allmean(self.compute_fvp(p, *fvp_arguments)) + self.cg_damping * p

        self.assign_old_eq_new()  # set old parameter values to new parameter values
        with self.timed("computegrad"):
            *loss_before, g = self.compute_lossandgrad(*args)
        loss_before = self.allmean(np.array(loss_before))
        g = self.allmean(g)
        if np.allclose(g, 0):
            logger.info("Got zero gradient. not updating")
        else:
            if g.dot(g) > 1e20:
                logger.warning("Gradient norm very large")
                g = g * 1e-4
            with self.timed("cg"):
                step_direction = cg(fisher_vector_product, g, cg_iters=self.cg_iters)
            assert np.isfinite(step_direction).all()
            shs = .5 * step_direction.dot(fisher_vector_product(step_direction))
            lm = np.sqrt(shs / self.max_kl)
            # logger.info("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            full_step = step_direction / lm
            expected_improvement = g.dot(full_step)
            surrogate_loss_before = loss_before[0]
            step_size = 1.0
            theta_before = self.get_flat()
            # line search with step_direction from gradient to ensure loss improves and KL constraint is not violated
            for _ in range(10):
                theta_new = theta_before + full_step * step_size
                self.set_from_flat(theta_new)
                mean_losses = surrogate_loss, kl_loss, *_ = self.allmean(np.array(self.compute_losses(*args)))
                improve = surrogate_loss - surrogate_loss_before
                logger.info("Expected: {}".format(expected_improvement))
                logger.info("Actual: {}".format(improve))
                if not np.isfinite(mean_losses).all():
                    logger.info("Got non-finite value of losses -- bad!")
                elif kl_loss > self.max_kl * 1.5:
                    logger.info("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.info("surrogate didn't improve. shrinking step.")
                else:
                    logger.info("Stepsize OK!")
                    break
                step_size *= .5
            else:
                logger.info("couldn't compute a good step")
                self.set_from_flat(theta_before)
            if self.num_workers > 1 and self.iters_so_far % 20 == 0:
                # list of tuples
                parameter_sums = self._COMM.allgather((theta_new.sum(), self.vfadam.getflat().sum()))
                assert all(np.allclose(ps, parameter_sums[0]) for ps in parameter_sums[1:])

        with self.timed("vf"):
            for _ in range(self.vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = self.allmean(self.compute_vflossandgrad(mbob, mbret))
                    self.vfadam.update(g, self.vf_stepsize)

        vf_loss = self.vf_loss(ob, td_lambda_return)

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
