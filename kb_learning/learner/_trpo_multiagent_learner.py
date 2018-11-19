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
import tensorflow as tf
import numpy as np
from baselines.common import tf_util, explained_variance, colorize, zipsame, dataset
from baselines.common.cg import cg
from baselines.common.mpi_adam import MpiAdam

from cluster_work import ClusterWork
from mpi4py import MPI

from kb_learning.envs import NormalizeActionWrapper, MultiObjectDirectControlEnv
from kb_learning.policy_networks.trpo_policy import MlpPolicy
from kb_learning.tools.trpo_tools import traj_segment_generator_ma, add_vtarg_and_adv_ma, flatten_lists, ActWrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

logger = logging.getLogger('trpo')


# np.seterr(all='raise')


class TRPOMultiAgentLearner(ClusterWork):
    _restore_supported = True
    _default_params = {
        'sampling': {
            'timesteps_per_batch': 2048,
            'done_after_steps':    1024,
            'num_objects':         None,
            'reward_function':     None,
            'swarm_reward':        False,
            'agent_reward':        False,
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
            'swarm_net_size':  (64,),
            'swarm_net_type':  'mean',
            'objects_net_size': (64,),
            'objects_net_type': 'mean',
            'extra_net_size':  (64,),
            'concat_net_size': (64,)
        },
    }

    def __init__(self):
        super().__init__()
        self.num_workers = None
        self.rank = None
        self.session = None

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
        self.lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards

    @contextmanager
    def timed(self, msg):
        if self.rank == 0:
            logger.debug(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            logger.debug(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(self, x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        self._COMM.Allreduce(x, out, op=MPI.SUM)
        out /= self.num_workers
        return out

    def reset(self, config: dict, rep: int):
        os.environ["OMP_NUM_THREADS"] = '2'

        sess_conf = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=2,
                                   intra_op_parallelism_threads=1)
        self.session = tf_util.make_session(config=sess_conf, make_default=True)

        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        if not self._COMM:
            self._COMM = MPI.COMM_WORLD

        if self._params['sampling']['num_objects']:
            if isinstance(self._params['sampling']['num_objects'], int):
                config['env_config'].objects = [config['env_config'].objects[0]] * self._params['sampling'][
                    'num_objects']

        # create env
        self.env = MultiObjectDirectControlEnv(configuration=config['env_config'],
                                               reward_function=self._params['sampling']['reward_function'],
                                               swarm_reward=self._params['sampling']['swarm_reward'],
                                               agent_reward=self._params['sampling']['agent_reward'],
                                               done_after_steps=self._params['sampling']['done_after_steps'])

        wrapped_env = NormalizeActionWrapper(self.env)

        # create policy function
        def policy_fn(name, ob_space, ac_space, env_config, agent_dims, object_dims, swarm_net_size, swarm_net_type,
                      objects_net_size, objects_net_type, extra_net_size, concat_net_size):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                             num_agent_observations=env_config.kilobots.num - 1, agent_obs_dims=agent_dims,
                             num_object_observations=len(env_config.objects), object_obs_dims=object_dims,
                             swarm_net_size=swarm_net_size, swarm_net_type=swarm_net_type,
                             objects_net_size=objects_net_size, objects_net_type=objects_net_type,
                             extra_net_size=extra_net_size, concat_net_size=concat_net_size)

        policy_params = dict(ob_space=self.env.observation_space,
                             ac_space=self.env.kilobots[0].action_space,
                             env_config=config['env_config'],
                             # we observe all other agents with (r, sin(a), cos(a), sin(th), cos(th), lin_vel, rot_vel)
                             agent_dims=7,
                             # we observe all objects with (r, sin(a), cos(a), sin(th), cos(th), valid_indicator)
                             object_dims=6,
                             swarm_net_size=self._params['policy']['swarm_net_size'],
                             swarm_net_type=self._params['policy']['swarm_net_type'],
                             objects_net_size=self._params['policy']['objects_net_size'],
                             objects_net_type=self._params['policy']['objects_net_type'],
                             extra_net_size=self._params['policy']['extra_net_size'],
                             concat_net_size=self._params['policy']['concat_net_size'])

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

        vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

        log_likelihood_ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        surrogate_gain = tf.reduce_mean(log_likelihood_ratio * atarg)

        optimgain = surrogate_gain + entropy_bonus
        losses = [optimgain, mean_kl, entropy_bonus, surrogate_gain, mean_entropy]
        self.loss_names = ["optimizer_gain", "mean_kl", "entropy_loss", "surrogate_gain", "mean_entropy"]

        dist = mean_kl

        all_var_list = pi.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        # var_list.extend([v for v in all_var_list if v.name.split("/")[1].startswith("feat")])
        vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        # vf_var_list.extend([v for v in all_var_list if v.name.split("/")[1].startswith("feat")])
        self.vfadam = MpiAdam(vf_var_list, comm=self._COMM)

        self.get_flat = tf_util.GetFlat(var_list)
        self.set_from_flat = tf_util.SetFromFlat(var_list)
        klgrads = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = tf_util.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
            start += sz
        gvp = tf.add_n(
            [tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
        fvp = tf_util.flatgrad(gvp, var_list)

        self.assign_old_eq_new = tf_util.function([], [], updates=[tf.assign(oldv, newv)
                                                                   for (oldv, newv) in
                                                                   zipsame(oldpi.get_variables(), pi.get_variables())])
        self.compute_losses = tf_util.function([ob, ac, atarg], losses)
        self.compute_lossandgrad = tf_util.function([ob, ac, atarg], losses + [tf_util.flatgrad(optimgain, var_list)])
        self.compute_fvp = tf_util.function([flat_tangent, ob, ac, atarg], fvp)
        self.compute_vflossandgrad = tf_util.function([ob, ret], tf_util.flatgrad(vferr, vf_var_list))
        self.vf_loss = tf_util.function([ob, ret], vferr)

        self.pi = ActWrapper(pi, policy_params)

        tf_util.initialize()
        th_init = self.get_flat()
        self._COMM.Bcast(th_init, root=0)
        self.set_from_flat(th_init)
        self.vfadam.sync()
        logger.info("Init param sum {}".format(th_init.sum()))

        self.seg_gen = traj_segment_generator_ma(pi, wrapped_env, self._params['sampling']['timesteps_per_batch'],
                                                 stochastic=True)

        self.gamma = self._params['trpo']['gamma']
        self.lam = self._params['trpo']['lambda']
        self.max_kl = self._params['trpo']['max_kl']

        self.vf_iters = self._params['trpo']['vf_iterations']
        self.vf_stepsize = self._params['trpo']['vf_step_size']

        self.cg_iters = self._params['trpo']['cg_iterations']
        self.cg_damping = self._params['trpo']['cg_damping']

    def iterate(self, config: dict, rep: int, n: int):
        # set random seed for repetition and iteration
        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        time_update_start = time.time()

        if self._params['sampling']['num_objects'] == 'random':
            num_objects = random.randint(1, len(config['env_config'].objects))
            self.env.num_objects = num_objects
            logger.debug('using env with {} objects'.format(num_objects))

        with self.timed("sampling"):
            seg = self.seg_gen.__next__()
        add_vtarg_and_adv_ma(seg, self.gamma, self.lam)

        ob = np.concatenate([s['ob'] for s in seg], axis=0)
        ac = np.concatenate([s['ac'] for s in seg], axis=0)
        advantage_target = np.concatenate([s['adv'] for s in seg], axis=0)
        td_lambda_return = np.concatenate([s['tdlamret'] for s in seg], axis=0)
        # predicted value function before udpate
        v_prediction_before = np.concatenate([s["vpred"] for s in seg], axis=0)
        # standardized advantage function estimate
        advantage_target = (advantage_target - advantage_target.mean()) / advantage_target.std()

        # if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        # if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = ob, ac, advantage_target
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
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            full_step = step_direction / lm
            expected_improvement = g.dot(full_step)
            surrbefore = loss_before[0]
            step_size = 1.0
            theta_before = self.get_flat()
            theta_new = None
            mean_losses = None
            for _ in range(10):
                theta_new = theta_before + full_step * step_size
                self.set_from_flat(theta_new)
                mean_losses = surr, kl, *_ = self.allmean(np.array(self.compute_losses(*args)))
                improve = surr - surrbefore
                logger.info("Expected: {}".format(expected_improvement))
                logger.info("Actual: {}".format(improve))
                if not np.isfinite(mean_losses).all():
                    logger.info("Got non-finite value of losses -- bad!")
                elif kl > self.max_kl * 1.5:
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
                paramsums = self._COMM.allgather((theta_new.sum(), self.vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        with self.timed("vf"):
            for _ in range(self.vf_iters):
                for (mbob, mbret) in dataset.iterbatches((ob, td_lambda_return),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = self.allmean(self.compute_vflossandgrad(mbob, mbret))
                    self.vfadam.update(g, self.vf_stepsize)
        vf_loss = self.vf_loss(ob, td_lambda_return)

        vf_loss_mean = np.mean(self._COMM.allgather(vf_loss))
        # len_ret_local = (seg["ep_lens"], seg["ep_rets"]) # local values
        len_ret_local = (seg[0]["ep_lens"], seg[0]["ep_rets"])  # local values
        len_ret_pairs = self._COMM.allgather(len_ret_local)  # list of tuples
        lens, rews = map(flatten_lists, zip(*len_ret_pairs))
        self.lenbuffer.extend(lens)
        self.rewbuffer.extend(rews)

        self.episodes_so_far += len(lens)
        self.timesteps_so_far += sum(lens)
        self.iters_so_far += 1

        if self._COMM.rank == 0:
            time_elapsed = time.time() - time_update_start

            return_values = dict(zip(self.loss_names, mean_losses))

            return_values.update({
                "ev_tdlam_before":         explained_variance(v_prediction_before, td_lambda_return),
                "vf_loss":                 vf_loss_mean,
                "episode_reward_mean":     np.mean(self.rewbuffer),
                "episode_length_mean":     np.mean(self.lenbuffer),
                "episodes_this_iteration": len(lens),
                "episodes":                self.episodes_so_far,
                "timesteps":               self.timesteps_so_far,
                "frames_per_second":       int(np.sum(lens) / time_elapsed)
            })

            important_values = ["episode_reward_mean"]

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
        tf.reset_default_graph()

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
