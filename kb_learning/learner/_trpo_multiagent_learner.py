import logging
import os
import random
import time
from collections import deque
from contextlib import contextmanager
from typing import Generator

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


class TRPOMultiAgentLearner(ClusterWork):
    _restore_supported = True
    _default_params = {
        'sampling': {
            'timesteps_per_batch': 2048,
            'done_after_steps':    1024
        },
        'trpo':     {
            'gamma':         0.99,
            'lambda':        0.98,
            'max_kl':        0.01,
            'entcoeff':      0.0,
            'cg_iterations': 10,
            'cg_damping':    1e-1,
            'vf_step_size':  1e-3,
            'vf_iterations': 5,
        },
        'policy':   {
            'swarm_size':   (64,),
            'objects_size': (64,),
            'hidden_size':  (64,)
        },
    }

    def __init__(self):
        super().__init__()
        self.num_workers = None
        self.rank = None

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
        self._comm.Allreduce(x, out, op=MPI.SUM)
        out /= self.num_workers
        return out

    def reset(self, config: dict, rep: int):
        os.environ["OMP_NUM_THREADS"] = '2'

        sess_conf = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=2,
                                   intra_op_parallelism_threads=1)
        tf_util.make_session(config=sess_conf, make_default=True)

        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        if not self._comm:
            self._comm = MPI.COMM_WORLD

        # create env
        env = MultiObjectDirectControlEnv(configuration=config['env_config'],
                                          done_after_steps=self._params['sampling']['done_after_steps'])

        wrapped_env = NormalizeActionWrapper(env)

        # create policy function
        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                             # we observe all other kilobots with (x, y, sin(th), cos(th), lin_vel, rot_vel)
                             num_observed_kilobots=env.num_kilobots - 1, kilobot_dims=6,
                             # we observe all objects with (x, y, sin(th), cos(th))
                             num_observed_objects=len(env.objects), object_dims=4,
                             hid_size=self._params['policy']['hidden_size'],
                             kb_me_size=self._params['policy']['swarm_size'],
                             ob_me_size=self._params['policy']['objects_size'])

        self.policy_fn = policy_fn

        self.num_workers = self._comm.Get_size()
        self.rank = self._comm.Get_rank()
        logger.debug('initializing with rank {} of {}'.format(self.rank, self.num_workers))
        np.set_printoptions(precision=3)
        # Setup losses and stuff
        # ----------------------------------------
        pi = policy_fn("pi", env.observation_space, env.kilobots[0].action_space)
        oldpi = policy_fn("oldpi", env.observation_space, env.kilobots[0].action_space)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        ob = tf_util.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = self._params['trpo']['entcoeff'] * meanent

        vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        surrgain = tf.reduce_mean(ratio * atarg)

        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

        dist = meankl

        all_var_list = pi.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        var_list.extend([v for v in all_var_list if v.name.split("/")[1].startswith("me")])
        vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        self.vfadam = MpiAdam(vf_var_list, comm=self._comm)

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

        act_params = {
            'name':     "pi",
            'ob_space': env.observation_space,
            'ac_space': env.kilobots[0].action_space,
        }

        self.pi = ActWrapper(pi, act_params)

        tf_util.initialize()
        th_init = self.get_flat()
        self._comm.Bcast(th_init, root=0)
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

        with self.timed("sampling"):
            seg = self.seg_gen.__next__()
        add_vtarg_and_adv_ma(seg, self.gamma, self.lam)

        ob = np.concatenate([s['ob'] for s in seg], axis=0)
        ac = np.concatenate([s['ac'] for s in seg], axis=0)
        atarg = np.concatenate([s['adv'] for s in seg], axis=0)
        tdlamret = np.concatenate([s['tdlamret'] for s in seg], axis=0)
        vpredbefore = np.concatenate([s["vpred"] for s in seg], axis=0)  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        # if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        # if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = ob, ac, atarg
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return self.allmean(self.compute_fvp(p, *fvpargs)) + self.cg_damping * p

        self.assign_old_eq_new()  # set old parameter values to new parameter values
        with self.timed("computegrad"):
            *lossbefore, g = self.compute_lossandgrad(*args)
        lossbefore = self.allmean(np.array(lossbefore))
        g = self.allmean(g)
        if np.allclose(g, 0):
            logger.info("Got zero gradient. not updating")
        else:
            with self.timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                meanlosses = surr, kl, *_ = self.allmean(np.array(self.compute_losses(*args)))
                improve = surr - surrbefore
                logger.info("Expected: {} Actual: {}".format(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.info("Got non-finite value of losses -- bad!")
                elif kl > self.max_kl * 1.5:
                    logger.info("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.info("surrogate didn't improve. shrinking step.")
                else:
                    logger.info("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.info("couldn't compute a good step")
                self.set_from_flat(thbefore)
            if self.num_workers > 1 and self.iters_so_far % 20 == 0:
                paramsums = self._comm.allgather((thnew.sum(), self.vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        with self.timed("vf"):
            for _ in range(self.vf_iters):
                for (mbob, mbret) in dataset.iterbatches((ob, tdlamret),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = self.allmean(self.compute_vflossandgrad(mbob, mbret))
                    self.vfadam.update(g, self.vf_stepsize)

        # lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        lrlocal = (seg[0]["ep_lens"], seg[0]["ep_rets"])  # local values
        listoflrpairs = self._comm.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        self.lenbuffer.extend(lens)
        self.rewbuffer.extend(rews)

        self.episodes_so_far += len(lens)
        self.timesteps_so_far += sum(lens)
        self.iters_so_far += 1

        if self._comm.rank == 0:
            time_elapsed = time.time() - time_update_start

            return_values = dict(zip(self.loss_names, meanlosses))

            return_values.update({
                "ev_tdlam_before":         explained_variance(vpredbefore, tdlamret),
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
                    logger.info(colorize("{}: {}".format(k, v), color='crimson', bold=True))
                else:
                    logger.info("{}: {}".format(k, v))

            return return_values

    def finalize(self):
        pass

    def save_state(self, config: dict, rep: int, n: int):
        # save parameters at every iteration
        policy_path = os.path.join(self._log_path_it, 'policy.pkl')
        self.pi.save(policy_path)

    def restore_state(self, config: dict, rep: int, n: int):
        # resore parameters
        policy_path = os.path.join(self._log_path_it, 'policy.pkl')
        logger.info('Restoring parameters from {}'.format(policy_path))
        self.pi = ActWrapper.load(policy_path, self.policy_fn)

        return True

    @classmethod
    def plot_results(cls, configs_results: Generator):
        pass
