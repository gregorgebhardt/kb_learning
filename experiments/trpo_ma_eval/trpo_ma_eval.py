import logging
import os
import random
from typing import Generator

import numpy as np
import pandas as pd
import tensorflow as tf
from baselines.common import tf_util
from cluster_work import ClusterWork

from kb_learning.envs import MultiObjectDirectControlEnv, NormalizeActionWrapper
from kb_learning.policy_networks.swarm_policy import SwarmPolicyNetwork
from kb_learning.tools.trpo_tools import ActWrapper, traj_segment_generator_ma

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

logger = logging.getLogger('trpo')


class TRPOMultiAgentEvaluation(ClusterWork):
    _default_params = {
        'num_objects':     8,
        'num_kilobots':    10,
        'reward_function': None,
        'policy_path': None,
        'render': False
    }

    def __init__(self):
        super().__init__()

        self.graph = None
        self.session = None

        self.env = None
        self.pi = None
        self.seg_gen = None

    def reset(self, config: dict, rep: int):
        os.environ["OMP_NUM_THREADS"] = '2'

        sess_conf = tf.ConfigProto(allow_soft_placement=True,
                                   inter_op_parallelism_threads=2,
                                   intra_op_parallelism_threads=1)
        self.graph = tf.Graph()
        self.session = tf_util.make_session(config=sess_conf, make_default=True, graph=self.graph)
        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        env_config = config['env_config']
        from itertools import cycle, islice
        env_config.objects = list(islice(cycle(env_config.objects),
                                         self._params['num_objects']))

        env_config.kilobots.num = self._params['num_kilobots']
        self.env = MultiObjectDirectControlEnv(configuration=config['env_config'],
                                               reward_function=self._params['reward_function'],
                                               agent_type='SimpleVelocityControlKilobot',
                                               done_after_steps=512, )

        # render video path
        self.env.video_path = self._log_path_rep
        self.env.render_mode = 'array'
        wrapped_env = NormalizeActionWrapper(self.env)

        def policy_fn(name, **_policy_params):
            return SwarmPolicyNetwork(name=name, **_policy_params)

        update_params = dict(
            ob_space=self.env.observation_space,
            num_agent_observations=self.env.num_kilobots - 1,
            num_object_observations=len(env_config.objects),
        )
        policy_path = self._params['policy_path']
        self.pi = ActWrapper.load(policy_path, policy_fn, update_params=update_params)

        self.seg_gen = traj_segment_generator_ma(self.pi, wrapped_env, 512, False, render=self._params['render'])

    def iterate(self, config: dict, rep: int, n: int):
        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        random.seed(self._seed)

        # create next trajectories
        seg = self.seg_gen.__next__()

        # evaluation
        results = {
            'mean_reward': np.mean(seg[0]['rew']),
            'return':      np.sum(seg[0]['rew']),
        }

        # return results
        return results

    def finalize(self):
        if self.env:
            self.env.close()
        self.session.close()
        del self.graph

    @classmethod
    def plot_results(cls, configs_results: Generator):
        pass


if __name__ == '__main__':
    TRPOMultiAgentEvaluation.run()
