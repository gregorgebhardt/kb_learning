import abc
import multiprocessing
from typing import Tuple, List

import numpy as np
import pandas as pd

import gym
import kb_learning.envs as kb_envs
import logging

logger = logging.getLogger('kb_learning.sampler')


class KilobotSampler:
    def __init__(self, num_episodes: int, num_steps_per_episode: int,
                 column_index: pd.Index, seed: int=0, *args, **kwargs):
        self._seed = seed

        self.num_episodes = num_episodes
        self.num_steps_per_episode = num_steps_per_episode

        self.column_index = column_index

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @abc.abstractmethod
    def _sample_sars(self) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError

    def __call__(self):
        sars_samples, info = self._sample_sars()

        index = pd.MultiIndex.from_product([range(self.num_episodes), range(self.num_steps_per_episode)])
        it_sars = pd.DataFrame(data=sars_samples, index=index, columns=self.column_index)
        return it_sars, info


class QuadEnvSampler(KilobotSampler):
    def __init__(self, w_factor, num_kilobots, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_factor = w_factor
        self.num_kilobots = num_kilobots

        self.env_id = self._get_env_id()
        # create an prototype of the environment
        self.env = gym.make(self.env_id)
        self.envs = []

    def _init_envs(self):
        self.envs = [gym.make(self.env_id) for _ in range(self.num_episodes)]

        for i, e in enumerate(self.envs):
            e.seed(self.seed * 100 + i)

    @abc.abstractmethod
    def _get_env_id(self):
        raise NotImplementedError


class SARSSampler(QuadEnvSampler):
    def __init__(self, num_episodes: int, num_steps_per_episode: int, num_kilobots: int,
                 column_index: pd.Index, policy=None, seed: int=0, *args, **kwargs):
        super().__init__(num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                         num_kilobots=num_kilobots, column_index=column_index, seed=seed, *args, **kwargs)

        self.policy = policy

    @abc.abstractmethod
    def _get_env_id(self):
        raise NotImplementedError

    def _sample_sars(self):
        if len(self.envs) is 0:
            self._init_envs()

        for i, e in enumerate(self.envs):
            e.seed(self.seed * 100 + i)

        # reset environments and obtain initial states
        states = np.array([e.reset() for e in self.envs])
        reward = np.empty((self.num_episodes, 1))
        info = list()

        state_dims = states.shape[1]
        action_dims = sum(self.env.action_space.shape)

        it_sars_data = np.empty((self.num_episodes * self.num_steps_per_episode, 2 * state_dims + action_dims + 1))

        for step in range(self.num_steps_per_episode):
            it_sars_data[step::self.num_steps_per_episode, :state_dims] = states

            actions = self.policy(states)
            srdi = [e.step(a) for e, a in zip(self.envs, actions)]

            for i in range(self.num_episodes):
                states[i, :] = srdi[i][0]
                reward[i] = srdi[i][1]
                info.append(srdi[i][3])

            # collect samples in DataFrame
            it_sars_data[step::self.num_steps_per_episode, state_dims:] = np.c_[actions, reward, states]

        return it_sars_data, info


class FixedWeightQuadEnvSampler(SARSSampler):
    def _get_env_id(self):
        return kb_envs.get_fixed_weight_quad_env(weight=self.w_factor, num_kilobots=self.num_kilobots)


class SampleWeightQuadEnvSampler(SARSSampler):
    def _get_env_id(self):
        return kb_envs.get_sample_weight_quad_env(num_kilobots=self.num_kilobots)


envs = []
worker_seed = None


def _init_worker(env_id, num_environments):
    # env_id = kb_envs.get_quadpushing_environment(weight=w_factor, num_kilobots=num_kilobots)
    global envs, worker_seed
    envs = [gym.make(env_id) for _ in range(num_environments)]
    worker_seed = multiprocessing.current_process().pid


def _set_worker_seed(seed):
    global envs, worker_seed
    for i, e in enumerate(envs):
        e.seed(worker_seed * 10000 + seed * 500 + i)
    np.random.seed(worker_seed * 10000 + seed * 400 + 12346)


def _do_work(policy, num_episodes, num_steps, seed):
    global envs
    _set_worker_seed(seed)
    # reset environments and obtain initial states
    states = np.array([e.reset() for e in envs[:num_episodes]])
    reward = np.empty((num_episodes, 1))
    info = list()

    state_dims = states.shape[1]
    action_dims = sum(envs[0].action_space.shape)

    it_sars_data = np.empty((num_episodes * num_steps, 2 * state_dims + action_dims + 1))

    for step in range(num_steps):
        it_sars_data[step::num_steps, :state_dims] = states

        actions = policy(states)
        srdi = [e.step(a) for e, a in zip(envs[:num_episodes], actions)]
        for i in range(num_episodes):
            states[i, :] = srdi[i][0]
            reward[i] = srdi[i][1]
            info.append(srdi[i][3])

        # collect samples in DataFrame
        it_sars_data[step::num_steps, state_dims:] = np.c_[actions, reward, states]

    return it_sars_data, info


class ParallelSARSSampler(SARSSampler):
    def __init__(self, num_episodes: int, num_steps_per_episode: int, num_kilobots: int, w_factor: float,
                 column_index: pd.Index, policy=None, seed: int=0, num_workers=None):
        super().__init__(num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                         num_kilobots=num_kilobots, column_index=column_index, w_factor=w_factor,
                         policy=policy, seed=seed)

        # self._init_worker(w_factor, num_kilobots, 2)
        self.num_workers = num_workers
        self.episodes_per_worker = (num_episodes // self.num_workers) + 1

        # for the cluster it is necessary to use the context forkserver here, using a forkserver prevents the forked
        # processes from taking over handles to files and similar stuff
        ctx = multiprocessing.get_context('forkserver')
        self.pool = ctx.Pool(processes=self.num_workers, initializer=_init_worker,
                             initargs=[self.env_id, self.episodes_per_worker])
        # self.pool.map(_set_worker_seed, [self._seed] * self.num_workers)

    def __del__(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool.close()

    # @SARSSampler.seed.setter
    # def seed(self, seed):
    #     self._seed = seed
    #     self.pool.map(_set_worker_seed, [self._seed] * self.num_workers)

    @abc.abstractmethod
    def _get_env_id(self):
        raise NotImplementedError

    def _sample_sars(self):
        episodes_per_work = [self.num_episodes // self.num_workers] * self.num_workers
        for i in range(self.num_episodes % self.num_workers):
            episodes_per_work[i] += 1
        episodes_per_work = filter(lambda a: a != 0, episodes_per_work)

        work = [(self.policy, episodes, self.num_steps_per_episode, self._seed) for episodes in episodes_per_work]

        # self._do_work(*work[0])
        results = self.pool.starmap_async(_do_work, work).get(1200)

        # combine results
        it_sars_data = results[0][0]
        it_info = results[0][1]

        for sars_i, info_i in results[1:]:
            it_sars_data = np.concatenate((it_sars_data, sars_i))
            it_info += info_i

        return it_sars_data, it_info


class ParallelFixedWeightQuadEnvSampler(ParallelSARSSampler):
    def _get_env_id(self):
        return kb_envs.get_fixed_weight_quad_env(weight=self.w_factor, num_kilobots=self.num_kilobots)


class ParallelSampleWeightQuadEnvSampler(ParallelSARSSampler):
    def _get_env_id(self):
        return kb_envs.get_sample_weight_quad_env(num_kilobots=self.num_kilobots)
