import abc
import multiprocessing
from typing import Tuple, List

import numpy as np
import pandas as pd

import gym
import kb_learning.envs as kb_envs


class KilobotSampler:
    _VERBOSE = False

    def __init__(self, num_episodes: int, num_steps_per_episode: int, num_kilobots: int, column_index: pd.Index,
                 *args, **kwargs):
        self.num_episodes = num_episodes
        self.num_steps_per_episode = num_steps_per_episode

        self.num_kilobots = num_kilobots

        self.column_index = column_index

    @abc.abstractmethod
    def _sample_sars(self) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError

    def __call__(self):
        sars_samples, info = self._sample_sars()

        index = pd.MultiIndex.from_product([range(self.num_episodes), range(self.num_steps_per_episode)])
        it_sars = pd.DataFrame(data=sars_samples, index=index, columns=self.column_index)
        return it_sars, info


class QuadPushingSampler(KilobotSampler):
    def __init__(self, num_episodes: int, num_steps_per_episode: int, num_kilobots: int,
                 column_index: pd.Index, w_factor: float, policy=None, *args, **kwargs):
        super().__init__(num_episodes, num_steps_per_episode, num_kilobots, column_index, *args, **kwargs)

        self.w_factor = w_factor

        self.policy = policy

        env_id = kb_envs.get_quadpushing_environment(weight=w_factor, num_kilobots=num_kilobots)
        self.envs = [gym.make(env_id) for _ in range(num_episodes)]

    def set_policy(self, policy):
        self.policy = policy

    def _sample_sars(self):
        # reset environments and obtain initial states
        states = np.array([e.reset() for e in self.envs])
        reward = np.empty((self.num_episodes, 1))
        info = list()

        state_dims = states.shape[1]
        action_dims = sum(self.envs[0].action_space.shape)

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


class ParallelQuadPushingSampler(QuadPushingSampler):
    envs = []

    def __init__(self, num_episodes: int, num_steps_per_episode: int,
                 num_kilobots: int, w_factor: float, column_index: pd.Index, policy=None, num_workers=None):
        super().__init__(num_episodes, num_steps_per_episode, num_kilobots, column_index, w_factor, policy)

        self.num_workers = num_workers
        episodes_per_worker = num_episodes // self.num_workers
        self.pool = multiprocessing.Pool(self.num_workers, initializer=ParallelQuadPushingSampler._init_worker,
                                         initargs=[w_factor, num_kilobots, episodes_per_worker])

    @staticmethod
    def _init_worker(w_factor, num_kilobots, num_environments):
        env_id = kb_envs.get_quadpushing_environment(weight=w_factor, num_kilobots=num_kilobots)
        ParallelQuadPushingSampler.envs = [gym.make(env_id) for _ in range(num_environments)]

    @staticmethod
    def _do_work(policy, num_episodes, num_steps):
        # reset environments and obtain initial states
        states = np.array([e.reset() for e in ParallelQuadPushingSampler.envs[:num_episodes]])
        reward = np.empty((num_episodes, 1))
        info = list()

        state_dims = states.shape[1]
        action_dims = sum(ParallelQuadPushingSampler.envs[0].action_space.shape)

        it_sars_data = np.empty((num_episodes * num_steps, 2 * state_dims + action_dims + 1))

        for step in range(num_steps):
            # import sys
            # sys.stdout.write('.')
            # sys.stdout.flush()

            it_sars_data[step::num_steps, :state_dims] = states

            actions = policy(states)
            srdi = [e.step(a) for e, a in zip(ParallelQuadPushingSampler.envs[:num_episodes], actions)]
            for i in range(num_episodes):
                states[i, :] = srdi[i][0]
                reward[i] = srdi[i][1]
                info.append(srdi[i][3])

            # collect samples in DataFrame
            it_sars_data[step::num_steps, state_dims:] = np.c_[actions, reward, states]

        return it_sars_data, info

    def _sample_sars(self):
        episodes_per_work = [self.num_episodes // self.num_workers] * self.num_workers
        episodes_per_work.append(self.num_episodes % self.num_workers)  # the last bit is for the fastest
        num_work = len(episodes_per_work)

        work = list(zip([self.policy]*num_work, episodes_per_work, [self.num_steps_per_episode]*num_work))

        results = self.pool.starmap(ParallelQuadPushingSampler._do_work, work)

        # combine results
        it_sars_data = results[0][0]
        it_info = results[0][1]

        for sars_i, info_i in results[1:]:
            it_sars_data = np.concatenate((it_sars_data, sars_i))
            it_info += info_i

        return it_sars_data, it_info