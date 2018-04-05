import abc
import multiprocessing
from typing import Tuple, List

import numpy as np
import pandas as pd

import gym
import logging

logger = logging.getLogger('kb_learning.sampler')


class KilobotSampler(object):
    def __init__(self, num_episodes: int, num_steps_per_episode: int,
                 sars_column_index: pd.Index, state_column_index: pd.Index, seed: int=0, *args, **kwargs):
        self._seed = seed

        self.max_episodes = num_episodes
        self.num_steps_per_episode = num_steps_per_episode

        self.sars_column_index = sars_column_index
        self.state_column_index = state_column_index

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @abc.abstractmethod
    def _sample_sars(self, policy, num_episodes: int, num_steps_per_episode: int) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError

    def __call__(self, policy, num_episodes: int = None, num_steps_per_episode: int = None):
        if num_episodes is None:
            num_episodes = self.max_episodes
        if num_steps_per_episode is None:
            num_steps_per_episode = self.num_steps_per_episode

        if num_episodes > self.max_episodes:
            num_episodes = self.max_episodes
            logger.warning('num_episodes > max_episodes! Setting num_episodes to max_episodes')

        sars_samples, info = self._sample_sars(policy, num_episodes, num_steps_per_episode)

        index = pd.MultiIndex.from_product([range(num_episodes), range(num_steps_per_episode)])
        it_sars = pd.DataFrame(data=sars_samples, index=index, columns=self.sars_column_index)
        it_info = pd.DataFrame(data=info, index=index, columns=self.state_column_index)
        return it_sars, it_info


class ObjectEnvSampler(KilobotSampler):
    def __init__(self, registration_function, w_factor=.0, num_kilobots=15, object_shape='quad',
                 object_width=.15, object_height=.15, light_type='circular', light_radius=.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_factor = w_factor
        self.num_kilobots = num_kilobots

        self.object_shape = object_shape
        self.object_width = object_width
        self.object_height = object_height

        self.light_type = light_type
        self.light_radius = light_radius

        self.registration_function = registration_function

        self.env_id = self._get_env_id()
        # create an prototype of the environment
        self.env = gym.make(self.env_id)
        self.envs = []

    def _init_envs(self, num_episodes):
        self.envs = [gym.make(self.env_id) for _ in range(num_episodes)]

        for i, e in enumerate(self.envs):
            e.seed(self.seed * 1000 + i)

    def _get_env_id(self):
        return self.registration_function(weight=self.w_factor, num_kilobots=self.num_kilobots,
                                          object_shape=self.object_shape, object_width=self.object_width,
                                          object_height=self.object_height, light_type=self.light_type,
                                          light_radius=self.light_radius)


class SARSSampler(ObjectEnvSampler):
    def _sample_sars(self, policy, num_episodes, num_steps_per_episode):
        if len(self.envs) != num_episodes:
            self._init_envs(num_episodes)

        # reset environments and obtain initial states
        states = np.array([e.reset() for e in self.envs])
        reward = np.empty((num_episodes, 1))
        info = list()

        observation_dims = states.shape[1]
        action_dims = sum(self.env.action_space.shape)
        state_dims = sum(sum(s.shape) for s in self.envs[0].state_space.spaces)

        it_sars_data = np.empty((num_episodes * num_steps_per_episode, 2 * observation_dims + action_dims + 1))
        it_info_data = np.empty((num_episodes * num_steps_per_episode, state_dims))

        for step in range(num_steps_per_episode):
            it_sars_data[step::num_steps_per_episode, :observation_dims] = states
            info.clear()

            actions = policy(states)
            srdi = [e.step(a) for e, a in zip(self.envs, actions)]

            for i in range(num_episodes):
                states[i, :] = srdi[i][0]
                reward[i] = srdi[i][1]
                info.append(srdi[i][3])

            # collect samples into matrix
            it_sars_data[step::num_steps_per_episode, observation_dims:] = np.c_[actions, reward, states]
            # collect environment information into matrix
            it_info_data[step::num_steps_per_episode, :] = np.array(info)

        return it_sars_data, it_info_data


envs = []


def _init_worker(registration_function, num_environments, w_factor, num_kilobots, object_shape, object_width,
                 object_height, light_type, light_radius):
    global envs
    env_id = registration_function(weight=w_factor, num_kilobots=num_kilobots, object_shape=object_shape,
                                   object_width=object_width, object_height=object_height, light_type=light_type,
                                   light_radius=light_radius)
    envs = [gym.make(env_id) for _ in range(num_environments)]


def _set_worker_seed(seed, work_seed):
    global envs
    for i, e in enumerate(envs):
        e.seed(work_seed * 10000 + seed * 500 + i)
    np.random.seed(work_seed * 10000 + seed * 400 + 12346)


def _do_work(policy, num_episodes, num_steps, seed, work_seed):
    global envs
    _set_worker_seed(seed, work_seed)
    # reset environments and obtain initial states
    states = np.array([e.reset() for e in envs[:num_episodes]])
    reward = np.empty((num_episodes, 1))
    info = list()

    observation_dims = states.shape[1]
    action_dims = sum(envs[0].action_space.shape)
    state_dims = sum(sum(s.shape) for s in envs[0].state_space.spaces)

    it_sars_data = np.empty((num_episodes * num_steps, 2 * observation_dims + action_dims + 1))
    it_info_data = np.empty((num_episodes * num_steps, state_dims))

    # do one additional step before
    actions = policy(states)
    srdi = [e.step(a) for e, a in zip(envs[:num_episodes], actions)]
    for i in range(num_episodes):
        states[i, :] = srdi[i][0]

    for step in range(num_steps):
        it_sars_data[step::num_steps, :observation_dims] = states
        info.clear()

        actions = policy(states)
        srdi = [e.step(a) for e, a in zip(envs[:num_episodes], actions)]
        for i in range(num_episodes):
            states[i, :] = srdi[i][0]
            reward[i] = srdi[i][1]
            info.append(srdi[i][3])

        # collect samples into matrix
        it_sars_data[step::num_steps, observation_dims:] = np.c_[actions, reward, states]
        # collect environment information into matrix
        it_info_data[step::num_steps, :] = np.array(info)

    return it_sars_data, it_info_data


class ParallelSARSSampler(SARSSampler):
    def __init__(self, num_workers: int=None, mp_context: str='forkserver', *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self._init_worker(w_factor, num_kilobots, 2)
        self._num_workers = num_workers
        self.__pool = None
        self._pool_timeout = 300
        self._num_restarts = 3
        if self._num_workers is None or self._num_workers <= 0:
            self._num_workers = multiprocessing.cpu_count()

        if self._num_workers > 1:
            self._episodes_per_worker = (self.max_episodes // self._num_workers) + 1

            # for the cluster it is necessary to use the context forkserver here, using a forkserver prevents the forked
            # processes from taking over handles to files and similar stuff
            self._context = multiprocessing.get_context(mp_context)
            self.__pool = self._create_pool()

    def __del__(self):
        del self._num_workers
        if self.__pool is not None:
            self.__pool.terminate()
            self.__pool.join()
            self.__pool.close()

            del self.__pool
            del self._episodes_per_worker

    def _create_pool(self) -> multiprocessing.Pool:
        return self._context.Pool(processes=self._num_workers, initializer=_init_worker,
                                  initargs=[self.registration_function, self._episodes_per_worker, self.w_factor,
                                            self.num_kilobots, self.object_shape, self.object_width,
                                            self.object_height, self.light_type, self.light_radius])

    def _sample_sars(self, policy, num_episodes, num_steps_per_episode):
        if self._num_workers == 1:
            return super(ParallelSARSSampler, self)._sample_sars(policy, num_episodes, num_steps_per_episode)
        else:
            episodes_per_work = [num_episodes // self._num_workers] * self._num_workers
            for i in range(num_episodes % self._num_workers):
                episodes_per_work[i] += 1
            episodes_per_work = list(filter(lambda a: a != 0, episodes_per_work))

            # construct work packages with policy, number of episodes, number of steps, seed and work-seed,
            # where work-seed is is taken from the range of work packages
            work = [(policy.to_dict(), episodes, num_steps_per_episode, self._seed, work_seed) for episodes, work_seed
                    in zip(episodes_per_work, range(len(episodes_per_work)))]

            for i in range(self._num_restarts):
                try:
                    results = self.__pool.starmap_async(_do_work, work, error_callback=self.__error_callback).get(
                        self._pool_timeout)
                    break
                except multiprocessing.TimeoutError:
                    logger.warning('got TimeoutError, restarting.')
                    # restart pool
                    self.__pool.terminate()
                    self.__pool.join()
                    self.__pool.close()

                    self.__pool = self._create_pool()
            else:
                # re-raise TimeoutError
                raise multiprocessing.TimeoutError

            # combine results
            it_sars_data = results[0][0]
            it_info_data = results[0][1]

            for sars_i, info_i in results[1:]:
                it_sars_data = np.concatenate((it_sars_data, sars_i))
                it_info_data = np.concatenate((it_info_data, info_i))

            return it_sars_data, it_info_data

    @staticmethod
    def __error_callback(exception: Exception):
        logger.error(exception)
