import random

import gym_kilobots
import numpy as np
from gym import spaces
from gym_kilobots.envs import YamlKilobotsEnv, DirectControlKilobotsEnv


class TargetArea:
    def __init__(self, center, width, height):
        assert width > .0, 'width needs to be positive'
        assert height > .0, 'height needs to be positive'

        self.__center = np.asarray(center)
        self.__width = width
        self.__height = height

        self.__lower_bound = np.array((-width/2, -height/2))
        self.__upper_bound = -self.__lower_bound

    @property
    def center(self):
        return self.__center

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    def __contains__(self, position):
        vec = position - self.__center
        return np.all(np.abs(vec) <= (self.__width/2, self.__height/2))


class MultiObjectEnv(YamlKilobotsEnv):
    _observe_object = True

    def __init__(self, *, configuration, done_after_steps=350, **kwargs):

        self._target_area = TargetArea((.0, .0), .2, .2)
        self._done_after_steps = done_after_steps
        self._num_objects = None

        super(MultiObjectEnv, self).__init__(configuration=configuration, **kwargs)

    @property
    def num_objects(self):
        return len(self.objects)

    @num_objects.setter
    def num_objects(self, n_obj):
        if isinstance(n_obj, 'str'):
            self._num_objects = 'random'
        self._num_objects = max(min(n_obj, len(self.conf.objects)), 1)

    def reset(self):
        return super(MultiObjectEnv, self).reset()

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                               + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {}

    def get_observation(self):
        observation = np.concatenate(tuple(k.get_position() for k in self._kilobots))

        for obj in self._objects:
            _object_orientation = obj.get_orientation()
            observation = np.r_[observation, obj.get_position(), np.sin(_object_orientation),
                                np.cos(_object_orientation)]

        observation = np.r_[observation, self._light.get_state()]

        return observation

    def _quadratic_cost(self, state, target, weights=1., normalize=True):
        diff = state - target

        if normalize:
            diff = diff / np.asarray(self.world_size)

        return -np.sum(diff**2 * weights, axis=1)

    def get_reward(self, old_state, action, new_state):
        obj_dims = self.object_state_space.shape[0]
        # kb_dims = self.kilobots_state_space.shape[0]

        # old_kb_state = old_state[:kb_dims].reshape((-1, 2))
        # new_kb_state = new_state[:kb_dims].reshape((-1, 2))

        old_obj_state = old_state[-obj_dims:]
        old_obj_state = np.c_[old_obj_state[0::3], old_obj_state[1::3]]
        old_finished_objs = [o in self._target_area for o in old_obj_state]
        old_obj_rew = self._quadratic_cost(old_obj_state, self._target_area.center, weights=20)
        old_obj_rew[old_finished_objs] = 1.

        new_obj_state = new_state[-obj_dims:]
        new_obj_state = np.c_[new_obj_state[0::3], new_obj_state[1::3]]
        new_finished_objs = [o in self._target_area for o in new_obj_state]
        new_obj_rew = self._quadratic_cost(new_obj_state, self._target_area.center, weights=20)
        new_obj_rew[new_finished_objs] = 1.

        obj_rew_diff = np.sum(new_obj_rew - old_obj_rew) * 100

        kb_rew_diff = .0
        # for o_old, o_new in zip(old_obj_state, new_obj_state):
        #     old_kb_o_rew = self._quadratic_reward(old_kb_state, o_old, weights=10, normalize=False)
        #     new_kb_o_rew = self._quadratic_reward(new_kb_state, o_new, weights=10, normalize=False)
        #     kb_rew_diff += np.sum(new_kb_o_rew - old_kb_o_rew) * 20

        return obj_rew_diff + kb_rew_diff

        # reward = .0
        # for obj in self._objects:
        #     if obj.get_position() in self._target_area:
        #         reward += 100.
        #
        # return reward / len(self._objects)

        # reward = 0
        # for i, obj in enumerate(self._objects):
        #     obj_dist = obj.get_position() / np.asarray(self.world_size) - self._target_area.center
        #     # quadratic reward
        #     # reward += -(obj_dist * .1).dot(obj_dist)
        #     # exponential reward
        #     reward += np.exp(-(obj_dist * 20).dot(obj_dist)) * 100
        #
        # # compute difference of reward as suggested by Riad
        # if self._last_reward:
        #     _lr = self._last_reward
        #     self._last_reward = reward
        #     return reward - _lr
        # else:
        #     self._last_reward = reward
        #     return .0

    def _init_objects(self):
        if self._num_objects == 'random':
            num_objs = random.randint(1, len(self.conf.objects))
            for o in self.conf.objects[:num_objs]:
                self._init_object(o.shape, o.width, o.height, o.init)
        else:
            for o in self.conf.objects[:self._num_objects]:
                self._init_object(o.shape, o.width, o.height, o.init)

    def has_finished(self, state, action):
        if np.all([o.get_position() in self._target_area for o in self._objects]) \
                and self._sim_steps >= self._steps_per_action:
            return True
        return self._sim_steps >= self._done_after_steps * self._steps_per_action

    def _draw_on_table(self, screen):
        w = self._target_area.width
        h = self._target_area.height
        vertices = np.array([[-w, -h], [-w, h], [w, h], [w, -h]]) / 2 + self._target_area.center
        screen.draw_polygon(vertices, color=(0.5, 0.5, 0.5), filled=False, width=.005)


class MultiObjectDirectControlEnv(DirectControlKilobotsEnv, MultiObjectEnv):
    def __init__(self, **kwargs):
        super(MultiObjectDirectControlEnv, self).__init__(**kwargs)

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + tuple(o.get_pose() for o in self._objects))

    def get_observation(self):
        # observations as bearing angle + distance
        # create local observations for each agent
        kb_states = np.array([[k.get_state()] for k in self._kilobots])
        # relative positions
        kb_rel_positions = -kb_states[..., :2] + kb_states[..., :2].reshape(1, -1, 2)
        # relative polar coordinates
        kb_rel_radius = np.linalg.norm(kb_rel_positions, axis=2, keepdims=True)
        kb_rel_angle = np.arctan2(kb_rel_positions[..., [1]], kb_rel_positions[..., [0]])
        kb_rel_angle -= kb_states[..., [2]]
        # local orientations
        kb_rel_orientations = -kb_states[..., [2]] + kb_states[..., 2].reshape(1, -1, 1)
        # absolute velocities
        kb_vel = np.tile(kb_states[..., 3:].reshape(1, -1, 2), (self.num_kilobots, 1, 1))

        # concat swarm observations
        A = np.concatenate((kb_rel_radius, np.sin(kb_rel_angle), np.cos(kb_rel_angle),
                            np.sin(kb_rel_orientations), np.cos(kb_rel_orientations), kb_vel), axis=2)

        # remove diagonal entries, i.e., self observations
        strided = np.lib.stride_tricks.as_strided
        m, _, d = A.shape
        s0, s1, s2 = A.strides
        A = strided(A.ravel()[d:], shape=(m - 1, m, d), strides=(s0 + s1, s1, s2)).reshape(m, m - 1, d)

        # reshape to two dimensional matrix
        A = A.reshape(self.num_kilobots, -1)

        # object observations
        obj_states = np.array([[o.get_pose() for o in self._objects]])
        # relative positions
        obj_rel_positions = -kb_states[..., :2].reshape(-1, 1, 2) + obj_states[..., :2]
        # relative polar coordinates
        obj_rel_radius = np.linalg.norm(obj_rel_positions, axis=2, keepdims=True)
        obj_rel_angle = np.arctan2(obj_rel_positions[..., [1]], obj_rel_positions[..., [0]])
        obj_rel_angle -= kb_states[..., [2]]
        # relative orientations
        obj_rel_orientations = -kb_states[..., [2]] + obj_states[..., 2].reshape(1, -1, 1)
        obj_valid = np.ones((self.num_kilobots, len(self._objects), 1))

        B = np.concatenate((obj_rel_radius, np.sin(obj_rel_angle), np.cos(obj_rel_angle),
                            np.sin(obj_rel_orientations), np.cos(obj_rel_orientations), obj_valid), axis=2)
        B = B.reshape(self.num_kilobots, -1)

        # append zeros if env has less objects than defined in conf
        if len(self._objects) < len(self.conf.objects):
            num_missing_objects = len(self.conf.objects) - len(self._objects)
            B = np.concatenate((B, np.zeros((self.num_kilobots, 6 * num_missing_objects))), axis=1)

        # relative target position
        target_rel_position = (-kb_states[..., :2] + self._target_area.center).squeeze()
        # relative polar coordinates
        target_rel_radius = np.linalg.norm(target_rel_position, axis=1, keepdims=True)
        target_rel_angle = np.arctan2(target_rel_position[:, 1], target_rel_position[:, 0]).reshape(-1, 1)
        target_rel_angle -= kb_states[..., 2]

        # proprioceptive observations
        kb_proprioception = kb_states.squeeze()

        return np.concatenate((A, B, target_rel_radius, np.sin(target_rel_angle), np.cos(target_rel_angle),
                               kb_proprioception, np.full((self.num_kilobots, 1), len(self.objects))), axis=1)

    def get_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        kb_state = state[:kb_dims].reshape((-1, 1, 2))

        # get object positions in a (m x 2) matrix
        obj_state = state[-obj_dims:]
        obj_state = np.c_[obj_state[0::3], obj_state[1::3]]
        # index which objects are in target area
        finished_objs = [o_pos in self._target_area for o_pos in obj_state]

        # cost for objects
        # obj_cost = self._quadratic_cost(obj_state, self._target_area.center, weights=20)
        # negative norm of distance to target
        obj_cost = -np.linalg.norm(obj_state - self._target_area.center, axis=1)
        # set cost to .0 for objects in target area
        obj_cost[finished_objs] = .0
        # sum up costs
        obj_cost = np.sum(obj_cost)

        # cost for kilobots
        # norm of distance between all kilobots and all objects
        kb_obj_dist = np.linalg.norm(kb_state - obj_state.reshape(1, -1, 2), axis=2)
        # sum over negative min distance for each agent
        kb_obj_cost = np.sum(-1 * np.min(kb_obj_dist, axis=1))

        # kb_obj_cost = np.sum(-1. + np.exp(-kb_obj_dist * .2))

        return obj_cost + kb_obj_cost

    @property
    def kilobots_observation_space(self):
        kb_low = self._kilobots[0].state_space.low
        kb_high = self._kilobots[0].state_space.high

        # radius, angle as sin+cos, orientation as sin+cos
        kb_low = np.r_[[.0, -1., -1., -1., -1.], kb_low[-2:]]
        kb_high = np.r_[[np.sqrt(-self.world_width * -self.world_height), 1., 1., 1., 1.], kb_high[-2:]]
        kb_low = np.tile(kb_low, self.num_kilobots - 1)
        kb_high = np.tile(kb_high, self.num_kilobots - 1)

        return spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)

    @property
    def object_observation_space(self):
        # radius, angle as sin+cos, orientation as sin+cos
        obj_low = np.array([.0, -1., -1., -1., -1., 0] * len(self.conf.objects))
        obj_high = np.array([np.sqrt(-self.world_width * -self.world_height), 1., 1., 1., 1., 1
                             ] * len(self.conf.objects))

        return spaces.Box(low=obj_low, high=obj_high, dtype=np.float64)

    @property
    def observation_space(self):
        # radius, angle as sin+cos
        target_low = np.array([.0, -1., -1.])
        target_high = np.array([np.sqrt(-self.world_width * -self.world_height), 1., 1.])

        proprio_low = self._kilobots[0].state_space.low
        proprio_high = self._kilobots[0].state_space.high

        obs_space_low = np.r_[self.kilobots_observation_space.low, self.object_observation_space.low,
                              target_low, proprio_low, 1]
        obs_space_high = np.r_[self.kilobots_observation_space.high, self.object_observation_space.high,
                               target_high, proprio_high, len(self.conf.objects)]

        return spaces.Box(np.tile(obs_space_low, (self.num_kilobots, 1)),
                          np.tile(obs_space_high, (self.num_kilobots, 1)), dtype=np.float64)

    def _init_kilobots(self, type='SimpleDirectControlKilobot'):
        num_kilobots = self.conf.kilobots.num

        # draw the kilobots positions uniformly from the world size
        kilobot_positions = np.random.rand(num_kilobots, 2) * np.asarray(self.world_size) + self.world_bounds[0]
        kilobot_orientations = np.random.rand(num_kilobots) * np.pi * 2 - np.pi

        # assert for each kilobot that it is within the world bounds and add kilobot to the world
        for position, orientation in zip(kilobot_positions, kilobot_orientations):
            position = np.maximum(position, self.world_bounds[0] + 0.02)
            position = np.minimum(position, self.world_bounds[1] - 0.02)
            kb_class = getattr(gym_kilobots.lib, type)
            self._add_kilobot(kb_class(self.world, position=position, orientation=orientation,
                                       velocity=[.01, .0]))
