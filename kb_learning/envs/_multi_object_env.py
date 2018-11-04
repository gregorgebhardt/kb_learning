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

        super(MultiObjectEnv, self).__init__(configuration=configuration, **kwargs)

    def reset(self):
        return super(MultiObjectEnv, self).reset()

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {}

    def get_observation(self):
        observation = np.concatenate(tuple(k.get_position() for k in self._kilobots)
                                     + (self._light.get_state(),))
        for obj in self._objects:
            _object_orientation = obj.get_orientation()
            observation = np.r_[observation, obj.get_position(), np.sin(_object_orientation),
                                np.cos(_object_orientation)]

        return observation

    def _quadratic_reward(self, state, target, weights=1., normalize=True):
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
        old_obj_rew = self._quadratic_reward(old_obj_state, self._target_area.center, weights=20)
        old_obj_rew[old_finished_objs] = 1.

        new_obj_state = new_state[-obj_dims:]
        new_obj_state = np.c_[new_obj_state[0::3], new_obj_state[1::3]]
        new_finished_objs = [o in self._target_area for o in new_obj_state]
        new_obj_rew = self._quadratic_reward(new_obj_state, self._target_area.center, weights=20)
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
        # create local observations for each agent
        kb_states = np.array([[k.get_state()] for k in self._kilobots])
        kb_rel_positions = -kb_states[..., :2] + kb_states[..., :2].reshape(1, -1, 2)
        kb_rel_orientations = -kb_states[..., [2]] + kb_states[..., 2].reshape(1, -1, 1)
        # todo absolute velocities or relative velocities?
        kb_vel = np.tile(kb_states[..., 3:].reshape(1, -1, 2), (self.num_kilobots, 1, 1))

        # concat swarm observations
        A = np.concatenate((kb_rel_positions, np.sin(kb_rel_orientations), np.cos(kb_rel_orientations), kb_vel), axis=2)

        # remove diagonal entries
        strided = np.lib.stride_tricks.as_strided
        m, _, d = A.shape
        s0, s1, s2 = A.strides
        A = strided(A.ravel()[d:], shape=(m - 1, m, d), strides=(s0 + s1, s1, s2)).reshape(m, m - 1, d)

        # reshape to two dimensional matrix
        A = A.reshape(self.num_kilobots, -1)

        # relative object observations
        obj_states = np.array([[o.get_pose() for o in self._objects]])
        obj_rel_positions = -kb_states[..., :2].reshape(-1, 1, 2) + obj_states[..., :2]
        obj_rel_orientations = -kb_states[..., [2]] + obj_states[..., 2].reshape(1, -1, 1)

        B = np.concatenate((obj_rel_positions, np.sin(obj_rel_orientations), np.cos(obj_rel_orientations)), axis=2)
        B = B.reshape(self.num_kilobots, -1)

        # relative target observation
        target_rel_position = (-kb_states[..., :2] + self._target_area.center).squeeze()

        # proprioceptive observations
        kb_proprioception = kb_states.squeeze()

        return np.concatenate((A, B, target_rel_position, kb_proprioception), axis=1)

    def get_reward(self, old_state, action, new_state):
        # TODO try other reward function (abs, sparse, normalization??)

        obj_dims = self.object_state_space.shape[0]
        # kb_dims = self.kilobots_state_space.shape[0]

        # old_kb_state = old_state[:kb_dims].reshape((-1, 2))
        # new_kb_state = new_state[:kb_dims].reshape((-1, 2))

        # old_obj_state = old_state[-obj_dims:]
        # old_obj_state = np.c_[old_obj_state[0::3], old_obj_state[1::3]]
        # old_finished_objs = [o_pos in self._target_area for o_pos in old_obj_state]
        # # old_obj_rew = self._quadratic_reward(old_obj_state, self._target_area.center, weights=20)
        # old_obj_rew = -np.abs(old_obj_state - self._target_area.center)
        # old_obj_rew[old_finished_objs] = .0

        # old_obj_rew[old_finished_objs] = 1.

        new_obj_state = new_state[-obj_dims:]
        new_obj_state = np.c_[new_obj_state[0::3], new_obj_state[1::3]]
        new_finished_objs = [o_pos in self._target_area for o_pos in new_obj_state]
        # new_obj_rew = np.exp(self._quadratic_reward(new_obj_state, self._target_area.center, weights=20))

        # new_obj_rew = - np.abs(new_obj_state - self._target_area.center)
        new_obj_rew = -np.linalg.norm(new_obj_state - self._target_area.center, axis=1, keepdims=True)
        new_obj_rew[new_finished_objs] = .0
        # obj_rew_diff = new_obj_rew - old_obj_rew

        # kb_rew_diff = .0
        # for o_old, o_new in zip(old_obj_state, new_obj_state):
        #     old_kb_o_rew = self._quadratic_reward(old_kb_state, o_old, weights=10, normalize=False)
        #     new_kb_o_rew = self._quadratic_reward(new_kb_state, o_new, weights=10, normalize=False)
        #     kb_rew_diff += np.sum(new_kb_o_rew - old_kb_o_rew) * 20

        return np.sum(new_obj_rew * 10) / len(self._objects)

    @property
    def kilobots_observation_space(self):
        kb_low = self._kilobots[0].state_space.low
        kb_high = self._kilobots[0].state_space.high

        kb_low = np.r_[[-self.world_width, -self.world_height, -1., -1.], kb_low[-2:]]
        kb_high = np.r_[[self.world_width, self.world_height, 1., 1.], kb_high[-2:]]
        kb_low = np.tile(kb_low, self.num_kilobots - 1)
        kb_high = np.tile(kb_high, self.num_kilobots - 1)

        return spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)

    @property
    def object_observation_space(self):
        obj_low = np.array([-self.world_width, -self.world_height, -1., -1.] * len(self._objects))
        obj_high = np.array([self.world_width, self.world_height, 1., 1.] * len(self._objects))

        return spaces.Box(low=obj_low, high=obj_high, dtype=np.float64)

    @property
    def observation_space(self):
        target_low = -np.asarray(self.world_size)
        target_high = np.asarray(self.world_size)

        proprio_low = self._kilobots[0].state_space.low
        proprio_high = self._kilobots[0].state_space.high

        obs_space_low = np.r_[self.kilobots_observation_space.low, self.object_observation_space.low,
                              target_low, proprio_low]
        obs_space_high = np.r_[self.kilobots_observation_space.high, self.object_observation_space.high,
                               target_high, proprio_high]

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
            self._add_kilobot(kb_class(self.world, position=position, orientation=orientation))
