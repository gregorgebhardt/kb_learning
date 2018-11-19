import random

import gym_kilobots
import numpy as np
from gym import spaces
from gym_kilobots.envs import YamlKilobotsEnv, DirectControlKilobotsEnv
from gym_kilobots.lib import SimpleDirectControlKilobot


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


class TargetPose:
    def __init__(self, pose, accuracy, periodic=True, period=2*np.pi):
        self.pose = np.asarray(pose)
        self.accuracy = np.asarray(accuracy)
        self.periodic = periodic
        self.period = period

    @property
    def position(self):
        return self.pose[:2]

    @property
    def orientation(self):
        return self.pose[2]

    @property
    def x_accuracy(self):
        return self.accuracy[0]

    @property
    def y_accuracy(self):
        return self.accuracy[1]

    @property
    def th_accuracy(self):
        return self.accuracy[2]

    def error(self, other_pose):
        error = np.abs(self.pose - other_pose)
        if self.periodic:
            error[2] %= self.period
            error[2] = min(error[2], self.period - error[2])
        return error

    def __contains__(self, other_pose):
        error = self.error(other_pose)
        return np.all(error < self.accuracy)


class MultiObjectEnv(YamlKilobotsEnv):
    _observe_object = True
    __steps_per_action = 20

    def __init__(self, *, configuration, done_after_steps=350, **kwargs):
        self._done_after_steps = done_after_steps
        self._num_objects = None

        super(MultiObjectEnv, self).__init__(configuration=configuration, **kwargs)

        # self._real_time = True

    @property
    def num_objects(self):
        return len(self.objects)

    @num_objects.setter
    def num_objects(self, n_obj):
        if isinstance(n_obj, 'str'):
            self._num_objects = 'random'
        self._num_objects = max(min(n_obj, len(self.conf.objects)), 1)

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {}

    @property
    def object_observation_space(self):
        objects_obs_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1.] * (len(self._objects)))
        objects_obs_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1.] * (len(self._objects)))
        return spaces.Box(low=objects_obs_low, high=objects_obs_high, dtype=np.float64)

    @property
    def observation_space(self):
        kb_obs_space = self.kilobots_observation_space
        _observation_spaces_low = kb_obs_space.low
        _observation_spaces_high = kb_obs_space.high

        obj_obs_space = self.object_observation_space
        _observation_spaces_low = np.concatenate((_observation_spaces_low, obj_obs_space.low))
        _observation_spaces_high = np.concatenate((_observation_spaces_high, obj_obs_space.high))

        light_obs_space = self.light_observation_space
        _observation_spaces_low = np.concatenate((_observation_spaces_low, light_obs_space.low))
        _observation_spaces_high = np.concatenate((_observation_spaces_high, light_obs_space.high))

        return spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high, dtype=np.float32)

    def get_observation(self):
        observation = np.concatenate(tuple(k.get_position() for k in self._kilobots))

        for obj in self._objects:
            observation = np.r_[observation, obj.get_position(), np.sin(obj.get_orientation()),
                                np.cos(obj.get_orientation())]
            # TODO include object shape here?

        observation = np.r_[observation, self._light.get_state()]

        return observation

    def _quadratic_cost(self, state, target, weights=1., normalize=True):
        diff = state - target

        if normalize:
            diff = diff / np.asarray(self.world_size)

        return -np.sum(diff**2 * weights, axis=1)

    def _init_objects(self):
        if self._num_objects == 'random':
            num_objs = random.randint(1, len(self.conf.objects))
            for o in self.conf.objects[:num_objs]:
                self._init_object(o.shape, o.width, o.height, o.init, getattr(o, 'color', None))
        else:
            for o in self.conf.objects[:self._num_objects]:
                self._init_object(o.shape, o.width, o.height, o.init, getattr(o, 'color', None))

    def has_finished(self, state, action):
        return self._sim_steps >= self._done_after_steps * self._steps_per_action

    def _draw_on_table(self, screen):
        # from gym_kilobots.kb_plotting import get_body_from_shape
        # focus_object_conf = self.conf.objects[self._focus_object_idx]
        # ghost_body = get_body_from_shape(focus_object_conf.shape, focus_object_conf.width, focus_object_conf.height,
        #                                  self._target_pose.pose)
        #
        # ghost_body.set_color((120, 120, 120, 80))
        # ghost_body.set_highlight_color((160, 160, 160, 80))
        # ghost_body.draw(viewer=screen)
        pass


class MultiObjectTargetAreaEnv(MultiObjectEnv):
    def __init__(self, **kwargs):
        self._target_area = TargetArea((.0, .0), .2, .2)

        super(MultiObjectTargetAreaEnv, self).__init__(**kwargs)

    def get_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # TODO check if we can use absolute cost instead of difference of quadratic costs here
        obj_state = state[-obj_dims:]
        obj_state = np.c_[obj_state[0::3], obj_state[1::3]]
        finished_objs = [o in self._target_area for o in obj_state]
        obj_rew = self._quadratic_cost(obj_state, self._target_area.center, weights=20)
        obj_rew[finished_objs] = 1.

        next_obj_state = next_state[-obj_dims:]
        next_obj_state = np.c_[next_obj_state[0::3], next_obj_state[1::3]]
        next_finished_objs = [o in self._target_area for o in next_obj_state]
        next_obj_rew = self._quadratic_cost(next_obj_state, self._target_area.center, weights=20)
        next_obj_rew[next_finished_objs] = 1.

        obj_rew_diff = np.sum(next_obj_rew - obj_rew) * 100

        kb_rew_diff = .0

        return obj_rew_diff + kb_rew_diff

    def has_finished(self, state, action):
        if np.all([o.get_position() in self._target_area for o in self._objects]) \
                and self._sim_steps >= self._steps_per_action:
            return True

        return super(MultiObjectTargetAreaEnv, self).has_finished(state, action)


class MultiObjectAssemblyEnv(MultiObjectEnv):
    @property
    def observation_space(self):
        kb_obs_space = self.kilobots_observation_space
        _observation_spaces_low = kb_obs_space.low
        _observation_spaces_high = kb_obs_space.high

        obj_obs_space = self.object_observation_space
        _observation_spaces_low = np.concatenate((_observation_spaces_low, obj_obs_space.low))
        _observation_spaces_high = np.concatenate((_observation_spaces_high, obj_obs_space.high))

        light_obs_space = self.light_observation_space
        _observation_spaces_low = np.concatenate((_observation_spaces_low, light_obs_space.low))
        _observation_spaces_high = np.concatenate((_observation_spaces_high, light_obs_space.high))

        return spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high,
                          dtype=np.float32)

    def get_observation(self):
        observation = np.concatenate(tuple(k.get_position() for k in self._kilobots))

        for obj in self._objects:
            observation = np.r_[observation, obj.get_position(), np.sin(obj.get_orientation()),
                                np.cos(obj.get_orientation())]

        observation = np.r_[observation, self._light.get_state()]

        return observation

    def get_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        # kb_dims = self.kilobots_state_space.shape[0]

        obj_poses = state[-obj_dims:].reshape(-1, 3)
        obj_positions = obj_poses[:, :2].reshape(-1, 1, 2)

        # compute point-wise distance between objects
        obj_dists = np.linalg.norm(obj_positions - np.transpose(obj_positions, (1, 0, 2)), axis=2)

        return -np.sum(obj_dists)

    def has_finished(self, state, action):
        return self._sim_steps >= self._done_after_steps * self._steps_per_action


class MultiObjectDirectControlEnv(DirectControlKilobotsEnv, MultiObjectEnv):
    def __init__(self, reward_function, agent_reward, swarm_reward, **kwargs):
        self._target_area = TargetArea((.0, .0), .2, .2)
        self._reward_function = reward_function
        self._agent_reward = agent_reward
        self._swarm_reward = swarm_reward

        self._agent_score = None
        self._swarm_score = None

        self._reward_ratio = 1.

        super(MultiObjectDirectControlEnv, self).__init__(**kwargs)

    def reset(self):
        return super(MultiObjectEnv, self).reset()

    def _configure_environment(self):
        super(MultiObjectDirectControlEnv, self)._configure_environment()
        self._agent_score = np.zeros(self.num_kilobots)
        self._swarm_score = .0

    def get_state(self):
        return np.concatenate(tuple(k.get_state() for k in self._kilobots)
                              + tuple(o.get_state() for o in self._objects))

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
        target_rel_position = (-kb_states[..., :2] + self._target_area.center)[:, 0, :]
        # relative polar coordinates
        target_rel_radius = np.linalg.norm(target_rel_position, axis=1, keepdims=True)
        target_rel_angle = np.arctan2(target_rel_position[:, 1], target_rel_position[:, 0]).reshape(-1, 1)
        target_rel_angle -= kb_states[..., 2]

        # proprioceptive observations
        kb_states = kb_states[:, 0, :]
        kb_proprioception = np.concatenate((kb_states[:, :2], np.sin(kb_states[:, [2]]), np.cos(kb_states[:, [2]]),
                                            kb_states[:, 3:], self._agent_score.reshape(-1, 1)), axis=1)

        return np.concatenate((A, B, target_rel_radius, np.sin(target_rel_angle), np.cos(target_rel_angle),
                               kb_proprioception, np.full((self.num_kilobots, 1), len(self.objects))), axis=1)

    @property
    def reward_ratio(self):
        return self._reward_ratio

    @reward_ratio.setter
    def reward_ratio(self, reward_ratio):
        assert .0 <= reward_ratio <= 1.
        self._reward_ratio = reward_ratio

    def get_reward(self, state, action, next_state):
        # obj_dims = self.object_state_space.shape[0]
        # kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        # kb_states = state[:kb_dims].reshape((-1, 5))
        # kb_positions = kb_states[:, :2].reshape(-1, 1, 2)
        # kb_velocities = kb_states[:, 3:]

        # get object positions in a (m x 2) matrix
        # obj_state = state[-obj_dims:].reshape(-1, 3)
        # obj_state = obj_state[:, :2]
        # index which objects are in target area
        # finished_objs = [o_pos in self._target_area for o_pos in obj_state]

        # obj_next_state = next_state[-obj_dims:].reshape(-1, 3)
        # obj_next_state = obj_next_state[:, :2]

        # cost for objects
        # obj_cost = self._quadratic_cost(obj_state, self._target_area.center, weights=20)
        # negative norm of distance to target
        # obj_cost = -np.linalg.norm(obj_state - self._target_area.center, axis=1)
        # obj_cost_next = -np.linalg.norm(obj_next_state - self._target_area.center, axis=1)

        # obj_gain = obj_cost_next - obj_cost
        # obj_gain[finished_objs] = .0

        # set cost to .0 for objects in target area
        # obj_cost[finished_objs] = 1.
        # sum up costs
        # obj_cost = np.sum(obj_cost)
        # obj_cost = np.asarray([obj_cost] * self.num_kilobots)

        # cost for kilobots
        # norm of distance between all kilobots and all objects
        # kb_obj_sqdist = np.sum(np.square(kb_positions - obj_state.reshape(1, -1, 2)), axis=2)
        # sum over negative min distance for each agent
        # kb_obj_cost = -1 * np.min(kb_obj_dist, axis=1)

        # # local positive gain in small area
        # pos_obj_gain = np.zeros_like(obj_gain)
        # pos_obj_gain[obj_gain > .0] = obj_gain[obj_gain > .0]
        # obj_gain_per_kb = 1500 * np.exp(-kb_obj_sqdist * .5 / .04**2).dot(pos_obj_gain)
        # # global cost for negative gains
        # neg_obj_gain = np.zeros_like(obj_gain)
        # neg_obj_gain[obj_gain < .0] = obj_gain[obj_gain < .0]
        # obj_gain_per_kb += 1000 * np.exp(-kb_obj_sqdist * .5 / .06 ** 2).dot(neg_obj_gain)

        # kb_obj_cost = np.sum(-1. + np.exp(-kb_obj_dist * .2))

        # velocity_cost = -1. * np.sum(np.abs(kb_velocities), axis=1)
        # vertigo_cost = -.01 * np.abs(kb_velocities[:, 1])
        # action_cost = -5. * np.sum(np.abs([kb.get_action() for kb in self._kilobots]), axis=1)

        # kb_in_target_cost = np.zeros(self.num_kilobots)
        # kb_in_target_cost[[kb in self._target_area for kb in kb_positions]] = -.1

        # reward = obj_gain_per_kb + vertigo_cost

        if self._reward_function == 'object_touching':
            reward = self._object_touching_reward(state, action, next_state)
        elif self._reward_function == 'object_collecting':
            reward = self._object_collecting_reward(state, action, next_state)
        elif self._reward_function == 'object_cleanup':
            reward = self._object_cleanup_reward(state, action, next_state)
        elif self._reward_function == 'object_cleanup_sparse':
            reward = self._object_cleanup_sparse_reward(state, action, next_state)
        else:
            raise Exception('reward function `{}` not implemented.'.format(self._reward_function))

        # TODO Sisyphus-Cleanup-Task
        #   reward and re-spawning for objects in the target area

        for kb, r in zip(self._kilobots, reward):
            if r > .1:
                kb.set_color((150, 150 + min([100, r * 50]), 150))
            elif r < -.1:
                kb.set_color((150 + min([100, -r * 50]), 150, 150))
            else:
                kb.set_color((150, 150, 150))

        return reward

    def _object_cleanup_sparse_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        kb_states = state[:kb_dims].reshape((-1, 5))
        kb_positions = kb_states[:, :2].reshape(-1, 1, 2)

        # get object positions in a (m x 2) matrix
        obj_state = state[-obj_dims:].reshape(-1, 3)
        obj_state = obj_state[:, :2]

        kb_obj_sqdist = np.sum(np.square(kb_positions - obj_state.reshape(1, -1, 2)), axis=2)

        kb_obj_touching = np.sqrt(kb_obj_sqdist) <= np.sqrt(2 * .036 ** 2)

        swarm_reward = .0
        for o, o_pos in zip(self._objects, obj_state):
            if o_pos in self._target_area:
                swarm_reward += 1

        return np.asarray(np.any(kb_obj_touching, axis=1), dtype=np.float64) + swarm_reward

    def _object_cleanup_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        kb_states = state[:kb_dims].reshape((-1, 5))
        kb_positions = kb_states[:, :2].reshape(-1, 1, 2)

        # get object positions in a (m x 2) matrix
        obj_state = state[-obj_dims:].reshape(-1, 3)
        obj_state = obj_state[:, :2]

        obj_next_state = next_state[-obj_dims:].reshape(-1, 3)
        obj_next_state = obj_next_state[:, :2]
        obj_push_vector = obj_state - obj_next_state
        obj_push_vector /= np.linalg.norm(obj_state - obj_next_state, axis=1, keepdims=True)
        obj_push_vector *= np.array([[o.width / 2] for o in self._objects])
        obj_push_loc = obj_state + obj_push_vector

        kb_obj_sqdist = np.sum(np.square(kb_positions - obj_state.reshape(1, -1, 2)), axis=2)
        kb_push_sqdist = np.sum(np.square(kb_positions - obj_push_loc.reshape(1, -1, 2)), axis=2)

        kb_obj_touching = np.sqrt(kb_obj_sqdist) <= np.sqrt(2 * .036 ** 2)
        kb_obj_pushing = np.sqrt(kb_push_sqdist) <= np.sqrt(2 * .036 ** 2)

        agent_reward = np.ones(self.num_kilobots) * self._agent_score * 10
        swarm_reward = self._swarm_score * 1
        for o, o_pos, t in zip(self._objects, obj_state, kb_obj_pushing.T):
            if o_pos in self._target_area:
                agent_reward[t] += 100
                self._agent_score[t] += 1
                swarm_reward += 10
                self._swarm_score += 1
                o.set_pose(self._get_random_object_init())

        reward = np.zeros(self.num_kilobots)
        if self._swarm_reward:
            reward += swarm_reward
        if self._agent_reward:
            reward += agent_reward
        if self._agent_reward and self._swarm_reward:
            reward /= 2
        reward += np.any(kb_obj_pushing, axis=1)

        return reward

    def _object_collecting_reward(self, state, *args, **kwargs):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        kb_states = state[:kb_dims].reshape((-1, 5))
        kb_positions = kb_states[:, :2].reshape(-1, 1, 2)

        # get object positions in a (m x 2) matrix
        obj_state = state[-obj_dims:].reshape(-1, 3)
        obj_state = obj_state[:, :2]

        kb_obj_sqdist = np.sum(np.square(kb_positions - obj_state.reshape(1, -1, 2)), axis=2)

        kb_obj_touching = np.sqrt(kb_obj_sqdist) <= np.sqrt(2 * .036 ** 2)

        for o, t in zip(self._objects, np.any(kb_obj_touching, axis=0)):
            if t:
                o.set_pose(self._get_random_object_init())

        reward = np.zeros(self.num_kilobots)
        if self._agent_reward:
            reward += np.sum(kb_obj_touching, axis=1)
        if self._swarm_reward:
            reward += np.sum(kb_obj_touching)
        if self._agent_reward and self._swarm_reward:
            reward /= 2

        return reward

    def _object_touching_reward(self, state, *args, **kwargs):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        kb_states = state[:kb_dims].reshape((-1, 5))
        kb_positions = kb_states[:, :2].reshape(-1, 1, 2)

        # get object positions in a (m x 2) matrix
        obj_state = state[-obj_dims:].reshape(-1, 3)
        obj_state = obj_state[:, :2]

        kb_obj_sqdist = np.sum(np.square(kb_positions - obj_state.reshape(1, -1, 2)), axis=2)

        kb_obj_touching = np.sqrt(kb_obj_sqdist) <= np.sqrt(2 * .036 ** 2)

        reward = np.zeros(self.num_kilobots)
        if self._swarm_reward:
            reward += np.sum(kb_obj_touching)
        if self._agent_reward:
            reward += np.sum(kb_obj_touching, axis=1)
        if self._agent_reward and self._swarm_reward:
            reward /= 2

        return reward

    def _get_random_object_init(self):
        object_init = super(MultiObjectDirectControlEnv, self)._get_random_object_init()
        # reject initial object positions within the target area
        while object_init[:2] in self._target_area:
            object_init = super(MultiObjectDirectControlEnv, self)._get_random_object_init()
        return object_init

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
        # radius, angle as sin+cos, orientation as sin+cos, object observation valid
        obj_low = np.array([.0, -1., -1., -1., -1., 0] * len(self.conf.objects))
        obj_high = np.array([np.sqrt(-self.world_width * -self.world_height), 1., 1., 1., 1., 1
                             ] * len(self.conf.objects))

        return spaces.Box(low=obj_low, high=obj_high, dtype=np.float64)

    @property
    def observation_space(self):
        # radius, angle as sin+cos
        target_low = np.array([.0, -1., -1.])
        target_high = np.array([np.sqrt(-self.world_width * -self.world_height), 1., 1.])

        # proprio_low = self._kilobots[0].state_space.low
        # position, orientation as sin+cos, linear vel, angular vel, score
        proprio_low = np.r_[self.world_bounds[0], -1., -1., .0, -.5 * np.pi, .0]
        # proprio_high = self._kilobots[0].state_space.high
        proprio_high = np.r_[self.world_bounds[1], 1., 1., .01, .5 * np.pi, np.inf]

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
                                       velocity=[.01, np.random.rand() * 0.1 * np.pi - 0.05 * np.pi]))

    @property
    def kilobots_state_space(self):
        kb_state_space = SimpleDirectControlKilobot.state_space
        kb_low = np.r_[self.world_bounds[0], kb_state_space.low[2:]]
        kb_high = np.r_[self.world_bounds[1], kb_state_space.high[2:]]
        return spaces.Box(low=np.tile(kb_low, self.num_kilobots), high=np.tile(kb_high, self.num_kilobots),
                          dtype=np.float64)

    def has_finished(self, state, action):
        if np.all([o.get_position() in self._target_area for o in self._objects]):
            return True
        return self._sim_steps >= self._done_after_steps * self._steps_per_action

    def _draw_on_table(self, screen):
        w = self._target_area.width
        h = self._target_area.height
        vertices = np.array([[-w, -h], [-w, h], [w, h], [w, -h]]) / 2 + self._target_area.center
        screen.draw_polygon(vertices, color=(0.5, 0.5, 0.5), filled=False, width=.005)
