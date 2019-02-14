import collections
import itertools
import random

import gym_kilobots
import numpy as np
from cycler import cycler
from gym import spaces
from gym_kilobots.envs import DirectControlKilobotsEnv, YamlKilobotsEnv
from gym_kilobots.kb_plotting import get_body_from_shape

import yaml
from copy import deepcopy


class TargetArea:
    def __init__(self, center, width, height):
        assert width > .0, 'width needs to be positive'
        assert height > .0, 'height needs to be positive'

        self.__center = np.asarray(center)
        self.__width = width
        self.__height = height

        self.__lower_bound = np.array((-width / 2, -height / 2))
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
        return np.all(np.abs(vec) <= (self.__width / 2, self.__height / 2))

    def __eq__(self, other):
        for k in self.__dict__:
            if k not in other.__dict__:
                return False
            if not self.__getattribute__(k) == other.__getattribute__(k):
                return False
        return True


class TargetPose(yaml.YAMLObject):
    yaml_tag = '!TargetPose'

    def __init__(self, pose, accuracy, periodic=True, frequency=1):
        self.pose = np.asarray(pose)
        self.accuracy = np.asarray(accuracy)
        self.periodic = periodic
        self.frequency = frequency

    @property
    def period(self):
        return 2 * np.pi / self.frequency

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
        error = np.abs(np.subtract(self.pose, other_pose))
        if self.periodic:
            error[2] %= self.period
            error[2] = min(error[2], self.period - error[2])
        return error

    def __contains__(self, other_pose):
        error = self.error(other_pose)
        return np.all(error < self.accuracy)

    def __eq__(self, other):
        for k in self.__dict__:
            if k not in other.__dict__:
                return False
            if not self.__getattribute__(k) == other.__getattribute__(k):
                return False
        return True


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
        if isinstance(n_obj, str):
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
        objects_obs_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1., 0])
        objects_obs_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1., 1])
        return spaces.Box(low=objects_obs_low, high=objects_obs_high, dtype=np.float64)

    @property
    def observation_space(self):
        kb_obs_space = self.kilobots_observation_space
        _observation_spaces_low = kb_obs_space.low
        _observation_spaces_high = kb_obs_space.high

        obj_obs_space = self.object_observation_space
        _observation_spaces_low = np.r_[_observation_spaces_low, np.tile(obj_obs_space.low, len(self._objects))]
        _observation_spaces_high = np.r_[_observation_spaces_high, np.tile(obj_obs_space.high, len(self._objects))]

        light_obs_space = self.light_observation_space
        _observation_spaces_low = np.concatenate((_observation_spaces_low, light_obs_space.low))
        _observation_spaces_high = np.concatenate((_observation_spaces_high, light_obs_space.high))

        return spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high, dtype=np.float32)

    def get_observation(self):
        observation = np.concatenate(tuple(k.get_position() for k in self._kilobots))

        for obj in self._objects:
            observation = np.r_[observation, obj.get_position(), np.sin(obj.get_orientation()),
                                np.cos(obj.get_orientation()), 1]
            # TODO include object shape here?

        observation = np.r_[observation, self._light.get_state()]

        return observation

    def _quadratic_cost(self, state, target, weights=1., normalize=True):
        diff = state - target

        if normalize:
            diff = diff / np.asarray(self.world_size)

        return -np.sum(diff ** 2 * weights, axis=1)

    def _init_objects(self):
        if self._num_objects == 'random':
            num_objs = random.randint(1, len(self.conf.objects))
        else:
            num_objs = self._num_objects
        for o in self.conf.objects[:num_objs]:
            self._init_object(object_shape=o.shape, object_width=o.width, object_height=o.height,
                              object_init=o.init, object_color=getattr(o, 'color', None))

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

        obj_states = next_state[-obj_dims:].reshape(-1, 3)
        obj_positions = obj_states[:, :2]
        finished_objs = [o in self._target_area for o in obj_positions]
        obj_cost = -1 * np.linalg.norm(obj_positions - self._target_area.center, axis=1)
        obj_cost[finished_objs] = 1.0

        kb_positions = next_state[:kb_dims].reshape(-1, 1, 2)
        kb_obj_dists = np.linalg.norm(kb_positions - obj_positions.reshape(1, -1, 2), axis=2)
        kb_obj_dists[:, finished_objs] = 10.

        kb_obj_cost = -1 * np.min(kb_obj_dists, axis=1)

        return np.sum(obj_cost)
        # return np.sum(obj_cost)

    def _get_random_object_init(self):
        object_init = super(MultiObjectTargetAreaEnv, self)._get_random_object_init()
        # reject initial object positions within the target area
        while object_init[:2] in self._target_area:
            object_init = super(MultiObjectTargetAreaEnv, self)._get_random_object_init()
        return object_init

    def _get_random_light_init(self, at_object=False):
        at_object = at_object and np.random.rand() < self.progress_factor
        if at_object:
            which_object = self._objects[np.random.choice(len(self._objects), 1)[0]]
            init_position = which_object.get_position()
            radius = 1.2 * max(which_object.width, which_object.height) / 2
            angle = np.random.rand() * 2 * np.pi - np.pi
            init_position += (np.cos(angle) * radius, np.sin(angle) * radius)
        else:
            init_position = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
        return init_position

    def has_finished(self, state, action):
        # if np.all([o.get_position() in self._target_area for o in self._objects]) \
        #         and self._sim_steps >= self._steps_per_action:
        #     return True

        return super(MultiObjectTargetAreaEnv, self).has_finished(state, action)

    def _draw_on_table(self, screen):
        w = self._target_area.width
        h = self._target_area.height
        vertices = np.array([[-w, -h], [-w, h], [w, h], [w, -h]]) / 2 + self._target_area.center
        screen.draw_polygon(vertices, color=(230, 230, 230), filled=True, width=.005)


class MultiObjectTargetPoseEnv(MultiObjectEnv):
    def __init__(self, configuration, **kwargs):
        self._target_poses = []
        self._body_ghosts = []

        super(MultiObjectTargetPoseEnv, self).__init__(configuration=configuration, **kwargs)

    def _configure_environment(self):
        super(MultiObjectTargetPoseEnv, self)._configure_environment()

        self._target_poses.clear()
        for obj_conf, obj in zip(self.conf.objects, self._objects):
            if hasattr(obj_conf, 'target'):
                tp = deepcopy(obj_conf.target)
                if tp.pose == 'random':
                    at_object = np.random.rand() < self.progress_factor
                    position = self._get_random_light_init(at_object)
                    if np.random.rand() < self.progress_factor:
                        orientation = init_orientation = np.random.rand() * 2 * np.pi - np.pi
                    else:
                        orientation = obj.get_orientation()
                    tp.pose = (*position, orientation)
                self._target_poses.append(tp)
            else:
                self._target_poses.append(TargetPose(pose=self._get_random_object_init(),
                                                     accuracy=(.02, .02, .05)))

        self._body_ghosts.clear()
        for c_obj, r_obj, tp in zip(self.conf.objects, self._objects, self._target_poses):
            body_ghost = get_body_from_shape(c_obj.shape, c_obj.width, c_obj.height, tp.pose)
            body_ghost.color = r_obj.color * 1.5
            body_ghost.highlight_color = r_obj.highlight_color * 1.5
            self._body_ghosts.append(body_ghost)

    @property
    def object_observation_space(self):
        # observe object position and sin, cos scaled by target point periodicity
        object_low = np.r_[self.world_x_range[0], self.world_y_range[0], -1., -1.]
        object_high = np.r_[self.world_x_range[1], self.world_y_range[1], 1., 1.]
        # observe position and sin, cos of target point
        target_low = np.r_[self.world_x_range[0], self.world_y_range[0], -1., -1.]
        target_high = np.r_[self.world_x_range[1], self.world_y_range[1], 1., 1.]

        objects_obs_low = np.r_[object_low, target_low, 0]
        objects_obs_high = np.r_[object_high, target_high, 1]
        return spaces.Box(low=objects_obs_low, high=objects_obs_high, dtype=np.float64)

    def get_observation(self):
        observation = np.concatenate(tuple(k.get_position() for k in self._kilobots))

        for obj, tp in zip(self._objects, self._target_poses):
            obj_obs = np.r_[obj.get_position(),
                            # divide orientation angle by period
                            np.sin(obj.get_orientation() / tp.period),
                            np.cos(obj.get_orientation() / tp.period)]
            tp_obs = np.r_[tp.position,
                           np.sin(tp.orientation / tp.period),
                           np.cos(tp.orientation / tp.period)]
            observation = np.r_[observation, obj_obs, tp_obs, 1]

            # TODO include object shape here?

        observation = np.r_[observation, self._light.get_state()]

        return observation

    def get_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        # kb_dims = self.kilobots_state_space.shape[0]

        obj_poses = state[-obj_dims:].reshape(-1, 3)
        next_obj_poses = next_state[-obj_dims:].reshape(-1, 3)

        obj_gain = np.zeros_like(obj_poses)
        next_obj_errors = np.zeros_like(obj_poses)
        for i, (pose, next_pose, tp) in enumerate(zip(obj_poses, next_obj_poses, self._target_poses)):
            next_obj_errors[i, :] = tp.error(next_pose)
            if next_pose in tp:
                obj_gain[i, :] = 1.
            else:
                obj_gain[i, :] = tp.error(pose) - next_obj_errors[i, :]

        # obj_errors = np.array([tp.error(o) for tp, o in zip(self._target_poses, obj_poses)])
        # next_obj_errors = np.array([tp.error(o) for tp, o in zip(self._target_poses, next_obj_poses)])
        # obj_gain = obj_errors - next_obj_errors
        #
        # for tp, ob, g in zip(self._target_poses, next_obj_poses, range(obj_gain.shape[0])):
        #     if ob in tp:
        #         obj_gain[g, :] = 1.

        # weight the translational gain higher and in a larger area around the target
        translation_weights = 20 * np.exp(-1. * np.sum(next_obj_errors[:, :2] ** 2, axis=1, keepdims=True) / 4)
        # weight the rotational gain lower and only in the area close to the target
        rotation_weights = 20 * np.exp(-1. * np.sum(next_obj_errors[:, :2] ** 2, axis=1, keepdims=True) / .05)

        reward = np.sum(translation_weights * obj_gain[:, :2])
        reward += np.sum(rotation_weights * obj_gain[:, 2:])

        return reward

    def _get_random_object_init(self):
        object_init = super(MultiObjectTargetPoseEnv, self)._get_random_object_init()
        # reject initial object positions within the target area
        while np.any([object_init in tp for tp in self._target_poses]):
            object_init = super(MultiObjectTargetPoseEnv, self)._get_random_object_init()
        return object_init

    def _get_random_light_init(self, at_object=False):
        at_object = at_object and np.random.rand() < self.progress_factor
        if at_object:
            which_object = self._objects[np.random.choice(len(self._objects), 1)[0]]
            init_position = which_object.get_position()
            radius = 1.2 * max(which_object.width, which_object.height) / 2
            angle = np.random.rand() * 2 * np.pi - np.pi
            init_position += (np.cos(angle) * radius, np.sin(angle) * radius)
        else:
            init_position = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
        return init_position

    def has_finished(self, state, action):
        if np.all([o.get_pose() in tp for o, tp in zip(self._objects, self._target_poses)]):
            return True
        return super(MultiObjectTargetPoseEnv, self).has_finished(state, action)

    def _draw_on_table(self, screen):
        for body_ghost in self._body_ghosts:
            body_ghost.draw(screen)


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
    def __init__(self, reward_function, agent_type='SimpleVelocityControlKilobot',
                 **kwargs):
        self._reward_function = reward_function
        self._object_colors = [(200, 150, 0),
                               (0, 150, 200),
                               (0, 200, 150),
                               (150, 0, 200)]
        self._num_cluster = 1

        self._reward_fuctions_with_obj_type = ['object_clustering', 'object_clustering_amp', 'fisher_clustering']
        self._observe_object_type = False
        if self._reward_function in self._reward_fuctions_with_obj_type:
            self._observe_object_type = True
            self._num_cluster = 2

        self._color_cycle = itertools.cycle(self._object_colors[:self._num_cluster])
        self._cluster_cycle = itertools.cycle(range(self._num_cluster))
        self._cluster_idx = []

        self._agent_score = None
        self._swarm_score = None

        self.agent_type = agent_type

        self._reward_ratio = 1.

        super(MultiObjectDirectControlEnv, self).__init__(**kwargs)

    def reset(self):
        self._cluster_idx = []
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
        # concat swarm observations
        A = np.concatenate((kb_rel_radius, np.sin(kb_rel_angle), np.cos(kb_rel_angle),
                            np.sin(kb_rel_orientations), np.cos(kb_rel_orientations)), axis=2)

        if kb_states.shape[-1] > 3:
            # absolute velocities
            kb_vel = np.tile(kb_states[..., 3:].reshape(1, -1, 2), (self.num_kilobots, 1, 1))
            # concat swarm observations
            A = np.concatenate((A, kb_vel), axis=2)

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

        # is this a valid object
        obj_valid = np.ones((self.num_kilobots, len(self._objects), 1))

        if self._observe_object_type:
            obj_type = np.zeros((self.num_kilobots, len(self._objects), self._num_cluster))
            obj_type[:, range(len(self._objects)), self._cluster_idx] = 1

            B = np.concatenate((obj_rel_radius, np.sin(obj_rel_angle), np.cos(obj_rel_angle),
                                np.sin(obj_rel_orientations), np.cos(obj_rel_orientations),
                                obj_type, obj_valid), axis=2)
        else:
            B = np.concatenate((obj_rel_radius, np.sin(obj_rel_angle), np.cos(obj_rel_angle),
                                np.sin(obj_rel_orientations), np.cos(obj_rel_orientations),
                                obj_valid), axis=2)

        object_dims = B.shape[2]
        B = B.reshape(self.num_kilobots, -1)

        # append zeros if env has less objects than defined in conf
        if len(self._objects) < len(self.conf.objects):
            num_missing_objects = len(self.conf.objects) - len(self._objects)
            B = np.concatenate((B, np.zeros((self.num_kilobots, object_dims * num_missing_objects))), axis=1)

        # proprioceptive observations
        kb_states = kb_states[:, 0, :]
        kb_proprio = np.concatenate((kb_states[:, :2], np.sin(kb_states[:, [2]]), np.cos(kb_states[:, [2]])), axis=1)
        if kb_states.shape[1] > 3:
            kb_proprio = np.concatenate((kb_proprio, kb_states[:, 3:]), axis=1)

        return np.concatenate((A, B, kb_proprio, np.full((self.num_kilobots, 1), len(self.objects))), axis=1)

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

        reward_function = getattr(self, '_' + self._reward_function + '_reward', None)
        if reward_function is None:
            raise NotImplementedError('reward function `{}` not implemented.'.format(self._reward_function))

        reward = reward_function(state, action, next_state)

        for kb, r in zip(self._kilobots, reward):
            if r > .1:
                kb.set_color((150, 150 + min([100, r * 50]), 150))
            elif r < -.1:
                kb.set_color((150 + min([100, -r * 50]), 150, 150))
            else:
                kb.set_color((150, 150, 150))

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

    def _moving_objects_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        obj_state = state[-obj_dims:].reshape(-1, 3)
        next_obj_state = next_state[-obj_dims:].reshape(-1, 3)

        reward = np.sum(np.linalg.norm(obj_state[:, :2] - next_obj_state[:, :2], axis=1))
        # reward -= .1 * np.sum(np.abs(obj_state[:, 2] - next_obj_state[:, 2]))

        return np.tile(reward, self.num_kilobots)

    def _move_objects_to_center_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        obj_state = state[-obj_dims:].reshape(-1, 3)
        next_obj_state = next_state[-obj_dims:].reshape(-1, 3)

        obj_dists = np.linalg.norm(obj_state[:, :2], axis=1)
        next_obj_dists = np.linalg.norm(next_obj_state[:, :2], axis=1)

        reward = np.sum(obj_dists - next_obj_dists)
        reward += .0005 * np.sum(np.cos(obj_state[:, 2] * 4))

        return np.tile(reward, self.num_kilobots)

    def _assembly_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        obj_state = state[-obj_dims:].reshape(-1, 3)
        next_obj_state = next_state[-obj_dims:].reshape(-1, 3)

        # compute reward based on relative distance between objects
        from scipy.spatial.distance import pdist
        obj_dists = pdist(obj_state[:, :2])
        next_obj_dists = pdist(next_obj_state[:, :2])

        dist_gain = np.sum(obj_dists - next_obj_dists)

        # symmetries = np.array([o_conf.symmetry for o_conf in self.conf.objects])

        # obj_angular_dist = pdist(np.c_[np.sin(obj_state[:, 2] * symmetries), np.cos(obj_state[:, 2] * symmetries)])
        # next_obj_angular_dist = pdist(np.c_[np.sin(next_obj_state[:, 2] * symmetries),
        #                                     np.cos(next_obj_state[:, 2] * symmetries)])
        #
        # angular_gain = np.sum(obj_angular_dist - next_obj_angular_dist)

        reward = dist_gain
        return np.tile(reward, self.num_kilobots)

    def _assembly_normalized_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        obj_state = state[-obj_dims:].reshape(-1, 3)
        next_obj_state = next_state[-obj_dims:].reshape(-1, 3)

        # compute reward based on relative distance between objects
        from scipy.spatial.distance import pdist
        obj_dists = pdist(obj_state[:, :2])
        next_obj_dists = pdist(next_obj_state[:, :2])

        dist_gain = np.mean(obj_dists - next_obj_dists)

        reward = dist_gain
        return np.tile(reward, self.num_kilobots)

    def _assembly_LL_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        obj_state = state[-obj_dims:].reshape(-1, 3)
        next_obj_state = next_state[-obj_dims:].reshape(-1, 3)

        # compute reward based on relative distance between objects
        from scipy.spatial.distance import pdist
        obj_dists = pdist(obj_state[:, :2])
        next_obj_dists = pdist(next_obj_state[:, :2])

        dist_gain = np.sum(obj_dists - next_obj_dists)

        # symmetries = np.array([o_conf.symmetry for o_conf in self.conf.objects])

        angular_distance = -np.cos(obj_state[0, 2] - obj_state[1, 2])
        next_angular_distance = -np.cos(next_obj_state[0, 2] - next_obj_state[1, 2])

        angular_distance_gain = angular_distance - next_angular_distance

        # obj_angular_dist = pdist(np.c_[np.sin(obj_state[:, 2] * symmetries), np.cos(obj_state[:, 2] * symmetries)])
        # next_obj_angular_dist = pdist(np.c_[np.sin(next_obj_state[:, 2] * symmetries),
        #                                     np.cos(next_obj_state[:, 2] * symmetries)])
        #
        # angular_gain = np.sum(obj_angular_dist - next_obj_angular_dist)
        actions = np.array([kb.get_action() for kb in self._kilobots])
        action_cost = - .01 * np.abs(actions[:, 0])

        reward = dist_gain + .5 * angular_distance_gain + action_cost
        return np.tile(reward, self.num_kilobots)

    def _object_clustering_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        obj_state = state[-obj_dims:].reshape(-1, 3)
        next_obj_state = next_state[-obj_dims:].reshape(-1, 3)

        # compute relative distance between objects
        from scipy.spatial.distance import pdist, squareform

        intra_cluster_gain = .0
        cluster_centers = []
        next_cluster_centers = []

        cluster_idx = np.asarray(self._cluster_idx)
        for c_idx in range(self._num_cluster):
            cluster_objs = cluster_idx == c_idx
            cluster_dists = pdist(obj_state[cluster_objs, :2])
            cluster_next_dists = pdist(next_obj_state[cluster_objs, :2])
            intra_cluster_gain += np.mean(cluster_dists - cluster_next_dists)

            cluster_centers.append(obj_state[cluster_objs, :2].mean(axis=0))
            next_cluster_centers.append(next_obj_state[cluster_objs, :2].mean(axis=0))

        extra_cluster_gain = np.mean(pdist(np.asarray(next_cluster_centers)) - pdist(np.asarray(cluster_centers)))

        reward = 1.5 * intra_cluster_gain / self._num_cluster + extra_cluster_gain

        return np.tile(reward, self.num_kilobots)

    def _object_clustering_amp_reward(self, state, action, next_state):
        reward = self._object_clustering_reward(state, action, next_state)
        return 20 * reward

    def _fisher_clustering_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        obj_state = state[-obj_dims:].reshape(-1, 3)
        next_obj_state = next_state[-obj_dims:].reshape(-1, 3)

        cluster_means = []
        cluster_vars = []
        next_cluster_means = []
        next_cluster_vars = []

        cluster_idx = np.asarray(self._cluster_idx)
        for c_idx in range(self._num_cluster):
            cluster_objs = cluster_idx == c_idx
            cluster_means.append(obj_state[cluster_objs, :2].mean(axis=0))
            cluster_vars.append(obj_state[cluster_objs, :2].var(axis=0))
            next_cluster_means.append(next_obj_state[cluster_objs, :2].mean(axis=0))
            next_cluster_vars.append(next_obj_state[cluster_objs, :2].var(axis=0))

        from scipy.spatial.distance import pdist
        fisher_crit = np.sum(pdist(np.array(cluster_means))**2) / np.sum(cluster_vars)
        next_fisher_crit = np.sum(pdist(np.array(next_cluster_means))**2) / np.sum(next_cluster_vars)

        return 20 * np.tile(next_fisher_crit - fisher_crit, self.num_kilobots)

    @staticmethod
    def _clustering(obj_positions: np.ndarray, num_clusters):
        mean_position = obj_positions.mean(axis=0)
        # normalize object_positions with mean
        obj_positions -= mean_position
        # check if an object is exactly on zero position
        obj_positions[np.all(obj_positions == (.0, .0), axis=1), :] = .01 * np.random.rand(2)
        # compute polar coordinates
        obj_angles = np.arctan2(obj_positions[:, 1], obj_positions[:, 0])

        clusters = np.split(np.argsort(obj_angles), num_clusters)

        return clusters

    @property
    def kilobots_observation_space(self):
        kb_low = self._kilobots[0].state_space.low
        kb_high = self._kilobots[0].state_space.high

        # radius, angle as sin+cos, orientation as sin+cos
        kb_low = np.r_[[.0, -1., -1., -1., -1.], kb_low[3:]]
        kb_high = np.r_[[np.sqrt(-self.world_width * -self.world_height), 1., 1., 1., 1.], kb_high[3:]]
        # kb_low = np.tile(kb_low, self.num_kilobots - 1)
        # kb_high = np.tile(kb_high, self.num_kilobots - 1)

        return spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)

    @property
    def object_observation_space(self):
        # radius, angle as sin+cos, orientation as sin+cos, type, object observation valid
        if self._reward_function in self._reward_fuctions_with_obj_type:
            cluster_one_hot_low = np.zeros(self._num_cluster)
            cluster_one_hot_high = cluster_one_hot_low + 1
            obj_low = np.r_[.0, -1., -1., -1., -1., cluster_one_hot_low, 0]
            obj_high = np.r_[np.sqrt(-self.world_width * -self.world_height), 1., 1., 1., 1., cluster_one_hot_high, 1]
        else:
            obj_low = np.r_[.0, -1., -1., -1., -1., 0]
            obj_high = np.r_[np.sqrt(-self.world_width * -self.world_height), 1., 1., 1., 1., 1]

        return spaces.Box(low=obj_low, high=obj_high, dtype=np.float64)

    @property
    def observation_space(self):
        agent_obs_space = self.kilobots_observation_space
        swarm_obs_space_low = np.tile(agent_obs_space.low, self.num_kilobots - 1)
        swarm_obs_space_high = np.tile(agent_obs_space.high, self.num_kilobots - 1)

        object_obs_space = self.object_observation_space
        objects_obs_space_low = np.tile(object_obs_space.low, len(self.conf.objects))
        objects_obs_space_high = np.tile(object_obs_space.high, len(self.conf.objects))

        if self.agent_type == 'SimpleAccelerationControlKilobot':
            # position, orientation as sin+cos, linear vel, angular vel, score
            proprio_low = np.r_[self.world_bounds[0], -1., -1., .0, -.5 * np.pi]
            proprio_high = np.r_[self.world_bounds[1], 1., 1., .01, .5 * np.pi]
        else:
            # position, orientation as sin+cos, score
            proprio_low = np.r_[self.world_bounds[0], -1., -1.]
            proprio_high = np.r_[self.world_bounds[1], 1., 1.]

        obs_space_low = np.r_[swarm_obs_space_low, objects_obs_space_low, proprio_low, 1]
        obs_space_high = np.r_[swarm_obs_space_high, objects_obs_space_high, proprio_high, len(self.conf.objects)]

        return spaces.Box(np.tile(obs_space_low, (self.num_kilobots, 1)),
                          np.tile(obs_space_high, (self.num_kilobots, 1)), dtype=np.float64)

    def _init_object(self, *, object_color, **kwargs):
        object_color = self._color_cycle.__next__()
        self._cluster_idx.append(self._cluster_cycle.__next__())
        super(MultiObjectDirectControlEnv, self)._init_object(object_color=object_color, **kwargs)

    def _init_kilobots(self, agent_type=None):
        num_kilobots = self.conf.kilobots.num

        # draw the kilobots positions uniformly from the world size
        kilobot_positions = np.random.rand(num_kilobots, 2) * np.asarray(self.world_size) + self.world_bounds[0]
        kilobot_orientations = np.random.rand(num_kilobots) * np.pi * 2 - np.pi

        # assert for each kilobot that it is within the world bounds and add kilobot to the world
        for position, orientation in zip(kilobot_positions, kilobot_orientations):
            position = np.maximum(position, self.world_bounds[0] + 0.02)
            position = np.minimum(position, self.world_bounds[1] - 0.02)
            kb_class = getattr(gym_kilobots.lib, self.agent_type)
            self._add_kilobot(kb_class(self.world, position=position, orientation=orientation,
                                       velocity=[.01, np.random.rand() * 0.1 * np.pi - 0.05 * np.pi]))

    @property
    def kilobots_state_space(self):
        kb_class = getattr(gym_kilobots.lib, self.agent_type)
        kb_state_space = kb_class.state_space
        kb_low = np.r_[self.world_bounds[0], kb_state_space.low[2:]]
        kb_high = np.r_[self.world_bounds[1], kb_state_space.high[2:]]
        return spaces.Box(low=np.tile(kb_low, self.num_kilobots), high=np.tile(kb_high, self.num_kilobots),
                          dtype=np.float64)

    def has_finished(self, state, action):
        return self._sim_steps >= self._done_after_steps * self._steps_per_action


class MultiObjectTargetAreaDirectControlEnv(MultiObjectDirectControlEnv):
    def __init__(self, **kwargs):
        self._target_area = TargetArea((.0, .0), .2, .2)

        super(MultiObjectTargetAreaDirectControlEnv, self).__init__(**kwargs)

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
        # concat swarm observations
        A = np.concatenate((kb_rel_radius, np.sin(kb_rel_angle), np.cos(kb_rel_angle),
                            np.sin(kb_rel_orientations), np.cos(kb_rel_orientations)), axis=2)

        if kb_states.shape[-1] > 3:
            # absolute velocities
            kb_vel = np.tile(kb_states[..., 3:].reshape(1, -1, 2), (self.num_kilobots, 1, 1))
            # concat swarm observations
            A = np.concatenate((A, kb_vel), axis=2)

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
        kb_proprio = np.concatenate((kb_states[:, :2], np.sin(kb_states[:, [2]]), np.cos(kb_states[:, [2]])), axis=1)
        if kb_states.shape[1] > 3:
            kb_proprio = np.concatenate((kb_proprio, kb_states[:, 3:]), axis=1)

        return np.concatenate((A, B, target_rel_radius, np.sin(target_rel_angle), np.cos(target_rel_angle),
                               kb_proprio, np.full((self.num_kilobots, 1), len(self.objects))), axis=1)

    @property
    def reward_ratio(self):
        return self._reward_ratio

    @reward_ratio.setter
    def reward_ratio(self, reward_ratio):
        assert .0 <= reward_ratio <= 1.
        self._reward_ratio = reward_ratio

    def _object_cleanup_sparse_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        kb_states = state[:kb_dims].reshape((self.num_kilobots, -1))
        kb_positions = kb_states[:, :2].reshape(-1, 1, 2)

        # get object positions in a (m x 2) matrix
        obj_state = state[-obj_dims:].reshape(-1, 3)
        obj_state = obj_state[:, :2]

        kb_obj_sqdist = np.sum(np.square(kb_positions - obj_state.reshape(1, -1, 2)), axis=2)

        kb_obj_touching = np.sqrt(kb_obj_sqdist) <= np.sqrt(2 * .036 ** 2)

        swarm_reward = .0
        for i, (o, o_pos) in enumerate(zip(self._objects, obj_state)):
            if o_pos in self._target_area:
                swarm_reward += 10
                kb_obj_touching[:, i] = False

        return np.asarray(np.any(kb_obj_touching, axis=1), dtype=np.float64) + swarm_reward

    def _object_cleanup_swarm_reward(self, state, action, next_state):
        obj_dims = self.object_state_space.shape[0]
        kb_dims = self.kilobots_state_space.shape[0]

        # get kilobot positions in a (n x 1 x 2) matrix
        kb_states = state[:kb_dims].reshape((self.num_kilobots, -1))
        kb_positions = kb_states[:, :2].reshape(-1, 1, 2)

        # get object positions in a (m x 2) matrix
        obj_state = state[-obj_dims:].reshape(-1, 3)
        obj_state = obj_state[:, :2]

        obj_target_cost = -np.linalg.norm(obj_state - self._target_area.center, axis=1)

        for i, (o, o_pos) in enumerate(zip(self._objects, obj_state)):
            if o_pos in self._target_area:
                obj_target_cost[i] = 0

        swarm_cost = .0
        for kb_pos in kb_positions:
            if kb_pos in self._target_area:
                swarm_cost -= 1

        return np.tile(np.sum(obj_target_cost), self.num_kilobots) + swarm_cost

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

    def _get_random_object_init(self):
        object_init = super(MultiObjectTargetAreaDirectControlEnv, self)._get_random_object_init()
        # reject initial object positions within the target area
        while object_init[:2] in self._target_area:
            object_init = super(MultiObjectTargetAreaDirectControlEnv, self)._get_random_object_init()
        return object_init

    @property
    def observation_space(self):
        agent_obs_space = self.kilobots_observation_space
        swarm_obs_space_low = np.tile(agent_obs_space.low, self.num_kilobots - 1)
        swarm_obs_space_high = np.tile(agent_obs_space.high, self.num_kilobots - 1)

        object_obs_space = self.object_observation_space
        objects_obs_space_low = np.tile(object_obs_space.low, len(self.conf.objects))
        objects_obs_space_high = np.tile(object_obs_space.high, len(self.conf.objects))

        # radius, angle as sin+cos
        target_low = np.array([.0, -1., -1.])
        target_high = np.array([np.sqrt(-self.world_width * -self.world_height), 1., 1.])

        if self.agent_type == 'SimpleAccelerationControlKilobot':
            # position, orientation as sin+cos, linear vel, angular vel, score
            proprio_low = np.r_[self.world_bounds[0], -1., -1., .0, -.5 * np.pi]
            proprio_high = np.r_[self.world_bounds[1], 1., 1., .01, .5 * np.pi]
        else:
            # position, orientation as sin+cos, score
            proprio_low = np.r_[self.world_bounds[0], -1., -1.]
            proprio_high = np.r_[self.world_bounds[1], 1., 1.]

        obs_space_low = np.r_[swarm_obs_space_low, objects_obs_space_low, target_low, proprio_low, 1]
        obs_space_high = np.r_[swarm_obs_space_high, objects_obs_space_high, target_high, proprio_high,
                               len(self.conf.objects)]

        return spaces.Box(np.tile(obs_space_low, (self.num_kilobots, 1)),
                          np.tile(obs_space_high, (self.num_kilobots, 1)), dtype=np.float64)

    def has_finished(self, state, action):
        if np.all([o.get_position() in self._target_area for o in self._objects]):
            return True
        return self._sim_steps >= self._done_after_steps * self._steps_per_action

    def _draw_on_table(self, screen):
        w = self._target_area.width
        h = self._target_area.height
        vertices = np.array([[-w, -h], [-w, h], [w, h], [w, -h]]) / 2 + self._target_area.center
        screen.draw_polygon(vertices, color=(0.5, 0.5, 0.5), filled=False, width=.005)
