import numpy as np

from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import PhototaxisKilobot, SimplePhototaxisKilobot, Body, CircularGradientLight
from gym.spaces import Box

import abc


class ObjectEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., 1.
    screen_size = screen_width, screen_height = 1000, 500

    _cost_vector = np.array([0.01, 0.1, 0.01])
    _scale_vector = np.array([5., 5., .5])

    _object_init = np.array([.0, .0, .0])
    _object_width, _object_height = .2, .2

    _light_radius = .2

    _spawn_type_ratio = -.95
    # _sampling_max_radius = .5
    _spawn_radius_variance = .2 * _light_radius
    _light_max_dist = .4
    _spawn_angle_mean = np.pi
    _spawn_angle_variance = .5 * np.pi

    action_space = Box(np.array([-.02, -.02]), np.array([.02, .02]), dtype=np.float64)

    def __init__(self):
        self._weight = None
        self._num_kilobots = None

        super().__init__()

    def get_state(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
                              + (self._transform_position(self._light.get_state()),)
                              + tuple(o.get_pose() for o in self._objects))

    def _transform_position(self, position):
        return self._objects[0].get_local_point(position)

    def get_observation(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),))

    def get_reward(self, state, _, new_state):
        obj_pose = state[-3:]
        obj_pose_new = new_state[-3:]

        # compute diff between last and current pose
        obj_pose_diff = obj_pose_new - obj_pose

        # scale diff
        obj_pose_diff_scaled = obj_pose_diff * self._scale_vector

        cost = np.abs(obj_pose_diff_scaled) * self._cost_vector

        w = self._weight

        r_trans = obj_pose_diff_scaled[0] - cost[2]
        r_rot = obj_pose_diff_scaled[2] - cost[0]
        return w * r_rot + (1 - w) * r_trans - cost[1]

    def _configure_environment(self):
        np.random.seed(self.seed()[0])
        # initialize object always at (0, 0) with 0 orientation (we do not need to vary the object position and
        # orientation since we will later adapt the state based on the object pose.)
        self._objects.append(self._create_object())

        # determine the sampling mode for this episode
        self.__spawn_randomly = np.random.rand() < np.abs(self._spawn_type_ratio)

        _light_init = self._sample_light_init()
        _light_init = np.maximum(_light_init, self.world_bounds[0])
        _light_init = np.minimum(_light_init, self.world_bounds[1])

        # initialize light position
        self._light = CircularGradientLight(position=_light_init, radius=self._light_radius,
                                            bounds=self.world_bounds)

        _kilobot_poses = self._sample_kilobot_init_poses()

        for _position, _orientation in _kilobot_poses:
            _position = np.maximum(_position, self.world_bounds[0] + 0.02)
            _position = np.minimum(_position, self.world_bounds[1] - 0.02)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=_position + _light_init,
                                                      orientation=_orientation, light=self._light))

        # step world once to resolve collisions
        self._step_world()

    @abc.abstractmethod
    def _create_object(self) -> Body:
        pass

    def _sample_light_init(self):
        if self.__spawn_randomly:
            # if we spawn the light randomly, we select random polar coordinates for the light
            # the radius is half the object width plus a uniform sample from the range [0, light_max_dist)
            _light_radius = (0.5 * self._object_width + self._light_max_dist * np.random.rand())
            # the angle is sampled normally with spawn_angle_mean and spawn_angle_variance
            angle = np.random.normal(self._spawn_angle_mean, self._spawn_angle_variance)

            _light_init = self._object_init[:2] + _light_radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            # otherwise, the light starts above the object
            _light_init = self._object_init[:2]

        # _light_init = np.random.normal(scale=self._light_max_dist, size=2)

        return _light_init

    def _sample_kilobot_init_poses(self):
        # kilobots start at the light position in a slightly random formation
        _kb_radius = PhototaxisKilobot.get_radius()
        _kilobot_offsets = np.array([[-_kb_radius, 0], [_kb_radius, 0], [0, _kb_radius], [0, -_kb_radius]])
        _start_angles = 2 * np.pi * np.random.randn(self._num_kilobots)

        kilobot_poses = []

        for i in range(self._num_kilobots):
            if self.__spawn_randomly:
                _position = (1 + i / 4) * _kilobot_offsets[i % 4, :] + .25 * (np.random.rand(2) - .5)
                # x = (start[0] + (10 + i / 4) * kilobot_offsets[i % 4, 0] + 0.25 * (np.random.rand() - 0.5))
                # y = (start[1] + (10 + i / 4) * kilobot_offsets[i % 4, 1] + 0.25 * (np.random.rand() - 0.5))
            else:
                _sampled_radius = np.maximum(np.random.normal(0, self._spawn_radius_variance), 1.1 * _kb_radius)
                _sampled_radius = np.minimum(_sampled_radius, self._light_radius)
                # x = float(start[0] + 1.1 * sampled_radius * np.cos(start_angles[i]))
                # y = float(start[1] + 1.1 * sampled_radius * np.sin(start_angles[i]))
                _position = _sampled_radius * np.r_[np.cos(_start_angles[i]), np.sin(_start_angles[i])]

            # the orientation of the kilobots is chosen randomly
            _orientation = 2 * np.pi * np.random.rand()

            kilobot_poses.append((_position, _orientation))

        return kilobot_poses


