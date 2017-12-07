import numpy as np

from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import PhototaxisKilobot, CornerQuad, CircularGradientLight

import abc


def QuadPushingEnvWith(weight, num_kilobots):
    class _QuadPushingEnv(QuadPushingEnv):
        @property
        def _weight(self):
            return weight

        @property
        def _num_kilobots(self):
            return num_kilobots

    return _QuadPushingEnv


class QuadPushingEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., .5

    _cost_vector = np.array([1., 1., 1.])

    _reward_scale_dx = 1.
    _reward_scale_da = 1.
    _reward_c1 = 100
    _reward_c2 = -30

    _object_init = np.array([.0, .0, .0])
    _object_size = .2

    _light_radius = .2

    _spawn_type_ratio = -.95
    # _sampling_max_radius = .5
    _spawn_radius_variance = .4 * _light_radius
    _light_max_dist = .5 * _object_size
    _spawn_angle_mean = 0
    _spawn_angle_variance = np.pi

    @property
    @abc.abstractmethod
    def _weight(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _num_kilobots(self):
        raise NotImplementedError

    def __init__(self):
        super().__init__()

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots) + (self._light.get_state(),))

    def _reward(self, state, _):
        object_state = self._objects[0].get_state()
        object_diff = self._object_init - object_state

        cost = object_diff * self._cost_vector
        w = self._weight

        r_trans = object_diff[0] - cost[2]
        r_rot = object_diff[2] - cost[0]
        return w * r_rot + (1 - w) * r_trans - cost[1]

    def _configure_environment(self):
        # initialize object always at (0, 0) with 0 orientation (we do not need to vary the object position and
        # orientation since we will later adapt the state based on the object pose.)
        self._objects.append(CornerQuad(width=self._object_size, height=self._object_size,
                                        position=self._object_init[:2], orientation=self._object_init[2],
                                        world=self.world))

        # determine the sampling mode for this episode
        _spawn_randomly = np.random.rand() < np.abs(self._spawn_type_ratio)

        if _spawn_randomly:
            # if we spawn the light randomly, we select random polar coordinates for the light
            # the radius is half the object width plus a uniform sample from the range [0, light_max_dist)
            _light_radius = (0.5 * self._object_size + self._light_max_dist * np.random.rand())
            # the angle is sampled normally with spawn_angle_mean and spawn_angle_variance
            angle = np.random.normal(self._spawn_angle_mean, self._spawn_angle_variance)

            _light_init = self._object_init[:2] + _light_radius * np.array([np.cos(angle), np.sin(angle)])
            _light_init = np.maximum(_light_init, self.world_bounds[0] + self._light_radius)
            _light_init = np.minimum(_light_init, self.world_bounds[1] - self._light_radius)
        else:
            # otherwise, the light starts above the object
            _light_init = self._object_init[:2]

        # initialize light position
        self._light = CircularGradientLight(position=_light_init, radius=self._light_radius, bounds=self.world_bounds)

        # kilobots start at the light position in a slightly random formation
        _kb_radius = PhototaxisKilobot.get_radius()
        _kilobot_offsets = np.array([[-_kb_radius, 0], [_kb_radius, 0], [0, _kb_radius], [0, -_kb_radius]])
        _start_angles = 2 * np.pi * np.random.randn(self._num_kilobots)

        for i in range(self._num_kilobots):
            if _spawn_randomly:
                _position = _light_init + (1 + i / 4) * _kilobot_offsets[i % 4, :] + .25 * (np.random.rand(2) - .5)
                # x = (start[0] + (10 + i / 4) * kilobot_offsets[i % 4, 0] + 0.25 * (np.random.rand() - 0.5))
                # y = (start[1] + (10 + i / 4) * kilobot_offsets[i % 4, 1] + 0.25 * (np.random.rand() - 0.5))
            else:
                _sampled_radius = np.maximum(np.random.normal(0, self._spawn_radius_variance), 1.1 * _kb_radius)
                _sampled_radius = np.minimum(_sampled_radius, self._light_radius)
                # x = float(start[0] + 1.1 * sampled_radius * np.cos(start_angles[i]))
                # y = float(start[1] + 1.1 * sampled_radius * np.sin(start_angles[i]))
                _position = _light_init + _sampled_radius * np.r_[np.cos(_start_angles[i]), np.sin(_start_angles[i])]

            # the orientation of the kilobots is chosen randomly
            _orientation = 2 * np.pi * np.random.rand()

            self._add_kilobot(PhototaxisKilobot(self.world, position=_position, orientation=_orientation,
                                                light=self._light))

        # step world once to resolve collisions
        self._step_world()
