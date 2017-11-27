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

    def _reward(self, state, _):
        object_state = state['objects'][0, :]
        object_diff = self._object_init - object_state

        cost = object_diff * self._cost_vector
        w = self._weight

        r_trans = object_diff[0] - cost[2]
        r_rot = object_diff[2] - cost[0]
        return w * r_rot + (1 - w) * r_trans - cost[1]

    def _configure_environment(self):
        self._object_init = np.array([.0, .0])
        self._objects.append(CornerQuad(width=.2, height=.2, position=self._object_init, world=self.world))

        max_dist = 1.2

        # determine the sampling mode for this episode
        spawnConcentrated = np.random.rand() < np.abs(self.samplingTypeRatio)
        useGridSampling = (self.samplingTypeRatio < 0)  # & (ep < 121)

        if useGridSampling:
            # px = objStart[0] + -0.4 + ((ep % 11)*0.08)
            # py = objStart[1] + -0.4 + (int(ep / 11)*0.08)
            # start = [px, py]
            pass
        elif spawnConcentrated:
            # light starts in circle around the object
            # start = startPositions[ep, :]
            radius = (0.1 + self.lightMaxDist * np.random.rand())
            angle = np.random.normal(self.spawnAngleMean, self.spawnAngleVariance, 1)

            start = np.r_[self._object_init[0] + radius * np.cos(angle), self._object_init[1] + radius * np.sin(angle)]
        else:
            start = self._object_init

        # initialize light position
        # TODO check world bounds
        self._light = CircularGradientLight(position=start, radius=.2, bounds: self.world.getBounds)

        # TODO spawn kilobots
        # kilobots start at the light position in a fixed formation
        for i in range(self._num_kilobots):
            if useGridSampling:
                bb_x = 1
                bb_y = 1
                x = start[0] - bb_x * 0.5 + np.random.rand() * bb_x
                y = start[1] - bb_y * 0.5 + np.random.rand() * bb_y
                lightPos = matrix([x, y])

            elif spawnConcentrated:
                x = (start[0] + (10 + i / 4) * kilobotOffsets[i % 4, 0] + 0.25 * (random.random() - 0.5))
                y = (start[1] + (10 + i / 4) * kilobotOffsets[i % 4, 1] + 0.25 * (random.random() - 0.5))
            else:
                sampledRadius = maximum(np.random.normal(0, self.spawnRadiusVariance, 1), 1.1 * radius)
                x = float(start[0] + 1.1 * self.SCALE_REAL_TO_SIM * sampledRadius * cos(start_angles[i]))
                y = float(start[1] + 1.1 * self.SCALE_REAL_TO_SIM * sampledRadius * sin(start_angles[i]))

            kilobot.body.position = vec2(x, y) * self.SCALE_REAL_TO_SIM


