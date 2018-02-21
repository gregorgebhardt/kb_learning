from ._quad_env import QuadEnv

import numpy as np


class LongQuadEnv(QuadEnv):
    _object_height = .6

    def _sample_light_init(self):
        # sample light randomly in the left hemisphere of the environment
        _light_init_x = -np.random.rand() * (self.world_width/2 - self._object_width - .1) + self._object_width
        _light_init_y = np.random.rand() * (self.world_height - .2) - self.world_width/2

        return np.array((_light_init_x, _light_init_y))

    def _sample_kilobot_init_poses(self):
        # TODO implement
        pass
