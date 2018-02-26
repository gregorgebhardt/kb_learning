import numpy as np
from ._quad_env import QuadEnv
import abc


class SampleWeightQuadEnv(QuadEnv):
    # def get_state(self):
    #     return np.concatenate((super().get_state()))

    # we return the weight only in the observation, since we use the get_state method for computing the reward
    def get_observation(self):
        return np.concatenate((super().get_observation(), [self._weight]))

    def _configure_environment(self):
        self._weight = np.random.rand()
        super()._configure_environment()


def SampleWeightQuadEnvWith(num_kilobots):
    class _SampleWeightQuadEnv(SampleWeightQuadEnv):
        def _configure_environment(self):
            self._num_kilobots = num_kilobots
            super()._configure_environment()

    return _SampleWeightQuadEnv


class SampleWeightQuadEnv_kb15(SampleWeightQuadEnv):
    def _configure_environment(self):
        self._num_kilobots = 15
        super()._configure_environment()
