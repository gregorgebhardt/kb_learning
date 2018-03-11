from ._quad_env import QuadEnv


def FixedWeightQuadEnvWith(weight, num_kilobots):
    class _QuadEnv(QuadEnv):
        def _configure_environment(self):
            self._weight = weight
            self._num_kilobots = num_kilobots
            super()._configure_environment()

    return _QuadEnv


class FixedWeightQuadEnv_w000_kb15(QuadEnv):
    def _configure_environment(self):
        self._weight = .0
        self._num_kilobots = 15
        super()._configure_environment()


class FixedWeightQuadEnv_w025_kb15(QuadEnv):
    def _configure_environment(self):
        self._weight = .25
        self._num_kilobots = 15
        super()._configure_environment()


class FixedWeightQuadEnv_w050_kb15(QuadEnv):
    def _configure_environment(self):
        self._weight = .5
        self._num_kilobots = 15
        super()._configure_environment()


class FixedWeightQuadEnv_w075_kb15(QuadEnv):
    def _configure_environment(self):
        self._weight = .75
        self._num_kilobots = 15
        super()._configure_environment()


class FixedWeightQuadEnv_w100_kb15(QuadEnv):
    def _configure_environment(self):
        self._weight = 1.
        self._num_kilobots = 15
        super()._configure_environment()
