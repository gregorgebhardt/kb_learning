from gym_kilobots.envs import KilobotsEnv

import abc


def QuadPushingEnvWithWeight(w):
    class QuadPushingEnvWithFactor(QuadPushingEnv):
        @property
        def _weight(self):
            return w

    return QuadPushingEnvWithFactor


class QuadPushingEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., .5

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

    def _reward(self, state, action):
        object_state = state['objects'][0, :]
        object_diff = self._object_init - object_state

        alpha = self.reward_alpha
        beta = self.reward_beta
        da_rot = cos(w * np.pi / 2) * da * self.reward_scale_da - sin(w * np.pi / 2) * dx * self.reward_scale_dx
        dx_rot = sin(w * np.pi / 2) * da * self.reward_scale_da + cos(w * np.pi / 2) * dx * self.reward_scale_dx
        da = da_rot
        dx = dx_rot
        a1 = (np.abs(np.arctan(da / (dx + 1e-6)) / (np.pi / 2))) ** alpha
        x = (1 - a1) ** 2 * (self.reward_c1 * np.power((da ** 2 + dx ** 2), beta)) * (sign(dx) + 1)
        x += (sign(dx) - 1) * dx * self.reward_c2
        return x

    def _configure_environment(self):
        # TODO initialize object position
        self._object_init = None

        # TODO spawn kilobots
