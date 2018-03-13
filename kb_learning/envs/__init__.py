from gym.envs.registration import register

# TODO replace with dynamic registration
from ._fixed_weight_quad_env import FixedWeightQuadEnv_w000_kb15, FixedWeightQuadEnv_w025_kb15, \
    FixedWeightQuadEnv_w050_kb15, FixedWeightQuadEnv_w075_kb15, FixedWeightQuadEnv_w100_kb15

from ._sample_weight_quad_env import SampleWeightQuadEnv_kb15

from ._complex_env import register_complex_object_env
from ._gradient_light_object_env import register_gradient_light_complex_object_env
from ._dual_light_object_env import register_dual_light_complex_object_env


class QuadEnvNotAvailableError(Exception):
    def __init__(self, weight, num_kilobots):
        self.message = "No quad-pushing environment for w = {} and #kb = {} availabe.".format(weight, num_kilobots)
        self.weight = weight
        self.num_kilobots = num_kilobots


__fixed_weight_gym_classes = [FixedWeightQuadEnv_w000_kb15, FixedWeightQuadEnv_w025_kb15, FixedWeightQuadEnv_w050_kb15,
                              FixedWeightQuadEnv_w075_kb15, FixedWeightQuadEnv_w100_kb15]

__fixed_weight_gym_names = ['FixedWeightQuadEnv_w000_kb15-v0',
                            'FixedWeightQuadEnv_w025_kb15-v0',
                            'FixedWeightQuadEnv_w050_kb15-v0',
                            'FixedWeightQuadEnv_w075_kb15-v0',
                            'FixedWeightQuadEnv_w100_kb15-v0']

for _cl, _id in zip(__fixed_weight_gym_classes, __fixed_weight_gym_names):
    register(_id, entry_point='kb_learning.envs:' + _cl.__name__)


def get_fixed_weight_quad_env(weight: float, num_kilobots: int):
    _id = 'FixedWeightQuadEnv_w{:03}_kb{}-v0'.format(int(weight * 100), num_kilobots)
    if _id in __fixed_weight_gym_names:
        return _id
    else:
        raise QuadEnvNotAvailableError(weight, num_kilobots)


__sample_weight_gym_classes = [SampleWeightQuadEnv_kb15]

__sample_weight_gym_names = ['SampleWeightQuadEnv_kb15-v0']

for _cl, _id in zip(__sample_weight_gym_classes, __sample_weight_gym_names):
    register(_id, entry_point='kb_learning.envs:' + _cl.__name__)


def get_sample_weight_quad_env(num_kilobots: int):
    _id = 'SampleWeightQuadEnv_kb{}-v0'.format(num_kilobots)
    if _id in __sample_weight_gym_names:
        return _id
    else:
        raise QuadEnvNotAvailableError('sampling', num_kilobots)
