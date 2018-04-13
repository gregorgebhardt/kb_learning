from ._object_env import ObjectEnv
# from ._gradient_light_object_env import GradientLightObjectEnv
# from ._dual_light_object_env import DualLightObjectEnv

from .sampler import SARSSampler, ParallelSARSSampler


def _check_parameters(weight: float, num_kilobots: int, object_shape: str, object_width: float, object_height: float,
                      light_type: str, light_radius: float=None):
    if weight is not None:
        assert type(weight) is float, "weight has to be of type float"
        assert .0 <= weight <= 1., "weight has to be in the interval [0.0, 1.0]"
    assert type(num_kilobots) is int, "num_kilobots has to be of type int"
    assert 0 < num_kilobots, "num_kilobots has to be a positive integer."
    assert .0 < object_width, "object_width has to be a positive float"
    assert .0 < object_height, "object_height has to be a positive float"
    assert light_type in ['circular', 'linear'], "light_type has to be in ['circular', 'linear']"
    if light_type == 'circular':
        assert light_radius > .0, "light radius has to be positive."


def register_object_env(weight: float, num_kilobots: int, object_shape: str, object_width: float,
                        object_height: float, light_type: str, light_radius: float=None):
    from gym.envs.registration import register, registry

    _check_parameters(weight, num_kilobots, object_shape, object_width, object_height, light_type, light_radius)

    weight_str = '{:03}'.format(int(weight * 100)) if weight is not None else 'RND'
    light_radius_str = '{:03}'.format(int(light_radius * 100)) if light_radius else ''
    _id = 'ObjectEnv_w{}_kb{}_{}_{:03}x{:03}_{}{}-v0'.format(weight_str, num_kilobots, object_shape,
                                                             int(object_width * 100), int(object_height * 100),
                                                             light_type, light_radius_str)

    if _id in registry.env_specs:
        return _id

    register(id=_id, entry_point='kb_learning.envs:ObjectEnv',
             kwargs=dict(object_shape=object_shape, object_width=object_width, object_height=object_height,
                         num_kilobots=num_kilobots, weight=weight, light_type=light_type, light_radius=light_radius))

    return _id


# def register_gradient_light_object_env(weight: float, num_kilobots: int, object_shape: str,
#                                        object_width: float, object_height: float):
#     from gym.envs.registration import register, registry
#
#     _check_parameters(weight, num_kilobots, object_shape, object_width, object_height)
#
#     _id = 'GradientLightObjectEnv_w{:03}_kb{}_{}_{:03}x{:03}-v0'.format(int(weight * 100), num_kilobots, object_shape,
#                                                          int(object_width * 100), int(object_height * 100))
#
#     if _id in registry.env_specs:
#         return _id
#
#     register(id=_id, entry_point='kb_learning.envs:GradientLightObjectEnv',
#              kwargs=dict(object_shape=object_shape, object_width=object_width, object_height=object_height,
#                          num_kilobots=num_kilobots, weight=weight))
#
#     return _id


# def register_dual_light_complex_object_env(weight: float, num_kilobots: int, object_shape: str,
#                                            object_width: float, object_height: float):
#     from gym.envs.registration import register, registry
#
#     _check_parameters(weight, num_kilobots, object_shape, object_width, object_height)
#
#     _id = 'DualLightObjectEnv_w{:03}_kb{}_{}_{:03}x{:03}-v0'.format(int(weight * 100), num_kilobots, object_shape,
#                                                               int(object_width * 100), int(object_height * 100))
#
#     if _id in registry.env_specs:
#         return _id
#
#     register(id=_id, entry_point='kb_learning.envs:DualLightObjectEnv',
#              kwargs=dict(object_shape=object_shape, object_width=object_width, object_height=object_height,
#                          num_kilobots=num_kilobots, weight=weight))
#
#     return _id
