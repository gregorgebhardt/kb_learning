from gym_kilobots.lib import Body, Quad, Circle, Triangle, LForm, TForm, CForm

from ._object_env import ObjectEnv, UnknownObjectException

import numpy as np


class ComplexObjectEnv(ObjectEnv):
    _object_shape = None

    def _create_object(self) -> Body:
        object_shape = self._object_shape.lower()

        if object_shape in ['quad', 'rect']:
            return Quad(width=self._object_width, height=self._object_height,
                        position=self._object_init[:2], orientation=self._object_init[2],
                        world=self.world)
        elif object_shape == 'triangle':
            return Triangle(width=self._object_width, height=self._object_height,
                            position=self._object_init[:2], orientation=self._object_init[2],
                            world=self.world)
        elif object_shape == 'circle':
            return Circle(radius=self._object_width, position=self._object_init[:2],
                          orientation=self._object_init[2], world=self.world)
        elif object_shape == 'l_shape':
            return LForm(width=self._object_width, height=self._object_height,
                         position=self._object_init[:2], orientation=self._object_init[2],
                         world=self.world)
        elif object_shape == 't_shape':
            return TForm(width=self._object_width, height=self._object_height,
                         position=self._object_init[:2], orientation=self._object_init[2],
                         world=self.world)
        elif object_shape == 'c_shape':
            return CForm(width=self._object_width, height=self._object_height,
                         position=self._object_init[:2], orientation=self._object_init[2],
                         world=self.world)
        else:
            raise UnknownObjectException('Shape of form {} not known.'.format(self._object_shape))


def ComplexObjectEnvWith(weight, num_kilobots, object_shape, object_width, object_height):
    if object_shape not in ['quad', 'rect', 'triangle', 'circle', 'l_shape', 't_shape', 'c_shape']:
        raise UnknownObjectException('Shape of form {} not known.'.format(object_shape))

    class _ComplexObjectEnv(ComplexObjectEnv):
        _object_width, _object_height = object_width, object_height
        _object_shape = object_shape

        def _configure_environment(self):
            if weight is None:
                self._weight = np.random.rand()
            else:
                self._weight = weight
            self._num_kilobots = num_kilobots
            super()._configure_environment()

    return _ComplexObjectEnv


__registered_complex_object_envs = list()


def register_complex_object_env(weight: float, num_kilobots: int, object_shape: str,
                                object_width: float, object_height: float):
    """Create a subclass of the Quad environment of the Kilobot gym with the given weight and number of
    Kilobots and register the subclass as a gym. Returns the id of the registered gym.

    :param weight: the weight for the Quad environment
    :param num_kilobots: the number of kilobots for the Quad environment
    :param object_shape: the shape of the object in the environment
    :return: the id of the registered environment
    """
    from gym.envs.registration import register
    from ._fixed_weight_quad_env import FixedWeightQuadEnvWith

    assert type(weight) is float, "weight has to be of type float"
    assert .0 <= weight <= 1., "weight has to be in the interval [0.0, 1.0]"
    assert type(num_kilobots) is int, "num_kilobots has to be of type int"
    assert 0 < num_kilobots, "num_kilobots has to be a positive integer."

    _name = 'QuadEnv_w{:03}_kb{}_{}_{:03}x{:03}'.format(int(weight * 100), num_kilobots, object_shape,
                                                        int(object_width * 100), int(object_height * 100))
    _id = 'QuadEnv_w{:03}_kb{}_{}_{:03}x{:03}-v0'.format(int(weight * 100), num_kilobots, object_shape,
                                                         int(object_width * 100), int(object_height * 100))

    if _id in __registered_complex_object_envs:
        return _id
    else:
        __registered_complex_object_envs.append(_id)

    globals()[_name] = ComplexObjectEnvWith(weight, num_kilobots, object_shape, object_width, object_height)
    register(id=_id, entry_point='kb_learning.envs._complex_env:' + _name)

    return _id