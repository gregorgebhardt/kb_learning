from gym_kilobots.lib import Body, Quad, Circle, Triangle, LForm, TForm, CForm

from ._object_env import ObjectEnv

import numpy as np


class UnknownObjectException(Exception):
    pass


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
        elif object_shape == 'lform':
            return LForm(width=self._object_width, height=self._object_height,
                         position=self._object_init[:2], orientation=self._object_init[2],
                         world=self.world)
        elif object_shape == 'tform':
            return TForm(width=self._object_width, height=self._object_height,
                         position=self._object_init[:2], orientation=self._object_init[2],
                         world=self.world)
        elif object_shape == 'cform':
            return CForm(width=self._object_width, height=self._object_height,
                         position=self._object_init[:2], orientation=self._object_init[2],
                         world=self.world)
        else:
            raise UnknownObjectException('Shape of form {} not known.'.format(self._object_shape))


def ComplexObjectEnvWith(weight, num_kilobots, object_shape, object_width, object_height):
    if object_shape not in ['quad', 'rect', 'triangle', 'circle', 'lform', 'tform', 'cform']:
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
