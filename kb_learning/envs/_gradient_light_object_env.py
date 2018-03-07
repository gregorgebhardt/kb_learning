from ._object_env import ObjectEnv, UnknownObjectException

from gym_kilobots.lib import GradientLight, Body, SimplePhototaxisKilobot, Quad, Circle, Triangle, LForm, TForm, CForm

import abc
import numpy as np


class GradientLightObjectEnv(ObjectEnv):
    world_size = world_width, world_height = .6, .6

    def __init__(self):
        super().__init__()

        self._spawn_type_ratio = .6

    @abc.abstractmethod
    def _create_object(self) -> Body:
        pass

    def get_state(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
                              + tuple(o.get_pose() for o in self._objects))

    def get_observation(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots))

    def _init_light(self):
        self._light = GradientLight()
        self.action_space = self._light.action_space

    def _init_kilobots(self):
        # self.__spawn_randomly = np.random.rand() < self._spawn_type_ratio

        # if self.__spawn_randomly and False:
        #     kilobot_positions = np.random.rand(self._num_kilobots, 2)
        #     kilobot_positions *= np.array(self.world_size) * np.array((.48, .98))
        #     kilobot_positions += self.world_bounds[0]
        #
        # else:
        # corner = np.random.randint(4)
        # up_down = corner // 2
        # left_right = corner % 2
        #
        # corner_coordinates = np.array((self.world_x_range[left_right], self.world_y_range[up_down]))

        spawn_mean = np.random.rand(2) * np.array(self.world_size) * np.array((.98, .98))
        spawn_mean += self.world_bounds[0]
        spawn_variance = np.random.rand() * self.world_width / 2

        kilobot_positions = np.random.normal(scale=spawn_variance, size=(self._num_kilobots, 2))
        kilobot_positions += spawn_mean

        for position in kilobot_positions:
            position = np.maximum(position, self.world_bounds[0] + 0.02)
            position = np.minimum(position, self.world_bounds[1] - 0.02)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=position, light=self._light))


class GradientLightComplexObjectEnv(GradientLightObjectEnv):
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


def GradientLightComplexObjectEnvWith(weight, num_kilobots, object_shape, object_width, object_height):
    if object_shape not in ['quad', 'rect', 'triangle', 'circle', 'l_shape', 't_shape', 'c_shape']:
        raise UnknownObjectException('Shape of form {} not known.'.format(object_shape))

    class _GradientLightComplexObjectEnv(GradientLightComplexObjectEnv):
        _object_width, _object_height = object_width, object_height
        _object_shape = object_shape

        def _configure_environment(self):
            if weight is None:
                self._weight = np.random.rand()
            else:
                self._weight = weight
            self._num_kilobots = num_kilobots
            super()._configure_environment()

    return _GradientLightComplexObjectEnv
