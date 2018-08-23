from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import CircularGradientLight, GradientLight, SimplePhototaxisKilobot

from gym_kilobots.lib import Quad, Circle, Triangle, LForm, TForm, CForm

from gym import spaces

import numpy as np


class EvalEnv(KilobotsEnv):
    def __new__(cls, configuration, *args, **kwargs):
        cls.world_width = configuration.width
        cls.world_height = configuration.height
        cls.world_size = cls.world_width, cls.world_height

        cls.screen_width = int(configuration.resolution * configuration.width)
        cls.screen_height = int(configuration.resolution * configuration.height)
        cls.screen_size = cls.screen_width, cls.screen_width

        return super(EvalEnv, cls).__new__(cls, *args, **kwargs)

    def __init__(self, configuration):
        self.conf = configuration
        super().__init__()

    def _configure_environment(self):
        for o in self.conf.objects:
            self._init_object(o.shape, o.width, o.height, o.init)

        objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -np.inf] * len(self._objects))
        objects_high = np.array([self.world_x_range[1], self.world_y_range[1], np.inf] * len(self._objects))
        self._object_state_space = spaces.Box(low=objects_low, high=objects_high, dtype=np.float64)

        if self.conf.light.type == 'circular':
            light_bounds = np.array(self.world_bounds) * 1.2
            action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01

            self._light = CircularGradientLight(position=self.conf.light.init, radius=self.conf.light.radius,
                                                bounds=light_bounds, action_bounds=action_bounds)

            self._light_observation_space = self._light.observation_space
            self._light_state_space = self._light.observation_space

        elif self.conf.light.type == 'linear':
            # sample initial angle from a uniform between -pi and pi
            self._light = GradientLight(angle=self.conf.light.init)

            self._light_state_space = self._light.observation_space
        else:
            raise UnknownLightTypeException()

        self.action_space = self._light.action_space

        self._init_kilobots(self.conf.kilobots.num, self.conf.kilobots.mean, self.conf.kilobots.std)

    def _init_object(self, object_shape, object_width, object_height, object_init):
        if object_shape in ['quad', 'rect']:
            obj = Quad(width=object_width, height=object_height,
                position=object_init[:2], orientation=object_init[2],
                world=self.world)
        elif object_shape == 'triangle':
            obj = Triangle(width=object_width, height=object_height,
                position=object_init[:2], orientation=object_init[2],
                world=self.world)
        elif object_shape == 'circle':
            obj = Circle(radius=object_width, position=object_init[:2],
                orientation=object_init[2], world=self.world)
        elif object_shape == 'l_shape':
            obj = LForm(width=object_width, height=object_height,
                position=object_init[:2], orientation=object_init[2],
                world=self.world)
        elif object_shape == 't_shape':
            obj = TForm(width=object_width, height=object_height,
                position=object_init[:2], orientation=object_init[2],
                world=self.world)
        elif object_shape == 'c_shape':
            obj = CForm(width=object_width, height=object_height,
                position=object_init[:2], orientation=object_init[2],
                world=self.world)
        else:
            raise UnknownObjectException('Shape of form {} not known.'.format(object_shape))

        self._add_object(obj)

    def _init_kilobots(self, num_kilobots, spawn_mean, spawn_std):
        # draw the kilobots positions from a normal with mean and variance selected above
        kilobot_positions = np.random.normal(scale=spawn_std, size=(num_kilobots, 2))
        kilobot_positions += spawn_mean

        # assert for each kilobot that it is within the world bounds and add kilobot to the world
        for position in kilobot_positions:
            position = np.maximum(position, self.world_bounds[0] + 0.02)
            position = np.minimum(position, self.world_bounds[1] - 0.02)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=position, light=self._light))

    def get_reward(self, state, action, new_state):
        return .0


class UnknownObjectException(Exception):
    pass


class UnknownLightTypeException(Exception):
    pass
