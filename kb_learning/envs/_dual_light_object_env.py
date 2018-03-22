from ._object_env import ObjectEnv

from gym_kilobots.lib import CompositeLight, GradientLight, SimplePhototaxisKilobot

import numpy as np


class DualLightObjectEnv(ObjectEnv):
    world_size = world_width, world_height = .6, .6

    def __init__(self):
        super().__init__()

        self._spawn_type_ratio = .6

    def get_state(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
                              + (self._light.get_state(),)
                              + tuple(o.get_pose() for o in self._objects))

    def get_observation(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
                              + (self._light.get_state(),))

    def _init_light(self):
        self._light = CompositeLight([GradientLight(), GradientLight()], reducer=np.max)
        # self._light = CompositeLight([CircularGradientLight(radius=.3, position=np.array([.0, .0]),
        #                                                       bounds=self.world_bounds),
        #                                 CircularGradientLight(radius=.3, position=np.array([.2, .0]),
        #                                                       bounds=self.world_bounds)],
        #                                reducer=max)
        self.action_space = self._light.action_space

    def _init_kilobots(self):
        spawn_mean = np.random.rand(2) * np.array(self.world_size) * np.array((.98, .98))
        spawn_mean += self.world_bounds[0]
        # spawn_variance = np.random.rand() * self.world_width / 2
        spawn_variance = np.random.rand()

        kilobot_positions = np.random.normal(scale=spawn_variance, size=(self._num_kilobots, 2))
        kilobot_positions += spawn_mean

        for position in kilobot_positions:
            position = np.maximum(position, self.world_bounds[0] + 0.02)
            position = np.minimum(position, self.world_bounds[1] - 0.02)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=position, light=self._light))
