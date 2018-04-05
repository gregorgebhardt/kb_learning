from ._object_env import ObjectEnv
from gym_kilobots.lib import GradientLight, SimplePhototaxisKilobot

import numpy as np


class GradientLightObjectEnv(ObjectEnv):
    world_size = world_width, world_height = .6, .6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._spawn_type_ratio = .6

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + tuple(o.get_pose() for o in self._objects))

    def get_observation(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots))

    def _init_light(self):
        self._light = GradientLight()
        self.action_space = self._light.action_space

    def _init_kilobots(self):
        # select a random mean position uniformly from .98 of the world size
        spawn_mean = np.random.rand(2) * np.array(self.world_size) * np.array((.98, .98))
        # fit this mean position to the world bounds, i.e., shift by lower bounds
        spawn_mean += self.world_bounds[0]
        # select a spawn std uniformly between zero and half the world width
        spawn_std = np.random.rand() * self.world_width / 2

        # draw the kilobots positions from a normal with mean and variance selected above
        kilobot_positions = np.random.normal(scale=spawn_std, size=(self._num_kilobots, 2))
        kilobot_positions += spawn_mean

        # assert for each kilobot that it is within the world bounds and add kilobot to the world
        for position in kilobot_positions:
            position = np.maximum(position, self.world_bounds[0] + 0.02)
            position = np.minimum(position, self.world_bounds[1] - 0.02)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=position, light=self._light))
