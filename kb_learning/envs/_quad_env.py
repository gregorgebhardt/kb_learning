from gym_kilobots.lib import Body, Quad

from ._object_env import ObjectEnv


class QuadEnv(ObjectEnv):
    def _create_object(self) -> Body:
        return Quad(width=self._object_width, height=self._object_height,
                    position=self._object_init[:2], orientation=self._object_init[2],
                    world=self.world)
