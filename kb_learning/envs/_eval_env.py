import yaml
from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import CircularGradientLight, GradientLight, SimplePhototaxisKilobot

from gym_kilobots.lib import Quad, CornerQuad, Circle, Triangle, LForm, TForm, CForm

from gym import spaces

import numpy as np

from kb_learning.tools import rot_matrix


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
        self.num_kilobots = self.conf.kilobots.num

        self.assembly_policy = None
        self.path = None

        super().__init__()

    def transform_world_to_object_point(self, point, object_idx=0):
        return self._objects[object_idx].get_local_point(point)

    def transform_world_to_object_pose(self, pose, object_idx=0):
        return self._objects[object_idx].get_local_pose(pose)

    def transform_object_to_world_point(self, point, object_idx=0):
        return self._objects[object_idx].get_world_point(point)

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
        elif object_shape in ['corner_quad', 'corner-quad']:
            obj = CornerQuad(width=object_width, height=object_height,
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

    def _draw_on_table(self, screen):
        if self.assembly_policy:
            way_points = self.assembly_policy.way_points
            for i in range(self.assembly_policy.idx, len(way_points)):
                self._draw_object_ghost(screen, self.assembly_policy.way_points[i].pose)
            self._draw_path(screen, [wp.position for wp in way_points[self.assembly_policy.idx:]])

    def _draw_on_top(self, screen):
        if self.path:
            trajectory_points = self.path.trajectory_points[self.path.idx:]
            self._draw_path(screen, trajectory_points, color=(150, 0, 150))

    def _draw_object_ghost(self, screen, pose, width=.15, height=.15):
        # draw the desired pose as grey square
        vertices = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
        vertices *= np.array([[width, height]]) / 2.

        # rotate vertices
        vertices = rot_matrix(pose[2]).dot(vertices.T).T

        # translate vertices
        vertices += pose[None, :2]

        screen.draw_polygon(vertices=vertices, color=(150, 150, 150), filled=True, width=.005)
        screen.draw_polygon(vertices=vertices[0:3], color=(170, 170, 170), width=.005)

    def _draw_path(self, screen, path, color=(150, 150, 150)):
        for p in path:
            screen.draw_circle(position=p, radius=.006, color=color)
        if len(path) < 2:
            return
        start = path[0]
        for p in path[1:]:
            screen.draw_line(start, p, color, width=.003)
            start = p


class EvalEnvConfiguration(yaml.YAMLObject):
    yaml_tag = '!EvalEnv'

    class ObjectConfiguration(yaml.YAMLObject):
        yaml_tag = '!ObjectConf'

        def __init__(self, shape, width, height, init):
            self.shape = shape
            self.width = width
            self.height = height
            self.init = init

    class LightConfiguration(yaml.YAMLObject):
        yaml_tag = '!LightConf'

        def __init__(self, obj_type, init, radius=None):
            self.type = obj_type
            self.init = init
            self.radius = radius

    class KilobotsConfiguration(yaml.YAMLObject):
        yaml_tag = '!KilobotsConf'

        def __init__(self, num, mean, std):
            self.num = num
            self.mean = mean
            self.std = std

    def __init__(self, width, height, resolution, objects, light, kilobots):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.objects = [self.ObjectConfiguration(**obj) for obj in objects]
        self.light = self.LightConfiguration(**light)
        self.kilobots = self.KilobotsConfiguration(**kilobots)


class UnknownObjectException(Exception):
    pass


class UnknownLightTypeException(Exception):
    pass
