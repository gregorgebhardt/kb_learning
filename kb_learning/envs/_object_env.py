import numpy as np

from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import SimplePhototaxisKilobot, CircularGradientLight, GradientLight
from gym import spaces

from gym_kilobots.lib import Quad, Circle, Triangle, LForm, TForm, CForm

from kb_learning.tools import compute_robust_mean_swarm_position


class ObjectEnv(KilobotsEnv):
    world_size = world_width, world_height = .8, .8
    screen_size = screen_width, screen_height = 500, 500

    # _light_max_dist = .4
    # _spawn_angle_mean = np.pi
    # _spawn_angle_variance = .5 * np.pi

    def __init__(self,
                 num_kilobots=None,
                 weight=.0,
                 object_shape='quad',
                 object_width=.15,
                 object_height=.15,
                 object_init=None,
                 light_type='circular',
                 light_radius=.2):
        self._weight = weight
        self._num_kilobots = num_kilobots

        if object_init:
            self._object_init = np.array(object_init)
        else:
            self._object_init = np.array([.0, .0, .0])
        self._object_width, self._object_height = object_width, object_height
        self._object_shape = object_shape

        self._sampled_weight = self._weight is None

        self._light_type = light_type
        self._light_radius = light_radius

        self._spawn_type_ratio = 1.
        self._max_spawn_std = .3 * self._light_radius

        # scaling of the differences in x-, y-position and rotation, respectively
        self._scale_vector = np.array([2., 2., .5])
        # cost for translational movements into x, y direction and rotational movements, respectively
        self._cost_vector = np.array([.02, 2., .002])

        self._kilobots_space: spaces.Box = None

        self._light_state_space: spaces.Box = None
        self._light_observation_space: spaces.Box = None

        self._weight_state_space: spaces.Box = None
        self._weight_observation_space: spaces.Box = None

        self._object_state_space: spaces.Box = None
        self._object_observation_space: spaces.Box = None

        super().__init__()

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + (([self._weight],) if self._sampled_weight else tuple())
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {'state': self.get_state()}

    def _transform_position(self, position):
        return self._objects[0].get_local_point(position)

    def get_observation(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
                              + ((self._transform_position(self._light.get_state()),) if self._light_type == 'circular' else tuple())
                              + (([self._weight],) if self._sampled_weight else tuple()))

    def get_reward(self, state, _, new_state):
        obj_pose = state[-3:]
        obj_pose_new = new_state[-3:]

        # punish distance of swarm to object
        swarm_mean = compute_robust_mean_swarm_position(state[:2*self._num_kilobots])
        sq_dist_swarm_object = (swarm_mean**2).sum()

        # punish distance of light to swarm
        sq_dist_swarm_light = ((swarm_mean - state[-2:])**2).sum()

        # compute diff between last and current pose
        obj_pose_diff = obj_pose_new - obj_pose

        # scale diff
        reward = obj_pose_diff * self._scale_vector

        cost = np.abs(obj_pose_diff) * self._cost_vector

        w = self._weight

        r_trans = reward[0] - cost[2]
        r_rot = reward[2] - cost[0]
        return w * r_rot + (1 - w) * r_trans - cost[1] - .001 * sq_dist_swarm_object - .001 * sq_dist_swarm_light

    def has_finished(self, state, action):
        # check if body is in contact with table
        if self.get_objects()[0].collides_with(self.table):
            # print('collision with table')
            return True

        if np.abs(self._objects[0].get_orientation()) > np.pi/2:
            # print('more than quarter rotation')
            return True

        if self._sim_steps >= 2500:
            # print('maximum number of sim steps.')
            return True

        return False

    def _configure_environment(self):
        if self._weight is None:
            self._weight_state_space = spaces.Box(np.array([.0]), np.array([1.]), dtype=np.float32)
            self._weight_observation_space = spaces.Box(np.array([.0]), np.array([1.]), dtype=np.float32)
            self._weight = np.random.rand()

        # initialize object always at (0, 0) with 0 orientation (we do not need to vary the object position and
        # orientation since we will later adapt the state based on the object pose.)
        self._init_object()

        # initialize light position
        self._init_light()

        self._init_kilobots()

        _state_space_low = self._kilobots_space.low
        _state_space_high = self._kilobots_space.high
        if self._light_state_space:
            _state_space_low = np.concatenate((_state_space_low, self._light_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self._light_state_space.high))
        if self._weight_state_space:
            _state_space_low = np.concatenate((_state_space_low, self._weight_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self._weight_state_space.high))
        if self._object_state_space:
            _state_space_low = np.concatenate((_state_space_low, self._object_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self._object_state_space.high))
        self.state_space = spaces.Box(low=_state_space_low, high=_state_space_high, dtype=np.float32)

        _observation_spaces_low = self._kilobots_space.low
        _observation_spaces_high = self._kilobots_space.high
        if self._light_observation_space:
            _observation_spaces_low = np.concatenate((_observation_spaces_low, self._light_observation_space.low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, self._light_observation_space.high))
        if self._weight_observation_space:
            _observation_spaces_low = np.concatenate((_observation_spaces_low, self._weight_observation_space.low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, self._weight_observation_space.high))
        if self._object_observation_space:
            _observation_spaces_low = np.concatenate((_observation_spaces_low, self._object_observation_space.low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, self._object_observation_space.high))
        self.observation_space = spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high, dtype=np.float32)

        # step world once to resolve collisions
        self._step_world()

    def _init_object(self):
        object_shape = self._object_shape.lower()

        if object_shape in ['quad', 'rect']:
            obj = Quad(width=self._object_width, height=self._object_height,
                       position=self._object_init[:2], orientation=self._object_init[2],
                       world=self.world)
        elif object_shape == 'triangle':
            obj = Triangle(width=self._object_width, height=self._object_height,
                           position=self._object_init[:2], orientation=self._object_init[2],
                           world=self.world)
        elif object_shape == 'circle':
            obj = Circle(radius=self._object_width, position=self._object_init[:2],
                         orientation=self._object_init[2], world=self.world)
        elif object_shape == 'l_shape':
            obj = LForm(width=self._object_width, height=self._object_height,
                        position=self._object_init[:2], orientation=self._object_init[2],
                        world=self.world)
        elif object_shape == 't_shape':
            obj = TForm(width=self._object_width, height=self._object_height,
                        position=self._object_init[:2], orientation=self._object_init[2],
                        world=self.world)
        elif object_shape == 'c_shape':
            obj = CForm(width=self._object_width, height=self._object_height,
                        position=self._object_init[:2], orientation=self._object_init[2],
                        world=self.world)
        else:
            raise UnknownObjectException('Shape of form {} not known.'.format(self._object_shape))

        self._objects.append(obj)

        objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -np.inf] * len(self._objects))
        objects_high = np.array([self.world_x_range[1], self.world_y_range[1], np.inf] * len(self._objects))
        self._object_state_space = spaces.Box(low=objects_low, high=objects_high, dtype=np.float64)

    def _init_light(self):
        # determine sampling mode for this episode
        self.__spawn_randomly = np.random.rand() < np.abs(self._spawn_type_ratio)

        if self._light_type == 'circular':
            self._init_circular_light()
        elif self._light_type == 'linear':
            self._init_linear_light()
        else:
            raise UnknownLightTypeException()

        self.action_space = self._light.action_space

    def _init_circular_light(self):
        if self.__spawn_randomly:
            # select light init position randomly as polar coordinates between .25π and 1.75π
            light_init_direction = np.random.rand() * np.pi * 1.5 + np.pi * .25
            light_init_radius = np.abs(np.random.normal() * max(self.world_size) / 2)

            light_init = light_init_radius * np.array([np.cos(light_init_direction), np.sin(light_init_direction)])
        else:
            # otherwise, the light starts above the object
            light_init = self._object_init[:2]

        light_init = np.maximum(light_init, self.world_bounds[0] * .95)
        light_init = np.minimum(light_init, self.world_bounds[1] * .95)

        light_bounds = np.array(self.world_bounds) * 1.2
        action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01

        self._light = CircularGradientLight(position=light_init, radius=self._light_radius,
                                            bounds=light_bounds, action_bounds=action_bounds)

        self._light_observation_space = self._light.observation_space
        self._light_state_space = self._light.observation_space

    def _init_linear_light(self):
        # sample initial angle from a uniform between -pi and pi
        init_angle = np.random.rand() * 2 * np.pi - np.pi
        self._light = GradientLight(angle=init_angle)

        self._light_state_space = self._light.observation_space

    def _init_kilobots(self):
        if self._light_type == 'circular':
            # select the mean position of the swarm to be the position of the light
            spawn_mean = self._light.get_state()
            spawn_std = np.random.rand() * self._max_spawn_std
            kilobot_positions = np.random.normal(loc=spawn_mean, scale=spawn_std, size=(self._num_kilobots, 2))

            # the orientation of the kilobots is chosen randomly
            orientations = 2 * np.pi * np.random.rand(self._num_kilobots)

        elif self._light_type == 'linear':
            # select spawn mean position randomly as polar coordinates between .25π and 1.75π
            spawn_direction = np.random.rand() * np.pi * 1.5 + np.pi * .25
            spawn_radius = np.abs(np.random.normal() * max(self.world_size) / 2)

            spawn_mean = spawn_radius * np.array([np.cos(spawn_direction), np.sin(spawn_direction)])
            spawn_mean = np.maximum(spawn_mean, self.world_bounds[0])
            spawn_mean = np.minimum(spawn_mean, self.world_bounds[1])

            # select a spawn std uniformly between zero and half the world width
            spawn_std = np.random.rand() * min(self.world_size) / 5

            # draw the kilobots positions from a normal with mean and variance selected above
            kilobot_positions = np.random.normal(scale=spawn_std, size=(self._num_kilobots, 2))
            kilobot_positions += spawn_mean

            # the orientation of the kilobots is chosen randomly
            orientations = 2 * np.pi * np.random.rand(self._num_kilobots)
        else:
            raise UnknownLightTypeException()

        # assert for each kilobot that it is within the world bounds and add kilobot to the world
        for position, orientation in zip(kilobot_positions, orientations):
            position = np.maximum(position, self.world_bounds[0] * .95)
            position = np.minimum(position, self.world_bounds[1] * .95)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=position, orientation=orientation,
                                                      light=self._light))

        # construct state space, observation space and action space
        kb_low = np.array([self.world_x_range[0], self.world_y_range[0]] * len(self._kilobots))
        kb_high = np.array([self.world_x_range[1], self.world_y_range[1]] * len(self._kilobots))
        self._kilobots_space = spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)


class UnknownObjectException(Exception):
    pass


class UnknownLightTypeException(Exception):
    pass
