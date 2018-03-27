import numpy as np

from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import PhototaxisKilobot, SimplePhototaxisKilobot, CircularGradientLight
from gym import spaces

from gym_kilobots.lib import Body, Quad, Circle, Triangle, LForm, TForm, CForm


class ObjectEnv(KilobotsEnv):
    world_size = world_width, world_height = .8, .8
    screen_size = screen_width, screen_height = 500, 500

    # scaling of the differences in x-, y-position and rotation, respectively
    _scale_vector = np.array([1., 1., 1.])
    # cost for translational movements into x, y direction and rotational movements, respectively
    _cost_vector = np.array([0.1, 0.1, 0.15])

    _object_init = np.array([.0, .0, .0])
    _object_width, _object_height = .15, .15

    _light_radius = .2

    _spawn_type_ratio = .90
    # _sampling_max_radius = .5
    _spawn_radius_variance = .2 * _light_radius
    _light_max_dist = .4
    _spawn_angle_mean = np.pi
    _spawn_angle_variance = .5 * np.pi

    _action_bounds = np.array([-.02, -.02]), np.array([.02, .02])

    def __init__(self, num_kilobots=None, weight=.0, object_shape='quad', object_width=.15, object_height=.15):
        self._weight = weight
        self._num_kilobots = num_kilobots
        self._object_width, self._object_height = object_width, object_height
        self._object_shape = object_shape

        self._sampled_weight = self._weight is None

        super().__init__()

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return self.get_state()

    def _transform_position(self, position):
        return self._objects[0].get_local_point(position)

    def get_observation(self):
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
                              + (self._transform_position(self._light.get_state()),)
                              + (([self._weight],) if self._sampled_weight else tuple()))

    def get_reward(self, state, _, new_state):
        obj_pose = state[-3:]
        obj_pose_new = new_state[-3:]

        # compute diff between last and current pose
        obj_pose_diff = obj_pose_new - obj_pose

        # scale diff
        obj_pose_diff_scaled = obj_pose_diff * self._scale_vector

        cost = np.abs(obj_pose_diff_scaled) * self._cost_vector

        w = self._weight

        r_trans = obj_pose_diff_scaled[0] - cost[2]
        r_rot = obj_pose_diff_scaled[2] - cost[0]
        return w * r_rot + (1 - w) * r_trans - cost[1]

    def _configure_environment(self):
        if self._weight is None:
            self._weight = np.random.rand()

        # initialize object always at (0, 0) with 0 orientation (we do not need to vary the object position and
        # orientation since we will later adapt the state based on the object pose.)
        self._objects.append(self._create_object())

        # initialize light position
        self._init_light()

        self._init_kilobots()

        # construct state space, observation space and action space
        kb_low = np.array([self.world_x_range[0], self.world_y_range[0]] * len(self._kilobots))
        kb_high = np.array([self.world_x_range[1], self.world_y_range[1]] * len(self._kilobots))
        kilobots_space = spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)

        objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -np.inf] * len(self._objects))
        objects_high = np.array([self.world_x_range[1], self.world_y_range[1], np.inf] * len(self._objects))
        objects_space = spaces.Box(low=objects_low, high=objects_high, dtype=np.float64)

        self.state_space = spaces.Tuple([kilobots_space, self._light.observation_space, objects_space])
        self.observation_space = spaces.Tuple([kilobots_space, self._light.observation_space])
        self.action_space = spaces.Box(*self._action_bounds, dtype=np.float64)

        # step world once to resolve collisions
        self._step_world()

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

    def _init_light(self):
        # determine sampling mode for this episode
        self.__spawn_randomly = np.random.rand() < np.abs(self._spawn_type_ratio)

        if self.__spawn_randomly:
            # if we spawn the light randomly, we select random polar coordinates for the light
            # the radius is half the object width plus a uniform sample from the range [0, light_max_dist)
            # light_radius = (0.5 * self._object_width + self._light_max_dist * np.random.rand())
            # the angle is sampled normally with spawn_angle_mean and spawn_angle_variance
            # angle = np.random.normal(self._spawn_angle_mean, self._spawn_angle_variance)

            # light_init = self._object_init[:2] + light_radius * np.array([np.cos(angle), np.sin(angle)])

            light_init = np.random.rand(2) * np.array(self.world_size) * np.array((.98, .98))
            light_init += self.world_bounds[0]
        else:
            # otherwise, the light starts above the object
            light_init = self._object_init[:2]

        light_init = np.maximum(light_init, self.world_bounds[0])
        light_init = np.minimum(light_init, self.world_bounds[1])

        light_bounds = np.array(self.world_bounds) * 1.2

        self._light = CircularGradientLight(position=light_init, radius=self._light_radius,
                                            bounds=light_bounds, action_bounds=self._action_bounds)

    def _init_kilobots(self):
        # kilobots start at the light position in a slightly random formation
        kb_radius = PhototaxisKilobot.get_radius()
        kilobot_offsets = np.array([[-kb_radius, 0], [kb_radius, 0], [0, kb_radius], [0, -kb_radius]])
        start_angles = 2 * np.pi * np.random.randn(self._num_kilobots)

        kilobot_poses = []

        for i in range(self._num_kilobots):
            if self.__spawn_randomly:
                position = (1 + i / 4) * kilobot_offsets[i % 4, :] + .25 * (np.random.rand(2) - .5)
                # x = (start[0] + (10 + i / 4) * kilobot_offsets[i % 4, 0] + 0.25 * (np.random.rand() - 0.5))
                # y = (start[1] + (10 + i / 4) * kilobot_offsets[i % 4, 1] + 0.25 * (np.random.rand() - 0.5))
            else:
                sampled_radius = np.maximum(np.random.normal(0, self._spawn_radius_variance), 1.1 * kb_radius)
                sampled_radius = np.minimum(sampled_radius, self._light_radius)
                # x = float(start[0] + 1.1 * sampled_radius * np.cos(start_angles[i]))
                # y = float(start[1] + 1.1 * sampled_radius * np.sin(start_angles[i]))
                position = sampled_radius * np.r_[np.cos(start_angles[i]), np.sin(start_angles[i])]

            # the orientation of the kilobots is chosen randomly
            orientation = 2 * np.pi * np.random.rand()

            kilobot_poses.append((position, orientation))

        for position, orientation in kilobot_poses:
            position += self._light.get_state()
            position = np.maximum(position, self.world_bounds[0] + 1.5 * kb_radius)
            position = np.minimum(position, self.world_bounds[1] - 1.5 * kb_radius)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=position, orientation=orientation,
                                                      light=self._light))


class UnknownObjectException(Exception):
    pass
