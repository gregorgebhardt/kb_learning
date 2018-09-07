import numpy as np

from kb_learning.envs import ObjectEnv
from kb_learning.tools import rot_matrix, compute_robust_mean_swarm_position

from gym import spaces


class ObjectAbsoluteEnv(ObjectEnv):
    _observe_objects = True

    def __init__(self,
                 num_kilobots=None,
                 object_shape='quad',
                 object_width=.15,
                 object_height=.15,
                 object_init=None,
                 light_type='circular',
                 light_radius=.2):

        super(ObjectAbsoluteEnv, self).__init__(num_kilobots=num_kilobots,
                                                object_shape=object_shape,
                                                object_width=object_width,
                                                object_height=object_height,
                                                object_init=object_init,
                                                light_type=light_type,
                                                light_radius=light_radius)

        self._desired_pose = None

        _state_space_low = self._kilobots_space.low
        _state_space_high = self._kilobots_space.high
        if self._light_state_space:
            _state_space_low = np.concatenate((_state_space_low, self._light_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self._light_state_space.high))
        if self._object_state_space:
            _state_space_low = np.concatenate((_state_space_low, self._object_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self._object_state_space.high))

        self.state_space = spaces.Box(low=_state_space_low, high=_state_space_high, dtype=np.float32)

        _observation_spaces_low = self._kilobots_space.low
        _observation_spaces_high = self._kilobots_space.high
        if self._light_observation_space:
            _observation_spaces_low = np.concatenate((_observation_spaces_low, self._light_observation_space.low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, self._light_observation_space.high))
        if self._object_observation_space:
            # the objects are observed as x, y, sin(theta), cos(theta)
            objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1.] * len(self._objects))
            objects_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1.] * len(self._objects))
            _observation_spaces_low = np.concatenate((_observation_spaces_low, objects_low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, objects_high))
            # # for the desired pose
            # _observation_spaces_low = np.concatenate((_observation_spaces_low, self._object_observation_space.low))
            # _observation_spaces_high = np.concatenate((_observation_spaces_high, self._object_observation_space.high))

        self.observation_space = spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high,
                                            dtype=np.float32)

    def get_desired_pose(self):
        return self._desired_pose

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {'desired_pose': self._desired_pose}

    def get_observation(self):
        if self._light_type == 'circular':
            _light_position = (self._light.get_state(),)
        else:
            _light_position = tuple()

        _object_orientation = self._objects[0].get_orientation()
        _object_sin_cos = ((np.sin(_object_orientation), np.cos(_object_orientation)),)

        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + _light_position
                              # + (self._objects[0].get_pose(),)
                              + (self._objects[0].get_position(),)
                              + _object_sin_cos
                              # + (self._object_desired,)
                              )

    def get_reward(self, state, *args):
        obj_pose = state[-3:]
        reward = .0

        # swarm_mean = compute_robust_mean_swarm_position(state[:2 * self._num_kilobots])

        # punish distance of swarm to object
        # reward -= .01 * ((swarm_mean - self.get_objects()[0].get_position()) ** 2).sum()

        # # punish distance of light to swarm
        # reward -= ((swarm_mean - self.get_light().get_state()) ** 2).sum()

        # compute diff between desired and current pose
        dist_obj_pose = self._desired_pose - obj_pose
        dist_obj_pose[2] = np.abs(np.sin(dist_obj_pose[2] / 2))
        reward -= .01 * dist_obj_pose.dot(dist_obj_pose)

        # print('{} reward: {}'.format(dist_obj_pose[2], reward))

        return reward

    def has_finished(self, state, action):
        # has finished if object reached goal pose with certain ε
        obj_pose = state[-3:]
        dist_obj_pose = self._desired_pose - obj_pose
        dist_obj_pose[2] = np.abs(np.sin(dist_obj_pose[2] / 2))

        sq_error_norm = dist_obj_pose.dot(dist_obj_pose)
        # print('sq_error_norm: {}'.format(sq_error_norm))

        if sq_error_norm < .005:
            return True

        if self._sim_steps >= 3500:
            # print('maximum number of sim steps.')
            return True

        return False

    def _get_init_object_pose(self):
        # sample the initial position uniformly from [-w/4, w/4] and [-h/4, h/4] (w = width, h = height)
        # TODO make area larger?
        _object_init_position = np.random.rand(2) * np.array(self.world_size) / 4 + np.array(self.world_bounds[0]) / 2
        # sample the initial orientation uniformly from [-π, +π] TODO revert to full 2π range
        _object_init_orientation = np.random.rand() * np.pi - np.pi / 2
        self._object_init = np.concatenate((_object_init_position, [_object_init_orientation]))
        
        return self._object_init

    def _get_desired_object_pose(self):
        # # sample the desired position uniformly between [-w/2+ow, w/2-ow] and [-h/2+oh, h/2-oh] (w = width, h = height)
        # _object_desired_position = np.random.rand(2) * self.world_size + np.array(self.world_bounds[0])
        # _object_size = np.array([self._object_width, self._object_height])
        # _object_desired_position = np.maximum(_object_desired_position, self.world_bounds[0] + _object_size)
        # _object_desired_position = np.minimum(_object_desired_position, self.world_bounds[1] - _object_size)
        # # sample the desired orientation uniformly from [-π, +π]
        # _object_desired_orientation = np.random.rand() * 2 * np.pi - np.pi
        # self._object_desired = np.concatenate((_object_desired_position, [_object_desired_orientation]))

        return np.zeros(3)

    def _configure_environment(self):
        self._desired_pose = self._get_desired_object_pose()

        super(ObjectAbsoluteEnv, self)._configure_environment()

    def _draw_on_table(self, screen):
        # draw the desired pose as grey square
        vertices = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
        vertices *= np.array([[self._object_width, self._object_height]]) / 2.

        # rotate vertices
        vertices = rot_matrix(self._desired_pose[2]).dot(vertices.T).T

        # translate vertices
        vertices += self._desired_pose[None, :2]

        screen.draw_polygon(vertices=vertices, color=(200, 200, 200), filled=True, width=.005)
        screen.draw_polygon(vertices=vertices[0:3], color=(220, 200, 200), width=.005)
