import numpy as np

from kb_learning.envs import ObjectEnv
from kb_learning.tools import compute_robust_mean_swarm_position, rot_matrix


class ObjectAbsoluteEnv(ObjectEnv):

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

        self._object_desired = None

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {'desired_pose': self._object_desired}

    def get_observation(self):
        if self._light_type == 'circular':
            _light_position = (self._transform_position(self._light.get_state()),)
        else:
            _light_position = tuple()

        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
                              + _light_position
                              + tuple(o.get_pose() for o in self._objects))

    def get_reward(self, state, *args):
        obj_pose = state[-3:]

        # punish distance of swarm to object
        swarm_mean = compute_robust_mean_swarm_position(state[:2 * self._num_kilobots])
        swarm_object_L2 = (swarm_mean ** 2).sum()

        # punish distance of light to swarm
        swarm_light_L2 = ((swarm_mean - self.get_light().get_state()) ** 2).sum()

        # compute diff between desired and current pose
        dist_obj_pose = self._object_desired - obj_pose
        obj_L2 = dist_obj_pose.dot(dist_obj_pose)

        # print('reward: {}'.format((swarm_object_L2, swarm_light_L2, obj_L2)))

        return swarm_object_L2 + swarm_light_L2 + .01 * obj_L2

    def has_finished(self, state, action):
        # has finished if object reached goal pose with certain ε
        obj_pose = state[-3:]
        obj_pose_diff = self._object_desired - obj_pose

        sq_error_norm = obj_pose_diff.dot(obj_pose_diff)
        # print('sq_error_norm: {}'.format(sq_error_norm))

        return sq_error_norm < .001

    def _configure_environment(self):
        # sample the initial position uniformly from [-w/4, w/4] and [-h/4, h/4] (w = width, h = height)
        _object_init_position = np.random.rand(2) * np.array(self.world_size) / 2 + np.array(self.world_bounds[0]) / 2
        # sample the initial orientation uniformly from [.0, 2π]
        _object_init_orientation = np.random.rand() * 2 * np.pi
        self._object_init = np.concatenate((_object_init_position, [_object_init_orientation]))

        # sample the desired position uniformly between [-w/2+ow, w/2-ow] and [-h/2+oh, h/2-oh] (w = width, h = height)
        _object_desired_position = np.random.rand(2) * self.world_size + np.array(self.world_bounds[0])
        _object_size = np.array([self._object_width, self._object_height])
        _object_desired_position = np.maximum(_object_desired_position, self.world_bounds[0] + _object_size)
        _object_desired_position = np.minimum(_object_desired_position, self.world_bounds[1] - _object_size)
        # sample the desired orientation uniformly from [.0, 2π]
        _object_desired_orientation = np.random.rand() * 2 * np.pi
        self._object_desired = np.concatenate((_object_desired_position, [_object_desired_orientation]))

        super(ObjectAbsoluteEnv, self)._configure_environment()

    def _draw_on_table(self, screen):
        # draw the desired pose as grey square
        vertices = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
        vertices *= np.array([[self._object_width, self._object_height]]) / 2.

        # rotate vertices
        vertices = rot_matrix(self._object_desired[2]).dot(vertices.T).T

        # translate vertices
        vertices += self._object_desired[None, :2]

        screen.draw_polygon(vertices=vertices, color=(200, 200, 200), filled=True, width=.005)
        screen.draw_polygon(vertices=vertices[0:3], color=(220, 200, 200), width=.005)
