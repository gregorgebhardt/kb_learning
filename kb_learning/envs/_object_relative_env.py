import numpy as np

from gym import spaces

from kb_learning.envs import ObjectEnv
from kb_learning.tools import compute_robust_mean_swarm_position


class ObjectRelativeEnv(ObjectEnv):
    def __init__(self,
                 num_kilobots=None,
                 weight=.0,
                 object_shape='quad',
                 object_width=.15,
                 object_height=.15,
                 object_init=None,
                 light_type='circular',
                 light_radius=.2,
                 done_after_steps=125):

        self._weight = weight
        self._sampled_weight = self._weight is None

        # scaling of the differences in x-, y-position and rotation, respectively
        self._scale_vector = np.array([250., 1., 60.])
        # cost for translational movements into x, y direction and rotational movements, respectively
        self._cost_vector = np.array([150, 150., 5.])

        self._done_after_steps = done_after_steps

        super(ObjectRelativeEnv, self).__init__(num_kilobots=num_kilobots,
                                                object_shape=object_shape,
                                                object_width=object_width,
                                                object_height=object_height,
                                                object_init=object_init,
                                                light_type=light_type,
                                                light_radius=light_radius)

        if self._weight is None:
            self._weight_state_space = spaces.Box(np.array([.0]), np.array([1.]), dtype=np.float32)
            self._weight_observation_space = spaces.Box(np.array([.0]), np.array([1.]), dtype=np.float32)
            self._weight = np.random.rand()

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

        self.observation_space = spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high,
                                            dtype=np.float32)

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + (([self._weight],) if self._sampled_weight else tuple())
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {'state': self.get_state()}

    def get_observation(self):
        if self._light_type == 'circular':
            light_observation = (self._transform_position(self._light.get_state()),)
            # light_observation = (self._light.get_state(),)
        else:
            light_observation = tuple()
        return np.concatenate(tuple(self._transform_position(k.get_position()) for k in self._kilobots)
        # return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + light_observation
                              + (([self._weight],) if self._sampled_weight else tuple()))

    def get_reward(self, state, _, new_state):
        obj_pose = state[-3:]
        obj_pose_new = new_state[-3:]

        # compute diff between last and current pose
        obj_pose_diff = obj_pose_new - obj_pose
        # obj_pose_diff = obj_pose_new

        # scale diff
        obj_reward = obj_pose_diff * self._scale_vector

        obj_cost = np.abs(obj_pose_diff) * self._cost_vector

        w = self._weight

        r_trans = obj_reward[0] - obj_cost[2]
        r_rot = obj_reward[2] - obj_cost[0]

        reward = w * r_rot + (1 - w) * r_trans - obj_cost[1]

        # # punish distance of swarm to object
        # swarm_mean = compute_robust_mean_swarm_position(state[:2*self._num_kilobots])
        # sq_dist_swarm_object = (swarm_mean**2).sum()
        # reward -= .001 * sq_dist_swarm_object

        # # punish distance of light to swarm
        # sq_dist_swarm_light = ((swarm_mean - state[-2:])**2).sum()
        # reward -= .001 * sq_dist_swarm_light

        return reward

    def has_finished(self, state, action):
        # check if body is in contact with table
        if self.get_objects()[0].collides_with(self.table):
            # print('collision with table')
            return True

        if np.abs(self._objects[0].get_orientation()) > np.pi/2:
            # print('more than quarter rotation')
            return True

        if self._sim_steps >= self._done_after_steps * self._steps_per_action:
            # print('maximum number of sim steps.')
            return True

        return False
