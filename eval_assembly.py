import numpy as np
import yaml
import enum
import pickle
from kb_learning.envs import EvalEnv
from kb_learning.ac_reps.spwgp import SparseWeightedGP
from kb_learning.planning import AStar, Path, GridWithObstacles, AssemblyPolicy, AssemblyPolicyConf


class PolicyState(enum.Enum):
    MOVE_SWARM_TO_OBJECT = 0
    MOVE_OBJECT_ALONG_TRAJECTORY = 1


class HierarchicalKilobotController:
    def __init__(self, policies, assembly_policy: AssemblyPolicy, env: EvalEnv):
        self.policies = policies
        self.assembly_policy = assembly_policy
        self.aux_policy = Path()
        self.object_aux_policy = Path()
        self.a_star = AStar(GridWithObstacles(env.world_width, env.world_height, 40), offset=-env.world_bounds[0])
        self.a_star.grid.potential_radius = 0.5
        self.a_star.grid.potential_factor = 10.5

        self.env = env

        self.push_objects = []
        self.num_push_objects = 0

        self.target_object_idx = self.assembly_policy.get_target_object_idx()

        self.policy_state = PolicyState.MOVE_SWARM_TO_OBJECT

    def get_swarm_position(self, observation):
        swarm_mean = observation[:self.env.num_kilobots * 2].reshape(self.env.num_kilobots, 2).mean(axis=0)
        return swarm_mean

    def compute_auxiliary_target(self, objects):
        obj_pos = objects[self.target_object_idx]
        policy_target_pos = self.assembly_policy.get_target_position()
        target_pos_offset = 0.2 * (obj_pos[0:2] - policy_target_pos) / (np.linalg.norm(obj_pos[0:2] - policy_target_pos) + 1e-6)
        return obj_pos[0:2] + target_pos_offset

    def update_a_star_object_positions(self, objects):
        self.a_star.grid.obstacles = []
        for i, obj in enumerate(objects):
            obj_pos = obj[:2]
            self.a_star.grid.obstacles.append(obj_pos)

    def move_swarm_to_object(self, kilobots, objects, light):
        # compute position of swarm (mean of all agents)
        swarm_pos = kilobots.mean(axis=0)

        # is auxiliary policy is empty, compute new path with A*
        if len(self.aux_policy.trajectory_points) == 0:
            self.update_a_star_object_positions(objects)
            aux_target_pos = self.compute_auxiliary_target(objects)
            self.aux_policy.trajectory_points = self.a_star.get_path(swarm_pos, aux_target_pos)

        target_pos = self.aux_policy.get_target_position()
        action = target_pos - light[0:2]
        return action

    def move_object_along_trajectory(self, kilobots, objects, light):
        self.target_object_idx = self.assembly_policy.get_target_object_idx()

        obj_pos = objects[self.target_object_idx, :2]
        obj_orientation = objects[self.target_object_idx, 2]

        # transform current state
        # transformed_light = self.env.transform_world_to_object_point(light, self.target_object_idx)
        # transformed_kilobots = np.array([self.env.transform_world_to_object_point(kb[:2], self.target_object_idx) \
        #                                 for kb in kilobots])

        translated_light = light - objects[self.target_object_idx, :2]
        translated_kilobots = np.array([kb[:2] - objects[self.target_object_idx, :2] for kb in kilobots])
        translated_state = np.concatenate([translated_kilobots, np.array([translated_light])], axis=0)

        # rotate state
        direction = self.object_aux_policy.get_target_position() - obj_pos
        direction_angle = -np.arctan2(direction[1], direction[0])

        rotation_error = obj_orientation - self.assembly_policy.get_target_orientation()
        # rotation_error = (rotation + np.pi) % (2 * np.pi) - np.pi
        rotation_direction = -np.sign(rotation_error)
        if rotation_direction == 0:
            rotation_direction = 1

        # print('rotation_directions: {}'.format(rotation_direction))

        rotation_matrix = np.array([[np.cos(direction_angle), -np.sin(direction_angle)],
                                    [np.sin(direction_angle), np.cos(direction_angle)]])

        # rotate (and if necessary mirror) the translated state
        transformed_state = translated_state.dot(rotation_matrix.T) * np.array([[1., rotation_direction]])
        # transformed_state *= np.array([[1., rotation_direction]])

        print('local kb mean: {}'.format(transformed_state[:-1, :].mean(axis=0)))

        # compute translation and rotation error to estimate closest w-factor
        translation_err = np.maximum(np.linalg.norm(direction) - self.assembly_policy.get_position_accuracy(), 0)
        rotation_error = np.sign(rotation_error) * np.maximum(np.abs(rotation_error) -
                                                              self.assembly_policy.get_orientation_accuracy(), 0)

        # controller
        error_ratio = np.abs(rotation_error * 0.1) / (np.abs(translation_err) + np.abs(rotation_error * 0.1) + 0.0001)
        policy_idx = int((len(self.policies) - 1) * error_ratio + 0.5)
        w_factor = policy_idx / (len(self.policies) - 1)
        print("computed error_ratio: {}, policy index: {}, chosen w-factor: {}".format(error_ratio, policy_idx,
                                                                                       w_factor))
        print('rot error: {:2.2f}, trans error: {:2.2f}'.format((rotation_error + np.pi) % (2 * np.pi) - np.pi,
                                                                translation_err))
        action = self.policies[policy_idx].get_mean(transformed_state.reshape(1, -1))

        # rotate action
        rotation_matrix = np.array([[np.cos(-direction_angle), -np.sin(-direction_angle) * rotation_direction],
                                    [np.sin(-direction_angle), np.cos(-direction_angle) * rotation_direction]])
        transformed_action = action.dot(rotation_matrix.T).flatten()
        # action = self.env.transform_object_to_world_point(action.flatten(), self.target_object_idx)

        print('action: {} transformed_action: {}'.format(action, transformed_action))

        return np.array(transformed_action)

    def update_policy_state(self, kilobots: np.ndarray, objects: np.ndarray, light: np.ndarray):
        swarm_pos = kilobots[:, :2].mean(axis=0)

        policy_changed = True
        while policy_changed:
            policy_changed = False
            obj_position = objects[self.target_object_idx, :2]
            obj_orientation = objects[self.target_object_idx, 2]

            if self.assembly_policy.done():
                return True

            if self.policy_state == PolicyState.MOVE_SWARM_TO_OBJECT:
                if len(self.aux_policy.trajectory_points) == 0:
                    self.update_a_star_object_positions(objects)
                    aux_target_pos = self.compute_auxiliary_target(objects)
                    print('Started A* search')
                    self.aux_policy.trajectory_points = self.a_star.get_path(swarm_pos, aux_target_pos)
                    print('Finished A* search')
                    policy_changed = True
                elif self.aux_policy.update_target_position(swarm_pos):
                    self.policy_state = PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY
                    policy_changed = True
            elif self.policy_state == PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY:
                if np.linalg.norm(swarm_pos - obj_position[0:2]) > 0.5:
                    self.policy_state = PolicyState.MOVE_SWARM_TO_OBJECT
                    self.object_aux_policy.clear()
                    policy_changed = True
                else:
                    finished, updated = self.assembly_policy.update_target_position(obj_position, obj_orientation)
                    if (not finished) and updated:
                        self.policy_state = PolicyState.MOVE_SWARM_TO_OBJECT
                        self.object_aux_policy.clear()
                        policy_changed = True
                    elif finished:
                        return True
                    elif not finished:
                        if len(self.object_aux_policy.trajectory_points) == 0:
                            self.update_a_star_object_positions(objects)
                            print('Started A* search')
                            self.object_aux_policy.trajectory_points = self.a_star.get_path(obj_position,
                                                                                            self.assembly_policy.get_target_position())
                            print('Finished A* search')
                            self.object_aux_policy.update_target_position(obj_position)
                            policy_changed = True
                        else:
                            finished = self.object_aux_policy.update_target_position_with_orientation(obj_position,
                                          obj_orientation, self.assembly_policy.get_target_pose_with_tolerances())
                            if finished:
                                policy_changed = True
        return False

    def compute_action(self, kilobots, objects, light):
        if self.update_policy_state(kilobots, objects, light):
            return [.0, .0]

        if self.policy_state == PolicyState.MOVE_SWARM_TO_OBJECT:
            self.env.path = self.aux_policy
            return self.move_swarm_to_object(kilobots, objects, light)
        elif self.policy_state == PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY:
            self.env.path = self.object_aux_policy
            return self.move_object_along_trajectory(kilobots, objects, light)


def main():
    configuration = '''
    !EvalEnv
    width: 2.0
    height: 1.5
    resolution: 500

    objects:
      - !ObjectConf
        shape: corner-quad
        width: .15
        height: .15
        init: [-0.6, -0.5, .2]

    light: !LightConf
      type: circular
      radius: .2
      init: [.0, .0]

    kilobots: !KilobotsConf
      num: 10
      mean: [-0.9, -0.5]
      std: .03
    '''

    conf = yaml.load(configuration)

    env = EvalEnv(conf)
    obs = env.reset()
    env.render()

    # load policies
    policies = []
    for w_factor in [.0, .5, 1.]:
        policy_file = 'experiments/data/fixed_weight/square/sampling.w_factor{}/log/02/policy_it49.pkl'.format(w_factor)
        with open(policy_file, 'rb') as f:
            policy_dict = pickle.load(f)
            policy = SparseWeightedGP.from_dict(policy_dict)
            policies.append(policy)

    # create assembly policy
    center = np.array([-.5, .5])
    thetas = np.linspace(1.5 * np.pi, 2 * np.pi, 20)

    way_points = []
    for theta in thetas:
        p = np.array([np.cos(theta), np.sin(theta)]) + center
        apc = AssemblyPolicyConf.AssemblyWayPoint(0, x=p[0], y=p[1], theta=theta-1.5*np.pi)
        way_points.append(apc)
    assembly_policy_config = AssemblyPolicyConf(way_points)

    assembly_policy = AssemblyPolicy(assembly_policy_config)
    env.assembly_policy = assembly_policy

    controller = HierarchicalKilobotController(policies, assembly_policy, env)
    reward_sum = 0

    while not controller.assembly_policy.done():
        action = controller.compute_action(**obs)
        env.target_pose = controller.assembly_policy.get_target_pose()

        obs, reward, done, info = env.step(action)
        reward_sum += reward

        if done:
            return


if __name__ == '__main__':
    main()
