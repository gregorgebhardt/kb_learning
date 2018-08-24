from enum import Enum
from typing import List

import numpy as np

from .ac_reps.spwgp import SparseWeightedGP
from .planning.a_star import AStar, GridWithObstacles
from .planning.assembly_policy import AssemblyPolicy, Path
from .tools import compute_robust_mean_swarm_position


class PolicyState(Enum):
    MOVE_SWARM_TO_OBJECT = 0
    MOVE_OBJECT_ALONG_TRAJECTORY = 1


class Controller:
    def __init__(self, object_pushing_policies: List[SparseWeightedGP],
                 assembly_policy: AssemblyPolicy, config={}):
        # object pushing policy
        self.object_pushing_policies = object_pushing_policies
        # assembly policy
        self.assembly_policy = assembly_policy
        # high level switching state
        self.policy_state = PolicyState.MOVE_SWARM_TO_OBJECT

        self.aux_path = Path()
        self.aux_object_path = Path()

        self.real_width = config.get('real_width', 3)
        self.real_height = config.get('real_height', 2)

        self.target_object_idx = self.assembly_policy.get_target_object_idx()

        self.a_star = AStar(GridWithObstacles(self.real_width, self.real_height, 20))
        self.a_star.grid.potential_radius = 0.5
        self.a_star.grid.potential_factor = 0.05

        # self.kb_mean = [0, 0]
        self.use_subtrajectory = False

    def _compute_auxiliary_target(self, object_position):
        """Compute an auxiliary target for the next push. The auxiliary target is located 0.2 m behind the next
        object to push with respect to the target position for the object."""
        policy_target_pos = self.assembly_policy.get_target_pose()
        push_vector = policy_target_pos[0:2] - object_position[0:2]
        target_pos_offset = 0.2 * (-push_vector) / (np.linalg.norm(push_vector) + 1e-6)

        return object_position + target_pos_offset

    def _update_a_star_object_positions(self, objects_state):
        self.a_star.grid.obstacles = []
        n_objects = int(len(objects_state) / 3)
        for i in range(0, n_objects):
            self.a_star.grid.obstacles.append(objects_state[3 * i:3 * i + 2])

    def _update_policy_state(self, kb_state, objects_state):
        kb_mean = compute_robust_mean_swarm_position(kb_state, 0.8)

        policy_changed = True
        while policy_changed:
            policy_changed = False

            self.target_object_idx = self.assembly_policy.get_target_object_idx()
            obj_position = objects_state[self.target_object_idx:self.target_object_idx + 2]
            obj_orientation = objects_state[self.target_object_idx + 2]

            # move the swarm to the next object
            if self.policy_state == PolicyState.MOVE_SWARM_TO_OBJECT:
                # if no points in the auxiliary policy are left get a new plan from A*
                if len(self.aux_path.trajectory_points) == 0:
                    self._update_a_star_object_positions(objects_state)
                    # get auxiliary position behind object
                    aux_target_pos = self._compute_auxiliary_target(obj_position)

                    # run A* to obtain trajectory and store in the aux_policy
                    print('Started A* search')
                    self.aux_path.trajectory_points = self.a_star.get_path(kb_mean, aux_target_pos)

                    dense_trajectory = []
                    p_prev = None
                    for p in self.aux_path.trajectory_points:
                        if p_prev is None:
                            dense_trajectory.append(p)
                        else:
                            dense_trajectory.append((p_prev + p) * 0.5)
                            dense_trajectory.append(p)
                        p_prev = p
                    self.aux_path.trajectory_points = dense_trajectory
                    print('Finished A* search')
                    policy_changed = True
                # else try updating the current trajectory point, switch to object pushing policy if last point is
                # reached
                if not self.aux_path.update_target_position(kb_mean):
                    self.policy_state = PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY
                    policy_changed = True

            # push the object
            elif self.policy_state == PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY:
                # if distance to object greater than 0.35 m switch to auxiliary policy
                if np.linalg.norm(kb_mean - obj_position) > 0.35:
                    self.policy_state = PolicyState.MOVE_SWARM_TO_OBJECT
                    self.aux_object_path.clear()
                    policy_changed = True
                # else update assembly policy
                else:
                    finished, updated = self.assembly_policy.update_target_position(obj_position, obj_orientation)

                    # if the assembly has not been finished yet, but target point has been updated, reposition swarm
                    if not finished and updated:
                        self.policy_state = PolicyState.MOVE_SWARM_TO_OBJECT
                        self.aux_object_path.clear()
                        policy_changed = True
                    # else continue execution
                    elif not finished:
                        if len(self.aux_object_path.trajectory_points) == 0:
                            self._update_a_star_object_positions(objects_state)
                            # compute a trajectory for the object either
                            # with A*
                            if self.use_subtrajectory:
                                print('Started A* search')
                                self.aux_object_path.trajectory_points = self.a_star.get_path(obj_position[0:2],
                                    self.assembly_policy.get_target_pose())
                                print('Finished A* search')
                            # or by just using a straight path to the target position
                            else:
                                self.aux_object_path = Path()
                                self.aux_object_path.trajectory_points = np.asarray(
                                    [(obj_position[0:2]), (self.assembly_policy.get_target_pose()[0:2])])
                                self.aux_object_path.update_target_position(obj_position[0:2])
                            policy_changed = True
                        else:
                            finished = self.aux_object_path.update_target_position_with_orientation(obj_position[0:2],
                                obj_orientation, self.assembly_policy.get_target_pose_with_tolerances())
                            if finished:
                                policy_changed = True
                    # if assembly policy has been finished return true
                    else:
                        return True
        return False

    def _move_swarm_to_object(self, light_pos):
        # swarm_pos = compute_robust_mean_swarm_position(kb_state, 0.8)
        # if len(self.aux_path.trajectory_points) == 0:
        #     self._update_a_star_object_positions(objects_state)
        #     aux_target_pos = self._compute_auxiliary_target(objects_state)
        #     self.aux_path.trajectory_points = self.a_star.get_path(swarm_pos, aux_target_pos)
        #     dense_trajectory = []
        #     p_prev = None
        #     for p in self.aux_path.trajectory_points:
        #         if p_prev is None:
        #             dense_trajectory.append(p)
        #         else:
        #             dense_trajectory.append((p_prev + p) * 0.5)
        #             dense_trajectory.append(p)
        #         p_prev = p
        #     self.aux_path.trajectory_points = dense_trajectory
        target_pos = self.aux_path.get_target_position()
        a = target_pos - light_pos[0:2]
        return a, target_pos

    def _move_object_along_trajectory(self, kb_state, objects_state, light_pos):
        target_pos = np.append(self.aux_object_path.get_target_position(),
                               self.assembly_policy.get_target_pose()[2:])

        # rotate state
        obj_position = objects_state[self.target_object_idx * 3:self.target_object_idx * 3 + 2]
        obj_orientation = objects_state[self.target_object_idx + 2]

        # positional error between object and target position
        direction = target_pos[0:2] - obj_position
        direction_angle = -np.arctan2(direction[1], direction[0])

        # rotational error between object and target rotation
        rotation = target_pos[2] - obj_orientation
        rot_err = (rotation + np.pi) % (2 * np.pi) - np.pi
        rot_direction = np.sign(rot_err)
        if rot_direction == 0:
            rot_direction = 1

        # rotate state into local frame directed at target
        # TODO combine data into state vector
        s = create_state_vector(light_pos, kb_state, obj_position)
        sx = s.flat[0::2] * np.cos(direction_angle) - s.flat[1::2] * np.sin(direction_angle)
        sy = rot_direction * (
                s.flat[1::2] * np.cos(direction_angle) + s.flat[0::2] * np.sin(direction_angle))
        s.flat[0::2] = sx
        s.flat[1::2] = sy

        # rotation
        rotation = target_pos[2] - obj_orientation

        trans_err = np.linalg.norm(direction)
        rot_err = (rotation + np.pi) % (2 * np.pi) - np.pi
        # controller
        err_ratio = np.abs(rot_err * 0.25) / (np.abs(trans_err) + np.abs(rot_err * 0.25))
        idx = int((len(self.object_pushing_policies) - 1) * err_ratio + 0.5)
        # print(self.err_ratio)
        print('idx', idx, ' val', err_ratio)
        err_ratio = idx / (len(self.object_pushing_policies) - 1)
        # print('rot %2.2f  trans %2.2f' % ((rot_err + np.pi) % (2 * np.pi) - np.pi, trans_err))
        a = self.object_pushing_policies[idx].get_mean(s)

        # rotate action
        ax = a[0, 0] * np.cos(-direction_angle) - rot_direction * a[0, 1] * np.sin(-direction_angle)
        ay = rot_direction * a[0, 1] * np.cos(-direction_angle) + a[0, 0] * np.sin(-direction_angle)
        a = np.array([0.0, 0.0])
        a[0] = ax
        a[1] = ay
        return a, target_pos

    def get_action(self, kb_state, objects_state, light_pos):
        if self._update_policy_state(kb_state, objects_state):
            return None

        a = None
        # determine action
        if self.policy_state == PolicyState.MOVE_SWARM_TO_OBJECT:
            a, target_pos = self._move_swarm_to_object(light_pos)
        elif self.policy_state == PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY:
            a, target_pos = self._move_object_along_trajectory(kb_state, objects_state, light_pos)
            n = np.linalg.norm(a)
            if n > 0.010:
                a = (a * 0.010 / n)

        # take action
        if len(a) != 2:
            raise ValueError('Action size misfit. Got:', len(a), ' Expected: 2')
        return a
