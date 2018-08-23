import numpy as np


class AssemblyPolicy:
    def __init__(self, policy_file):
        self.trajectory_points = []
        self.idx = 0
        self._read_policy(policy_file)

    def _read_policy(self, file):
        print('loading', file)
        lines = open(file, 'r')
        for l in lines:
            l_split = l.split()
            p = [int(l_split[0]), float(l_split[1]), float(l_split[2]), float(l_split[3]), float(l_split[4]),
                 float(l_split[5])]
            self.trajectory_points.append(p)

    def get_target_object_idx(self):
        if self.idx == len(self.trajectory_points):
            return -1
        return self.trajectory_points[self.idx][0]

    def update_target_position(self, pos, orientation):
        """update current trajectory point based on distance"""
        # if we are past the last point, return True
        if self.idx == len(self.trajectory_points):
            return True, True

        # compute rotational error
        rot_err = abs(orientation - self.trajectory_points[self.idx][3] + np.pi) % (2 * np.pi) - np.pi
        # compute translational error
        trans_err = np.linalg.norm(self.trajectory_points[self.idx][1:3] - pos)

        # compare rotational and translational error to given required precision in trajectory
        update = False
        if (trans_err < self.trajectory_points[self.idx][4]) & (abs(rot_err) < self.trajectory_points[self.idx][5]):
            self.idx += 1
            update = True
        return False, update

    def get_target_pose(self):
        return self.trajectory_points[self.idx][1:4]

    def get_target_position(self):
        return self.trajectory_points[self.idx][1:3]

    def get_target_orientation(self):
        return self.trajectory_points[self.idx][3]

    def get_target_pose_with_tolerances(self):
        return self.trajectory_points[self.idx][1:]


class Path:
    def __init__(self):
        self.trajectory_points = []
        self.idx = 0

    def clear(self):
        self.trajectory_points = []
        self.idx = 0

    def update_target_position(self, pos):
        """update current trajectory point based on distance"""
        # if we are past the last point, clear trajectory and return True
        if self.idx == len(self.trajectory_points):
            self.trajectory_points = []
            self.idx = 0
            return True

        # compute distance from position to current trajectory point
        trans_err = np.linalg.norm(self.trajectory_points[self.idx][0:2] - pos)
        update = False
        # if we are close enough to the current trajectory point update to next trajectory point
        if trans_err < 0.05:
            self.idx += 1
            update = True

        # check recursively
        if update:
            return self.update_target_position(pos)
        else:
            return False

    def update_target_position_with_orientation(self, pos, orientation, target_position_with_tolerances):
        """update current trajectory point based on translational and rotational distance"""
        # if we are past the last point, clear trajectory and return False
        if self.idx == len(self.trajectory_points):
            self.trajectory_points = []
            self.idx = 0
            return False

        trans_err = np.linalg.norm(self.trajectory_points[self.idx] - pos)
        rot_err = abs(orientation - target_position_with_tolerances[2] + np.pi) % (2 * np.pi) - np.pi
        update = False

        if (trans_err < target_position_with_tolerances[3]) & (abs(rot_err) < target_position_with_tolerances[4]):
            self.idx += 1
            update = True
        return True, update

    def get_target_position(self):
        return self.trajectory_points[self.idx]
