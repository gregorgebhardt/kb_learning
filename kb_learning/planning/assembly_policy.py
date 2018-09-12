import numpy as np
import yaml


class AssemblyPolicyConf(yaml.YAMLObject):
    yaml_tag = '!AssemblyPolicy'

    class AssemblyWayPoint(yaml.YAMLObject):
        yaml_tag = '!AssemblyWayPoint'

        def __init__(self, object_idx, x, y, theta, position_accuracy=0.05, orientation_accuracy=0.05):
            self.object_idx = object_idx
            self.x = x
            self.y = y
            self.theta = theta
            self.position = np.array([self.x, self.y])
            self.pose = np.array([self.x, self.y, self.theta])
            self.position_accuracy = position_accuracy
            self.orientation_accuracy = orientation_accuracy

    def __init__(self, way_points):
        if len(way_points) > 0 and type(way_points[0]) == self.AssemblyWayPoint:
            self.way_points = way_points
        else:
            self.way_points = [self.AssemblyWayPoint(wp) for wp in way_points]


class AssemblyPolicy:
    def __init__(self, configuration: AssemblyPolicyConf):
        self.idx = 0
        self.way_points = configuration.way_points

    def get_target_object_idx(self):
        if self.idx == len(self.way_points):
            return -1
        return self.way_points[self.idx].object_idx

    def done(self):
        return self.idx == len(self.way_points)

    def update_target_position(self, pos, orientation):
        """update current trajectory point based on distance"""
        # if we are past the last point, return True
        if self.idx == len(self.way_points):
            return True, True

        # compute rotational error
        rot_err = abs(orientation - self.way_points[self.idx].theta + np.pi) % (2 * np.pi) - np.pi
        # compute translational error
        trans_err = np.linalg.norm(self.way_points[self.idx].position - pos)

        # compare rotational and translational error to given required precision in trajectory
        update = False
        if (trans_err < self.way_points[self.idx].position_accuracy) & \
                (abs(rot_err) < self.way_points[self.idx].orientation_accuracy):
            self.idx += 1
            update = True
        return False, update

    def get_target_pose(self):
        return self.way_points[self.idx].pose

    def get_target_position(self):
        return self.way_points[self.idx].position

    def get_target_orientation(self):
        return self.way_points[self.idx].theta

    def get_target_pose_with_tolerances(self):
        return np.hstack((self.way_points[self.idx].pose,
                          self.way_points[self.idx].position_accuracy,
                          self.way_points[self.idx].orientation_accuracy))

    def get_position_accuracy(self):
        return self.way_points[self.idx].position_accuracy

    def get_orientation_accuracy(self):
        return self.way_points[self.idx].orientation_accuracy


class Path:
    def __init__(self):
        self.trajectory_points = []
        self.idx = 0

    def clear(self):
        self.trajectory_points = []
        self.idx = 0

    def update_target_position(self, pos, trans_threshold=.025):
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
        if trans_err < trans_threshold:
            self.idx += 1
            # check recursively
            return self.update_target_position(pos)

        # check if object is closer to target than current position and update idx
        dist_obj_last = np.linalg.norm(self.trajectory_points[-1][0:2] - pos)
        dist_current_last = np.linalg.norm(self.trajectory_points[-1][0:2] - self.trajectory_points[self.idx][0:2])
        if dist_obj_last < dist_current_last:
            self.idx += 1
            # check recursively
            return self.update_target_position(pos)

        return False

    def update_target_position_with_orientation(self, pos, orientation, target_position_with_tolerances):
        """update current trajectory point based on translational and rotational distance"""
        # if we are past the last point, clear trajectory and return True
        if self.idx == len(self.trajectory_points):
            self.trajectory_points = []
            self.idx = 0
            return True

        trans_err = np.linalg.norm(self.trajectory_points[self.idx] - pos)
        rot_err = abs(orientation - target_position_with_tolerances[2] + np.pi) % (2 * np.pi) - np.pi
        update = False

        if (trans_err < target_position_with_tolerances[3]/2) & (abs(rot_err) < target_position_with_tolerances[4]/2):
            self.idx += 1
            # check recursively
            return self.update_target_position_with_orientation(pos, orientation, target_position_with_tolerances)

        # check if object is closer to target than current position and update idx
        dist_obj_last = np.linalg.norm(self.trajectory_points[-1][0:2] - pos)
        dist_current_last = np.linalg.norm(self.trajectory_points[-1][0:2] - self.trajectory_points[self.idx][0:2])
        if dist_obj_last < dist_current_last:
            self.idx += 1
            # check recursively
            return self.update_target_position_with_orientation(pos, orientation, target_position_with_tolerances)

        return False

    def get_target_position(self):
        return self.trajectory_points[self.idx]
