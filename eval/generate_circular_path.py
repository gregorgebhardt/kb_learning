import numpy as np

from kb_learning.planning import WayPointConf


def generate_circular_path(center: np.ndarray, radius=.8, angular_range=(-np.pi/2, .0), num_way_points=10):
    angles = np.linspace(*angular_range, num_way_points)
    arc_points = np.c_[np.cos(angles), np.sin(angles)] * radius + np.asarray(center)
    orientations = angles + np.pi/2
    way_points = [WayPointConf(None, x, y, theta, position_accuracy=.05, orientation_accuracy=.2)
                  for x, y, theta in np.c_[arc_points, orientations]]

    return way_points
