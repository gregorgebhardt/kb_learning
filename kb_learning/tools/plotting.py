from matplotlib import Axes
import pandas as pd
import numpy as np
from typing import Union

def plot_light_trajectory(states: Union[np.ndarray, pd.DataFrame], axes: Axes=None):
    if axes is None:
        axes = Axes()

    if type(states) is pd.DataFrame:
        states[-2:].unstack(0)

    light_trajectory = states[:, :-2]

    axes.plot(light_trajectory[:, 0], light_trajectory[:, 1], '-y')

def plot_swarm_scatter(states: Union[np.ndarray, pd.DataFrame], axes: Axes = None):
    if axes is None:
        axes = Axes()

    swarm