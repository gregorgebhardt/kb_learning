from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from typing import Union


def plot_light_trajectory(states: Union[np.ndarray, pd.DataFrame], axes: Axes):
    if type(states) is pd.DataFrame:
        light_states = states['light']
        # light_states.unstack(0).plot(x='x', y='y', ax=axes, legend=False)
        if type(light_states.index) is pd.MultiIndex:
            light_x = light_states['x'].unstack(0).values
            light_y = light_states['y'].unstack(0).values
        else:
            light_x = light_states['x'].values
            light_y = light_states['y'].values
    else:
        light_x = states[..., -2]
        light_y = states[..., -1]

    axes.plot(light_x, light_y)


#  def plot_swarm_scatter(states: Union[np.ndarray, pd.DataFrame], axes: Axes = None):
#     if axes is None:
#         axes = Axes()
#
#     swarm
