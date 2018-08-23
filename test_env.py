import numpy as np

import yaml

from kb_learning.envs import EvalEnv

center = np.array([-.5, .5])
angles = np.linspace(1.5 * np.pi, 2 * np.pi, 10)
waypts = np.hstack((np.array([np.cos(angles), np.sin(angles)]).T + center,
                    (angles - 1.5 * np.pi).reshape(-1, 1)))


class EvalEnvConfiguration(yaml.YAMLObject):
    yaml_tag = '!EvalEnv'

    class ObjectConfiguration(yaml.YAMLObject):
        yaml_tag = '!ObjectConf'

        def __init__(self, shape, width, height, init):
            self.shape = shape
            self.width = width
            self.height = height
            self.init = init

    class LightConfiguration(yaml.YAMLObject):
        yaml_tag = '!LightConf'

        def __init__(self, type, init, radius=None):
            self.type = type
            self.init = init
            self.radius = radius

    class KilobotsConfiguration(yaml.YAMLObject):
        yaml_tag = '!KilobotsConf'

        def __init__(self, num, mean, std):
            self.num = num
            self.mean = mean
            self.std = std

    def __init__(self, width, height, resolution, objects, light, kilobots):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.objects = [self.ObjectConfiguration(**obj) for obj in objects]
        self.light = self.LightConfiguration(**light)
        self.kilobots = self.KilobotsConfiguration(**kilobots)


configuration = '''
!EvalEnv
width: 1.5
height: 1.5
resolution: 500

objects:
  - !ObjectConf
    shape: quad
    width: .15
    height: .15
    init: [-0.5, -0.5, .0]

light: !LightConf
  type: circular
  radius: .3
  init: [.0, .0]
  
kilobots: !KilobotsConf
  num: 15
  mean: [.0, .0]
  std: .1
'''

conf = yaml.load(configuration)

env = EvalEnv(conf)

for i in range(500):
    env.render()
    env.step(env.action_space.sample())

# sampler = SARSSampler(register_object_env, num_episodes=1, num_steps_per_episode=1, weight=weight,
#                       num_kilobots=num_kilobots, object_shape=object_shape,
#                       object_width=object_width, object_height=object_height,
#                       light_type=light_type, light_radius=light_radius)
# kernel = KilobotEnvKernel(30)
# policy = SparseWeightedGP(kernel, output_dim=1)
#
# sars = sampler(policy, 1, 1)