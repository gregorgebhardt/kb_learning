import numpy as np
from kb_learning.learner._learner import QuadPushingACRepsLearner


zeros = np.array([[.0, .0]])


def policy(s):
    return zeros


learner = QuadPushingACRepsLearner()
learner.policy = policy

learner._get_samples_parallel(100, 125, .0, 15)
