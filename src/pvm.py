import numpy as np

class PVM:

    def __init__(self, total_steps, w_init):
        self.memory = np.transpose(np.array([w_init] * total_steps))

    def get_weight(self, index):
        return self.memory[:, index]

    def update_weight(self, index, w_t):
        self.memory[:, index] = w_t
