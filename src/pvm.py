import numpy as np


SAMPLE_BIAS = (
    5e-5
)  # Beta in the geometric distribution for online training sample batches


class PVM:
    """
    This is the memory stack called PVM in the paper
    """

    def __init__(self, total_steps, batch_size, w_init):

        # initialization of the memory
        # we have a total_step_times the initialization portfolio tensor
        self.memory = np.transpose(np.array([w_init] * total_steps))
        self.sample_bias = SAMPLE_BIAS
        self.total_steps = total_steps
        self.batch_size = batch_size

    def get_w(self, index):
        # return the weight from the PVM at time t
        return self.memory[:, index]

    def update(self, index, w_t):
        # update the weight at time t
        self.memory[:, index] = w_t

    def draw(self, beta=SAMPLE_BIAS):
        """
        returns a valid step so you can get a training batch starting at this step
        """
        while 1:
            zeta = np.random.geometric(p=beta)
            i_start = self.total_steps - self.batch_size + 1 - zeta
            if i_start >= 0:
                return i_start
