import numpy as np


def get_max_draw_down(xs):
    xs = np.array(xs)
    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
    j = np.argmax(xs[:i])  # start of period

    return xs[j] - xs[i]
