import matplotlib.pyplot as plt
import numpy as np
import math
import os

OUTPUT_DIR = "doc_graphs/"


def sigmoid(input_ndarray):
    output_ndarray = []
    for item in input_ndarray:
        output_ndarray.append(1 / (1 + math.exp(-item)))
    return output_ndarray


def relu(input_ndarray):
    output_ndarray = []
    for item in input_ndarray:
        output_ndarray.append(max(item, 0))
    return output_ndarray


def step(input_ndarray):
    output_ndarray = []
    for item in input_ndarray:
        if item >= 0:
            output_ndarray.append(1)
        else:
            output_ndarray.append(0)
    return output_ndarray


fig, ax = plt.subplots()
x = np.arange(-4, 4, 0.01)

ax.plot(x, sigmoid(x), "--", label="Logistic")
ax.plot(x, relu(x), ":", label="ReLU")
ax.plot(x, step(x), "", label="Step")

legend = ax.legend(loc="upper left", shadow=False, fontsize="large")

plt.ylim(-0.1, 1.1)
output_path = os.path.join(OUTPUT_DIR, "activation_funcs.png")
plt.savefig(output_path)
plt.show()
