import matplotlib

matplotlib.use("TkAgg")

import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
from src.params import (
    RATIO_GREEDY,
    BATCH_SIZE,
    LEARNING_RATE,
    N_EPISODES,
    KERNEL1_SIZE,
    RATIO_REGUL,
)


def plot_training_results(  # pylint: disable= too-many-arguments, too-many-locals
    window_length,
    n_batches,
    p_list,
    p_list_eq,
    p_list_s,
    p_list_fu,
    w_list,
    list_final_pf,
    list_final_pf_eq,
    list_final_pf_s,
    input_data_type,
    asset_list,
):

    no_of_asset = len(asset_list)

    plt.title(
        "Portfolio Value (Test Set) {}: {}, {}, {}, {}, {}, {}, {}, {}".format(
            input_data_type,
            BATCH_SIZE,
            LEARNING_RATE,
            RATIO_GREEDY,
            N_EPISODES,
            window_length,
            KERNEL1_SIZE,
            n_batches,
            RATIO_REGUL,
        )
    )
    plt.plot(p_list, label="Agent Portfolio Value")
    plt.plot(p_list_eq, label="Equi-weighted Portfolio Value")
    plt.plot(p_list_s, label="Secured Portfolio Value")
    for i in range(no_of_asset):
        plt.plot(
            p_list_fu[i], label="Full Stock {} Portfolio Value".format(asset_list[i])
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.0)
    plt.show()

    names = ["Money"] + asset_list
    w_list = np.array(w_list)
    for j in range(no_of_asset + 1):
        plt.plot(w_list[:, j], label="Weight Stock {}".format(names[j]))
        plt.title("Weight evolution during testing")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
    plt.show()

    plt.plot(np.array(p_list) - np.array(p_list_eq))

    index1 = 0
    index2 = -1

    plt.plot(list_final_pf[index1:index2], label="Agent Portfolio Value")
    plt.plot(list_final_pf_eq[index1:index2], label="Baseline Portfolio Value")
    plt.plot(list_final_pf_s[index1:index2], label="Secured Portfolio Value")
    plt.legend()
    plt.show()

    plt.plot((np.array(list_final_pf) - np.array(list_final_pf_eq)))
