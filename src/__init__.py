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
    train_options,
    test_performance_lists,
    train_performance_lists,
    input_data_type,
    asset_list,
):

    p_list = test_performance_lists["p_list"]
    p_list_eq = test_performance_lists["p_list_eq"]
    p_list_s = test_performance_lists["p_list_s"]
    p_list_fu = test_performance_lists["p_list_fu"]
    w_list = test_performance_lists["w_list"]

    policy_network = train_performance_lists["policy_network"]
    equal_weighted = train_performance_lists["equal_weighted"]
    single_asset = train_performance_lists["single_asset"]

    no_of_asset = len(asset_list)

    plt.title(
        "Portfolio Value (Test Set) {}: {}, {}, {}, {}, {}, {}, {}, {}".format(
            input_data_type,
            BATCH_SIZE,
            LEARNING_RATE,
            RATIO_GREEDY,
            N_EPISODES,
            train_options["window_length"],
            KERNEL1_SIZE,
            train_options["n_batches"],
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

    plt.plot(policy_network[index1:index2], label="Agent Portfolio Value")
    plt.plot(equal_weighted[index1:index2], label="Baseline Portfolio Value")
    plt.plot(single_asset[index1:index2], label="Secured Portfolio Value")
    plt.legend()
    plt.show()

    plt.plot((np.array(policy_network) - np.array(equal_weighted)))
