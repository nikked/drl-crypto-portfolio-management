import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
from src.params import (
    EPSILON_GREEDY_THRESHOLD,
    LEARNING_RATE,
    KERNEL1_SIZE,
    MAX_PF_WEIGHT_PENALTY,
)


def plot_train_results(  # pylint: disable= too-many-arguments, too-many-locals
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
    only_cash = train_performance_lists["only_cash"]

    no_of_asset = len(asset_list)

    plt.title(
        "Portfolio Value (Test Set) {}: {}, {}, {}, {}, {}, {}, {}, {}".format(
            input_data_type,
            train_options["batch_size"],
            LEARNING_RATE,
            EPSILON_GREEDY_THRESHOLD,
            train_options["n_episodes"],
            train_options["window_length"],
            KERNEL1_SIZE,
            train_options["n_batches"],
            MAX_PF_WEIGHT_PENALTY,
        )
    )
    plt.plot(p_list, label="Agent Portfolio Value")
    plt.plot(p_list_eq, label="Equi-weighted Portfolio Value")
    plt.plot(p_list_s, label="Secured Portfolio Value")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.0)
    plt.show()

    names = ["Money"] + asset_list
    w_list = np.array(w_list)
    for j in range(no_of_asset + 1):
        if names[j] == "Money":
            continue

        plt.plot(w_list[:, j], label="Weight Stock {}".format(names[j]))
        plt.title("Weight evolution during testing")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.5)
    plt.show()
