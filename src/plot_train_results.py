import os
import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
from src.params import (
    EPSILON_GREEDY_THRESHOLD,
    LEARNING_RATE,
    KERNEL1_SIZE,
    MAX_PF_WEIGHT_PENALTY,
)

OUTPUT_DIR = "train_graphs/"
CASH_NAME = "BTC"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def plot_train_results(  # pylint: disable= too-many-arguments, too-many-locals
    train_configs, test_performance_lists, train_performance_lists, asset_list
):

    _plot_portfolio_value_progress(test_performance_lists, asset_list, train_configs)

    _plot_weight_evolution(asset_list, test_performance_lists["w_list"], train_configs)


def _plot_portfolio_value_progress(test_performance_lists, asset_list, train_configs):

    p_list = test_performance_lists["p_list"]
    p_list_eq = test_performance_lists["p_list_eq"]
    p_list_s = test_performance_lists["p_list_s"]

    plt.title(
        "Portfolio Value (Test Set): {}, {}, {}, {}, {}, {}, {}, {}".format(
            train_configs["batch_size"],
            LEARNING_RATE,
            EPSILON_GREEDY_THRESHOLD,
            train_configs["n_episodes"],
            train_configs["window_length"],
            KERNEL1_SIZE,
            train_configs["n_batches"],
            MAX_PF_WEIGHT_PENALTY,
        )
    )
    plt.plot(p_list, label="Agent Portfolio Value")
    plt.plot(p_list_eq, label="Equi-weighted Portfolio Value")
    plt.plot(p_list_s, label="Secured Portfolio Value")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.0)

    output_path = os.path.join(OUTPUT_DIR, "portfolio_value_test_set.png")
    plt.savefig(output_path)

    if train_configs["plot_results"]:
        plt.show()

    plt.clf()


def _plot_weight_evolution(asset_list, w_list, train_configs):

    names = [CASH_NAME] + asset_list
    w_list = np.array(w_list)
    for j in range(len(asset_list) + 1):
        if names[j] == CASH_NAME:
            continue

        plt.plot(w_list[1:, j], label="Weight Stock {}".format(names[j]))
        plt.title("Weight evolution during testing")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.5)

    output_path = os.path.join(OUTPUT_DIR, "weight_evolution.png")
    plt.savefig(output_path)

    if train_configs["plot_results"]:
        plt.show()
