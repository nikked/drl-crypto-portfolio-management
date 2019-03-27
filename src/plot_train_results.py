import os
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from src.params import (
    EPSILON_GREEDY_THRESHOLD,
    LEARNING_RATE,
    KERNEL1_SIZE,
    MAX_PF_WEIGHT_PENALTY,
)

OUTPUT_DIR = "train_graphs/"
CASH_NAME = "BTC"

"""
TODO
plot BTC price change during period
"""

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def plot_train_results(  # pylint: disable= too-many-arguments, too-many-locals
    train_configs,
    test_performance_lists,
    train_performance_lists,
    asset_list,
    train_time_secs,
):

    timestamp_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    if "test_mode" in train_configs:
        timestamp_now = "test"

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12.5, 9.5)

    _plot_portfolio_value_progress(
        axes[0][0], test_performance_lists, asset_list, train_configs
    )
    _plot_weight_evolution(
        axes[0][1], asset_list, test_performance_lists["w_list"], train_configs
    )
    _plot_train_params(axes[1][1], train_configs, train_time_secs, timestamp_now)

    output_path = os.path.join(OUTPUT_DIR, f"train_results_{timestamp_now}.png")
    plt.savefig(output_path)

    if train_configs["plot_results"]:
        plt.show()


def _plot_train_params(axis, train_configs, train_time_secs, timestamp_now):

    train_params_str = f"""
Start date:  {train_configs['start_date']}
End date:  {train_configs['end_date']}
Batch size: {train_configs['batch_size']}
Trading period: {train_configs['trading_period_length']}
Train window length: {train_configs['window_length']}
Training duration: {train_time_secs} seconds
Training timestamp: {timestamp_now}
    """

    axis.set_axis_off()
    axis.text(
        x=0.0,
        y=0.5,
        s=train_params_str,
        # ha='center',
        # va='center',
        size=10,
    )


def _plot_portfolio_value_progress(
    axis, test_performance_lists, asset_list, train_configs
):

    p_list = test_performance_lists["p_list"]
    p_list_eq = test_performance_lists["p_list_eq"]

    axis.set_title(
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
    axis.plot(p_list, label="Agent")
    axis.plot(p_list_eq, label="Equally weighted")

    axis.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.0)


def _plot_weight_evolution(axis, asset_list, w_list, train_configs):

    names = [CASH_NAME] + asset_list
    w_list = np.array(w_list)
    for j in range(len(asset_list) + 1):
        if names[j] == CASH_NAME:
            continue

        axis.plot(w_list[1:, j], label="{}".format(names[j]))
        axis.set_title("Weight evolution during testing")
        axis.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.5)
