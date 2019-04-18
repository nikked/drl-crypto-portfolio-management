import os
from pprint import pprint
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from src.params import (
    EPSILON_GREEDY_THRESHOLD,
    LEARNING_RATE,
    KERNEL1_SIZE,
    MAX_PF_WEIGHT_PENALTY,
)

DATA_DIR = "crypto_data/"
OUTPUT_DIR = "train_graphs/"
CASH_NAME = "BTC"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# move crypto price above weights
# test and train times


def plot_train_results(  # pylint: disable= too-many-arguments, too-many-locals
    train_configs,
    train_performance_lists,
    test_performance_lists,
    asset_list,
    train_time_secs,
    train_test_val_steps,
):

    pprint("Making a plot of results")
    print("\nTrain configs")
    pprint(train_configs)
    print("\nTrain test val steps")
    pprint(train_test_val_steps)

    timestamp_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fig, axes = plt.subplots(nrows=6, ncols=1)

    # width, height
    fig.set_size_inches(16.6, 23.4)

    gs = axes[4].get_gridspec()
    axes[4].remove()
    axes[5].remove()

    weight_ax = fig.add_subplot(gs[4:6])

    btc_price_data, btc_price_sharpe = _get_btc_price_data_for_period(
        train_configs, train_test_val_steps
    )

    _plot_backtest_perf_metadata(
        axes[0], test_performance_lists, btc_price_sharpe, btc_price_data, train_configs, timestamp_now, train_time_secs
    )
    _plot_portfolio_value_progress_test(
        axes[1], test_performance_lists, btc_price_data
    )
    _plot_crypto_price_test(
        axes[2], test_performance_lists, btc_price_data, asset_list
    )
    _plot_btc_price(
        axes[3], btc_price_data
    )
    _plot_weight_evolution(
        weight_ax, asset_list, test_performance_lists["w_list"], btc_price_data
    )

    # Rotate x axis labels
    for axis in fig.axes:
        plt.sca(axis)
        plt.xticks(rotation=30)

    output_fn = timestamp_now

    if "train_session_name" in train_configs:
        output_fn = f"{train_configs['train_session_name']}"

    elif "test_mode" in train_configs:
        if train_configs["test_mode"]:
            output_fn = "test"

    output_path = os.path.join(OUTPUT_DIR, f"train_results_{output_fn}.png")
    plt.subplots_adjust(hspace=0.5)
    print(f"Saving plot to path: {output_path}")
    plt.savefig(output_path, bbox_inches="tight")

    if train_configs["plot_results"]:
        plt.show()

    pprint("Exiting")


def _plot_backtest_perf_metadata(axis, test_performance_lists, btc_price_sharpe, btc_price_data, train_configs, timestamp_now, train_time_secs):


    axis.set_ylim(-3, 2)
    columns = ("Strategy", "Ptf value", "Sharpe", "MDD")

    sharpe_ratios = test_performance_lists["sharpe_ratios"]
    max_drawdowns = test_performance_lists["max_drawdowns"]

    portfolio_final_value = round(test_performance_lists["p_list"][-1], 2)
    portfolio_eq_final_value = round(
        test_performance_lists["p_list_eq"][-1], 2)
    btc_long_final_value = round(btc_price_data[-1], 2)

    back_test_start_timestamp = btc_price_data.index[0]
    train_end_timestamp = back_test_start_timestamp - timedelta(days=1)

    clust_data = [
        [
            "DRL",
            portfolio_final_value,
            round(sharpe_ratios["p_list"], 3),
            round(max_drawdowns["p_list"], 3),
        ],
        [
            "Eq weight",
            portfolio_eq_final_value,
            round(sharpe_ratios["p_list_eq"], 3),
            round(max_drawdowns["p_list_eq"], 3),
        ],
        # ["Long Bitcoin",
        #  None,
        #  btc_long_final_value,
        #  round(btc_price_sharpe, 3)],
    ]

    train_time_table_columns = ("Dataset", "Start date", "End date")

    train_time_table_clust_data = [
        [
            "Train period",
            datetime.strptime(
                train_configs['start_date'], '%Y%m%d').strftime('%Y-%m-%d'),
            train_end_timestamp.strftime('%Y-%m-%d'),
        ],
        [
            "Test period",
            back_test_start_timestamp.strftime('%Y-%m-%d'),
            datetime.strptime(
                train_configs['end_date'], '%Y%m%d').strftime('%Y-%m-%d'),
        ],
    ]

    divider = make_axes_locatable(axis)

    axis.set_axis_off()
    perf_table = axis.table(
        cellText=clust_data,
        colLabels=columns,
        loc="center",
    )

    perf_table.auto_set_font_size(False)
    perf_table.set_fontsize(10)
    perf_table.scale(1.0, 2)

    axis1 = divider.append_axes("right", size="80%", pad=0.2, sharex=axis)
    axis1.set_axis_off()
    date_table = axis1.table(
        cellText=train_time_table_clust_data,
        colLabels=train_time_table_columns,
        loc="center",
    )

    date_table.auto_set_font_size(False)
    date_table.set_fontsize(10)
    date_table.scale(1.0, 2)
    axis2 = divider.append_axes("right", size="80%", pad=0.2, sharex=axis)

    train_params_str = f"""
No. batches: {train_configs['n_batches']}
No. episodes: {train_configs['n_episodes']}
Batch size: {train_configs['batch_size']}

Trading period: {train_configs['trading_period_length']}
Train window length: {train_configs['window_length']}
    """
    axis2.set_axis_off()
    axis2.text(
        x=0.1,
        y=0.2,
        s=train_params_str,
        # ha='center',
        size=10,
    )


def _plot_portfolio_value_progress_test(axis, test_performance_lists, btc_price_data):

    p_list = test_performance_lists["p_list"]
    p_list_eq = test_performance_lists["p_list_eq"]

    p_list_series = pd.Series(p_list, index=btc_price_data.index)
    p_list_eq_series = pd.Series(p_list_eq, index=btc_price_data.index)

    axis.set_title("Portfolio Value (Test Set)")

    axis.plot(p_list_series, label="Agent")
    axis.plot(p_list_eq_series, label="Equally weighted")

    axis.xaxis.set_major_locator(
        mdates.DayLocator(interval=7)
    )  # to get a tick every 15 minutes
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # optional formatting

    axis.legend()


def _plot_crypto_price_test(axis, test_performance_lists, btc_price_data, asset_list):

    axis.set_title("Cryptocurrency price evolution (BTC as cash)")


    p_list_fu = test_performance_lists["p_list_fu"]

    for i in range(len(asset_list)):
        crypto_price_series = pd.Series(
            p_list_fu[i], index=btc_price_data.index)
        axis.plot(crypto_price_series, label="{}".format(asset_list[i]))

    axis.xaxis.set_major_locator(
        mdates.DayLocator(interval=7)
    )  # to get a tick every 15 minutes
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # optional formatting

    axis.set_yscale('log')

    axis.get_yaxis().get_major_formatter().labelOnlyBase = False
    axis.legend()


def _plot_btc_price(axis, btc_price_data):

    axis.set_title("BTC/USD price")

    axis.plot(btc_price_data, label="BTC/USD")

    axis.xaxis.set_major_locator(
        mdates.DayLocator(interval=7)
    )  # to get a tick every 15 minutes
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # optional formatting

    axis.get_yaxis().get_major_formatter().labelOnlyBase = False
    axis.legend()


def _get_btc_price_data_for_period(train_configs, train_test_val_steps):
    """
    We want to compare our agent to the performance of BTC. For that we reason
    we calculate how an BTC investment of same magnitude would have performed
    during the test period.

    """
    t_confs = train_configs

    # Open BTC price data from file
    btc_price_fn = f"USDT_BTC_{t_confs['start_date']}-{t_confs['end_date']}_{t_confs['trading_period_length']}.csv"
    btc_price_fp = os.path.join(
        DATA_DIR,
        "USDT_BTC",
        f"{t_confs['start_date']}-{t_confs['end_date']}",
        btc_price_fn,
    )
    data = pd.read_csv(btc_price_fp).fillna("bfill").copy()

    # Set BTC price data index to real date
    data["datetime"] = pd.to_datetime(data["date"], unit="s", utc=True)
    data["datetime"] = data["datetime"] + pd.Timedelta("02:00:00")
    data = data.set_index("datetime")

    btc_data_test_period = data.close[
        train_test_val_steps["train"] + train_test_val_steps["validation"]:
    ]

    btc_price_sharpe = (btc_data_test_period[-1] - btc_data_test_period[0]) / np.std(
        btc_data_test_period
    )

    return btc_data_test_period, btc_price_sharpe

    # """
    # Calculate the final output series by first setting its value to initial
    # portfolio value and then multiplying the prev value with the BTC price diff
    # of the period
    # """
    # btc_price_data = pd.Series()
    # btc_price_data = btc_price_data.append(
    #     pd.Series(t_confs["portfolio_value"]))

    # btc_data_price_diffs = btc_data_test_period.pct_change()

    # for idx in range(1, len(btc_data_price_diffs)):
    #     prev_pf_value = btc_price_data[idx - 1]
    #     current_price_diff = btc_data_price_diffs[idx]
    #     changed_btc_pf_value = prev_pf_value * (current_price_diff + 1)
    #     btc_price_data.at[idx] = changed_btc_pf_value

    # btc_price_sharpe = (btc_price_data[idx] - btc_price_data[0]) / np.std(
    #     btc_price_data
    # )
    # btc_price_data.index = btc_data_test_period.index

    # return btc_price_data, btc_price_sharpe


def _plot_weight_evolution(axis, asset_list, w_list, btc_price_data):

    names = [CASH_NAME] + asset_list
    w_list = np.array(w_list)
    for j in range(len(asset_list) + 1):
        if names[j] == CASH_NAME:
            continue

        w_list_series = pd.Series(w_list[1:, j], btc_price_data.index[1:])

        axis.plot(w_list_series, label="{} weight".format(names[j]), alpha=0.5)

    axis.set_title("Portfolio weight evolution")
    axis.legend(
        # bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.5
    )

    axis.xaxis.set_major_locator(
        mdates.DayLocator(interval=7)
    )  # to get a tick every 15 minutes
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # optional formatting


"""
Training timestamp: {timestamp_now}
Training duration: {train_time_secs} seconds

No. batches: {train_configs['n_batches']}
No. episodes: {train_configs['n_episodes']}
Batch size: {train_configs['batch_size']}

Trading period: {train_configs['trading_period_length']}
Train window length: {train_configs['window_length']}

Kernel size: {KERNEL1_SIZE}

Epsilon greedy threshold: {EPSILON_GREEDY_THRESHOLD}
Learning rate: {LEARNING_RATE}
Max weight penalty: {MAX_PF_WEIGHT_PENALTY}
"""
