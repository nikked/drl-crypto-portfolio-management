import time
import pytz
import json
import math
import os
from pprint import pprint
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from src.params import (
    EPSILON_GREEDY_THRESHOLD,
    TRADING_COST,
    LEARNING_RATE,
    KERNEL1_SIZE,
    N_FILTER_1,
    N_FILTER_2,
    MAX_PF_WEIGHT_PENALTY,
)

DATA_DIR = "crypto_data/"
GRAPH_OUTPUT_DIR = "train_graphs/"
JSON_OUTPUT_DIR = "train_jsons/"
CASH_NAME = "BTC"

if not os.path.exists(GRAPH_OUTPUT_DIR):
    os.mkdir(GRAPH_OUTPUT_DIR)

if not os.path.exists(JSON_OUTPUT_DIR):
    os.mkdir(JSON_OUTPUT_DIR)

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

    fig, axes = plt.subplots(nrows=7, ncols=1)

    # width, height
    fig.set_size_inches(16.6, 23.4)

    gs = axes[2].get_gridspec()
    axes[2].remove()
    axes[3].remove()
    axes[4].remove()
    axes[5].remove()

    price_ax = fig.add_subplot(gs[2:4])
    weight_ax = fig.add_subplot(gs[4:6])

    btc_price_data, btc_price_sharpe = _get_btc_price_data_for_period(
        train_configs, train_test_val_steps
    )

    dynamic_annualized_sharpe, static_annualized_sharpe, eq_annualized_sharpe, test_start, test_end = _plot_backtest_perf_metadata(
        axes[0],
        test_performance_lists,
        btc_price_sharpe,
        btc_price_data,
        train_configs,
        train_time_secs,
        train_test_val_steps,
    )
    _plot_portfolio_value_progress_test(axes[1], test_performance_lists, btc_price_data)
    _plot_crypto_price_test(
        price_ax, test_performance_lists, btc_price_data, asset_list
    )
    _plot_weight_evolution(
        weight_ax, asset_list, test_performance_lists["w_list"], btc_price_data
    )
    _plot_btc_price(axes[6], btc_price_data)

    # Rotate x axis labels
    for axis in fig.axes:
        plt.sca(axis)
        plt.xticks(rotation=30)

    timestamp_now = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    output_fn = None

    session_name = train_configs["train_session_name"]

    if "test_mode" in train_configs:
        if train_configs["test_mode"]:
            output_fn = f"{session_name}"

    if not output_fn:
        output_fn = f"{session_name}_{timestamp_now}"

    output_path = os.path.join(GRAPH_OUTPUT_DIR, f"train_results_{output_fn}.png")
    plt.subplots_adjust(hspace=0.5)
    print(f"Saving plot to path: {output_path}")
    plt.savefig(output_path, bbox_inches="tight")

    if train_configs["plot_results"]:
        plt.show()

    json_path = os.path.join(JSON_OUTPUT_DIR, f"train_history_{session_name}.json")
    print(f"\nWriting simulation data to filepath: {json_path}")

    start_time = time.time()

    try:
        with open(json_path, "r") as file:
            history_dict = json.load(file)

    except FileNotFoundError:
        history_dict = {}

    history_dict[timestamp_now] = {
        "dynamic": {
            "pf_value": test_performance_lists["p_list"][-1],
            "std_dev": test_performance_lists["std_devs"]["p_list"],
            "sharpe_ratio": test_performance_lists["sharpe_ratios"]["p_list"],
            "sharpe_ratio_ann": dynamic_annualized_sharpe,
            "mdd": test_performance_lists["max_drawdowns"]["p_list"],
        },
        "static": {
            "pf_value": test_performance_lists["p_list_static"][-1],
            "std_dev": test_performance_lists["std_devs"]["p_list_static"],
            "sharpe_ratio": test_performance_lists["sharpe_ratios"]["p_list_static"],
            "sharpe_ratio_ann": static_annualized_sharpe,
            "mdd": test_performance_lists["max_drawdowns"]["p_list_static"],
        },
        "eq_weight": {
            "pf_value": test_performance_lists["p_list_eq"][-1],
            "std_dev": test_performance_lists["std_devs"]["p_list_eq"],
            "sharpe_ratio": test_performance_lists["sharpe_ratios"]["p_list_eq"],
            "sharpe_ratio_ann": eq_annualized_sharpe,
            "mdd": test_performance_lists["max_drawdowns"]["p_list_eq"],
        },
        "initial_weights": list(test_performance_lists["w_list"][1]),
        "asset_list": asset_list,
        "test_start": test_start,
        "test_end": test_end,
        "trading_period_length": train_configs["trading_period_length"],
    }

    with open(json_path, "w") as file:
        json.dump(history_dict, file)

    end_time = time.time()
    train_time_secs = round(end_time - start_time, 4)

    print(f"JSON was open for {train_time_secs} seconds\n")


def _annualize_sharpe_ratio(days_between, sharpe_ratio):

    ratio_of_full_year = 365 / days_between

    annualized_sharpe = sharpe_ratio * math.sqrt(ratio_of_full_year)

    return round(annualized_sharpe, 4)


def _plot_backtest_perf_metadata(
    axis,
    test_performance_lists,
    btc_price_sharpe,
    btc_price_data,
    train_configs,
    train_time_secs,
    train_test_val_steps,
):

    # axis.set_ylim(-3, 2)
    columns = ("Strategy", "Ptf value", "Sharpe", "Sharpe (ann.)", "MDD")

    sharpe_ratios = test_performance_lists["sharpe_ratios"]
    max_drawdowns = test_performance_lists["max_drawdowns"]

    portfolio_final_value = round(test_performance_lists["p_list"][-1], 4)
    portfolio_static_final_value = round(test_performance_lists["p_list_static"][-1], 4)
    portfolio_eq_final_value = round(test_performance_lists["p_list_eq"][-1], 4)

    back_test_start_timestamp = btc_price_data.index[0]
    train_end_timestamp = back_test_start_timestamp - timedelta(days=1)

    back_test_day_count = (btc_price_data.index[-1] - back_test_start_timestamp).days

    train_start_dt = datetime.strptime(train_configs["start_date"], "%Y%m%d")
    train_start_dt = train_start_dt.replace(tzinfo=pytz.UTC)
    train_day_count = (train_end_timestamp.to_pydatetime() - train_start_dt).days

    dynamic_annualized_sharpe = _annualize_sharpe_ratio(
        back_test_day_count, sharpe_ratios["p_list"]
    )
    static_annualized_sharpe = _annualize_sharpe_ratio(
        back_test_day_count, sharpe_ratios["p_list_static"]
    )
    eq_annualized_sharpe = _annualize_sharpe_ratio(
        back_test_day_count, sharpe_ratios["p_list_eq"]
    )

    clust_data = [
        [
            "Dynamic agent",
            portfolio_final_value,
            round(sharpe_ratios["p_list"], 4),
            dynamic_annualized_sharpe,
            round(max_drawdowns["p_list"], 4),
        ],
        [
            "Static agent",
            portfolio_static_final_value,
            round(sharpe_ratios["p_list_static"], 4),
            static_annualized_sharpe,
            round(max_drawdowns["p_list_static"], 4),
        ],
        [
            "Equal weighted",
            portfolio_eq_final_value,
            round(sharpe_ratios["p_list_eq"], 4),
            eq_annualized_sharpe,
            round(max_drawdowns["p_list_eq"], 4),
        ],
    ]

    train_time_table_columns = ("Dataset", "Start date", "End date", "Days", "Steps")

    test_start = back_test_start_timestamp.strftime("%Y-%m-%d")
    test_end = datetime.strptime(train_configs["end_date"], "%Y%m%d").strftime(
        "%Y-%m-%d"
    )

    train_time_table_clust_data = [
        [
            "Train period",
            datetime.strptime(train_configs["start_date"], "%Y%m%d").strftime(
                "%Y-%m-%d"
            ),
            train_end_timestamp.strftime("%Y-%m-%d"),
            train_day_count,
            train_test_val_steps["train"],
        ],
        [
            "Test period",
            test_start,
            test_end,
            back_test_day_count,
            train_test_val_steps["test"],
        ],
    ]

    divider = make_axes_locatable(axis)

    axis.set_axis_off()

    axis.set_title(
        train_configs["train_session_name"].replace("_", " "),
        fontdict={"fontsize": 20, "position": (0.0, 0.85)},  # x, y
        horizontalalignment="left",
    )
    perf_table = axis.table(
        cellText=clust_data, colLabels=columns, loc="center", cellLoc="center"
    )

    perf_table.auto_set_font_size(False)
    perf_table.set_fontsize(10)
    perf_table.scale(1.0, 2)
    _format_table(perf_table)

    axis1 = divider.append_axes("right", size="57%", pad=0.8, sharex=axis)
    axis1.set_axis_off()
    date_table = axis1.table(
        cellText=train_time_table_clust_data,
        colLabels=train_time_table_columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.3, 0.3, 0.3, 0.2, 0.2],
    )

    date_table.auto_set_font_size(False)
    date_table.set_fontsize(10)
    date_table.scale(1.0, 2)
    _format_table(date_table)

    axis2 = divider.append_axes("right", size="40%", pad=0.35, sharex=axis)

    #     train_params_str = f"""
    # No. batches: {train_configs['n_batches']}
    # No. episodes: {train_configs['n_episodes']}
    # Batch size: {train_configs['batch_size']}

    # Trading period: {train_configs['trading_period_length']}
    # Train window length: {train_configs['window_length']}
    # """
    """
    No. conv filters, layer 1: {N_FILTER_1}
    No. conv filters, layer 2: {N_FILTER_2}
    Kernel size: : {KERNEL1_SIZE}
    """
    axis2.set_axis_off()

    trading_period = train_configs["trading_period_length"]

    # import pdb; pdb.set_trace()

    config_table_data = [
        ["No. batches", train_configs["n_batches"]],
        ["No. episodes", train_configs["n_episodes"]],
        ["Batch size", train_configs["batch_size"]],
        ["Trading fee", f"{TRADING_COST*100}%"],
        ["Trading period", trading_period],
        ["Window length", train_configs["window_length"]],
        ["Conv Filters, 1", N_FILTER_1],
        ["Conv Filters, 2", N_FILTER_2],
        ["Kernel size", KERNEL1_SIZE],
        ["ε greedy thld", EPSILON_GREEDY_THRESHOLD],
    ]

    config_table_columns = ("Parameter", "Value")

    config_table = axis2.table(
        cellText=config_table_data,
        colLabels=config_table_columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.5, 0.3],
    )

    config_table.auto_set_font_size(False)
    config_table.set_fontsize(8)
    config_table.scale(0.8, 1.6)

    _format_table(config_table)

    return (
        dynamic_annualized_sharpe,
        static_annualized_sharpe,
        eq_annualized_sharpe,
        test_start,
        test_end,
    )


def _format_table(table):
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(
                fontproperties=FontProperties(weight="bold"), color="white"
            )
            cell.set_facecolor("#4C72B0")


def _plot_portfolio_value_progress_test(axis, test_performance_lists, btc_price_data):

    p_list = test_performance_lists["p_list"]
    p_list_eq = test_performance_lists["p_list_eq"]
    p_list_static = test_performance_lists["p_list_static"]

    p_list_series = pd.Series(p_list, index=btc_price_data.index)
    p_list_eq_series = pd.Series(p_list_eq, index=btc_price_data.index)
    p_list_static_series = pd.Series(p_list_static, index=btc_price_data.index)

    axis.set_title("Portfolio Value")

    axis.plot(p_list_series, label="Dynamic agent")
    axis.plot(p_list_static_series, label="Static agent")
    axis.plot(p_list_eq_series, label="Equal weighted")

    axis.set_ylabel("Price performance vs BTC")

    axis.legend()
    _date_gridify_axis(axis)


def _plot_crypto_price_test(axis, test_performance_lists, btc_price_data, asset_list):

    axis.set_title("Cryptocurrency price evolution")

    p_list_fu = test_performance_lists["p_list_fu"]

    linestyles = ["-", "-", "-", ":", "-", "-", "-", "-", "-", "-", ":"]
    for i in range(len(asset_list)):
        crypto_price_series = pd.Series(p_list_fu[i], index=btc_price_data.index)
        axis.plot(
            crypto_price_series,
            linestyle=linestyles[i % len(linestyles)],
            label="{}".format(asset_list[i]),
        )

    axis.set_yscale("log")

    axis.get_yaxis().get_major_formatter().labelOnlyBase = False
    axis.legend(loc="upper left")
    axis.set_ylabel("Relative price (logarithmic)")
    _date_gridify_axis(axis)


def _plot_btc_price(axis, btc_price_data):

    axis.set_title("BTC/USD price")

    axis.plot(btc_price_data, label="BTC/USD")

    axis.get_yaxis().get_major_formatter().labelOnlyBase = False
    axis.legend()
    axis.set_ylabel("USD")
    _date_gridify_axis(axis)


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
        train_test_val_steps["train"] + train_test_val_steps["validation"] :
    ]

    btc_price_sharpe = (btc_data_test_period[-1] - btc_data_test_period[0]) / np.std(
        btc_data_test_period
    )

    return btc_data_test_period, btc_price_sharpe

    """

    OLD CODE THAT RETURNS BTC RELATIVE PRICE DIFF


    Calculate the final output series by first setting its value to initial
    portfolio value and then multiplying the prev value with the BTC price diff
    of the period
    """
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

    markers = ["2", ".", "x", "+", "1"]

    cash_idx = None
    for j in range(len(asset_list) + 1):
        if names[j] == CASH_NAME:
            cash_idx = j
            continue

        w_list_series = pd.Series(w_list[1:, j], btc_price_data.index[1:])

        axis.plot(
            w_list_series,
            label="{} weight".format(names[j]),
            alpha=0.95,
            linestyle="",
            marker=markers[j % len(markers)],
        )

    cash_list_series = pd.Series(w_list[1:, cash_idx], btc_price_data.index[1:])

    axis.plot(
        cash_list_series,
        label="{} weight".format(names[cash_idx]),
        alpha=0.95,
        linestyle="-",
    )

    axis.set_title("Portfolio weight evolution")
    axis.legend()
    axis.set_ylabel("Asset weight (%)")

    _date_gridify_axis(axis)


def _date_gridify_axis(axis):
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axis.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 8, 16]))

    # Or if you want different settings for the grids:
    axis.grid(which="minor", alpha=0.3)

    # Or if you want different settings for the grids:
    axis.grid(which="major", alpha=0.9)


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
