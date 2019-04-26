import os
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable


JSON_OUTPUT_DIR = "train_jsons/"

HISTOGRAM_OUTPUT_DIR = "train_histograms/"

if not os.path.exists(HISTOGRAM_OUTPUT_DIR):
    os.mkdir(HISTOGRAM_OUTPUT_DIR)


def make_train_histograms(session_name):
    json_path = os.path.join(JSON_OUTPUT_DIR, f"train_history_{session_name}.json")

    with open(json_path, "r") as file:
        history_dict = json.load(file)

    filtered_history = _filter_history_dict(history_dict)

    dynamic_pf_values = []
    dynamic_mdds = []
    dynamic_sharpe_ratios = []

    static_pf_values = []
    static_mdds = []
    static_sharpe_ratios = []

    cash_investments = []
    crypto_weight_std_devs = []

    n_simulations = len(filtered_history)

    first_key = list(filtered_history.keys())[0]

    asset_list = filtered_history[first_key]["asset_list"]
    eq_pf_value = filtered_history[first_key]["eq_weight"]["pf_value"]
    eq_sharpe_ratio = filtered_history[first_key]["eq_weight"]["sharpe_ratio"]
    eq_mdd = filtered_history[first_key]["eq_weight"]["mdd"]

    for timestamp, session_stats in filtered_history.items():

        dynamic = session_stats["dynamic"]
        static = session_stats["static"]

        initial_weights = session_stats["initial_weights"]

        dynamic_pf_values.append(dynamic["pf_value"])
        dynamic_mdds.append(dynamic["mdd"])
        dynamic_sharpe_ratios.append(dynamic["sharpe_ratio"])

        static_pf_values.append(static["pf_value"])
        static_mdds.append(static["mdd"])
        static_sharpe_ratios.append(static["sharpe_ratio"])

        cash_investments.append(initial_weights[0])

        crypto_weights = initial_weights[1:]
        crypto_weight_std_devs.append(np.std(crypto_weights))

    fig, axes = plt.subplots(nrows=5, ncols=2)

    # width, height
    fig.set_size_inches(16.6, 23.4)

    gs = axes[0][0].get_gridspec()
    axes[0][0].remove()
    axes[0][1].remove()

    table_ax = fig.add_subplot(gs[0:2])

    _plot_histogram_metadata_table(
        table_ax,
        n_simulations,
        session_name,
        asset_list,
        dynamic_pf_values,
        dynamic_mdds,
        dynamic_sharpe_ratios,
        static_pf_values,
        static_mdds,
        static_sharpe_ratios,
        cash_investments,
        crypto_weight_std_devs,
        eq_pf_value,
        eq_sharpe_ratio,
        eq_mdd,
    )

    _plot_histogram(
        axes[1][0],
        dynamic_pf_values,
        "Dynamic agent: Distribution of Portfolio Values",
        "Portfolio value",
    )
    _plot_histogram(
        axes[1][1],
        static_pf_values,
        "Static agent: Distribution of Portfolio Values",
        "Portfolio value",
    )

    _plot_histogram(
        axes[2][0],
        dynamic_sharpe_ratios,
        "Dynamic agent: Distribution of Sharpe Ratios",
        "Sharpe ratio",
    )
    _plot_histogram(
        axes[2][1],
        static_sharpe_ratios,
        "Static agent: Distribution of Sharpe Ratios",
        "Sharpe ratio",
    )

    _plot_histogram(
        axes[3][0],
        dynamic_mdds,
        "Dynamic agent: Distribution of Maximum Drawdowns",
        "Maximum drawdown",
    )
    _plot_histogram(
        axes[3][1],
        static_mdds,
        "Static agent: Distribution of Maximum Drawdowns",
        "Maximum drawdown",
    )

    _plot_histogram(
        axes[4][0], crypto_weight_std_devs, "Both agents: Distribution of the standard deviation of weights", "Stdev of weights"
    )
    _plot_histogram(axes[4][1], cash_investments,
                    "Both agents: Distribution of BTC weights", "BTC weight")

    output_path = os.path.join(HISTOGRAM_OUTPUT_DIR, f"histogram_{session_name}.png")
    plt.subplots_adjust(hspace=0.5)
    print(f"Saving plot to path: {output_path}")
    plt.savefig(output_path, bbox_inches="tight")


def _filter_history_dict(history_dict):

    filtered_history = {}
    for timestamp, train_data in history_dict.items():

        initial_weights = train_data['initial_weights']

        # Ignore train runs with negative weight
        if any(value < 0 for value in initial_weights):
            continue

        # Ignore train runs with huge weight
        if any(value > 0.7 for value in initial_weights):
            continue

        filtered_history[timestamp] = train_data

    return filtered_history


def _plot_histogram_metadata_table(
    axis,
    n_simulations,
    session_name,
    asset_list,
    dynamic_pf_values,
    dynamic_mdds,
    dynamic_sharpe_ratios,
    static_pf_values,
    static_mdds,
    static_sharpe_ratios,
    cash_investments,
    crypto_weight_std_devs,
    eq_pf_value,
    eq_sharpe_ratio,
    eq_mdd,
):

    divider = make_axes_locatable(axis)

    # add sample size of simulation

    axis.set_axis_off()

    axis.set_title(
        f"Simulation statistics: {session_name.replace('_', ' ')} (n={n_simulations})",
        fontdict={"fontsize": 20, "position": (0.0, 0.85)},  # x, y
        horizontalalignment="left",
    )

    dynamic_columns = ("Dynamic agent", "Average", "Stdev")
    dynamic_data = [
        [
            "Ptf. value",
            round(np.mean(dynamic_pf_values), 4),
            round(np.std(dynamic_pf_values), 4),
        ],
        [
            "Sharpe ratio",
            round(np.mean(dynamic_sharpe_ratios), 4),
            round(np.std(dynamic_sharpe_ratios), 4),
        ],
        ["MDD", round(np.mean(dynamic_mdds), 4),
         round(np.std(dynamic_mdds), 4)],
        [
            "Stdev of weights",
            round(np.mean(crypto_weight_std_devs), 4),
            round(np.std(crypto_weight_std_devs), 4),
        ],
        [
            "BTC weight",
            round(np.mean(cash_investments), 4),
            round(np.std(cash_investments), 4),
        ],
    ]

    dynamic_table = axis.table(
        cellText=dynamic_data, colLabels=dynamic_columns, loc="center", cellLoc="center"
    )
    _format_table(dynamic_table)

    static_columns = ("Static agent", "Average", "Stdev")
    static_data = [
        [
            "Ptf. value",
            round(np.mean(static_pf_values), 4),
            round(np.std(static_pf_values), 4),
        ],
        [
            "Sharpe ratio",
            round(np.mean(static_sharpe_ratios), 4),
            round(np.std(static_sharpe_ratios), 4),
        ],
        ["MDD", round(np.mean(static_mdds), 4), round(np.std(static_mdds), 4)],
        [
            "Stdev of weights",
            round(np.mean(crypto_weight_std_devs), 4),
            round(np.std(crypto_weight_std_devs), 4),
        ],
        [
            "BTC weight",
            round(np.mean(cash_investments), 4),
            round(np.std(cash_investments), 4),
        ],
    ]

    axis1 = divider.append_axes("right", size="100%", pad=0.3, sharex=axis)
    axis1.set_axis_off()

    static_table = axis1.table(
        cellText=static_data, colLabels=static_columns, loc="center", cellLoc="center"
    )
    _format_table(static_table)

    axis2 = divider.append_axes("right", size="66%", pad=0.3, sharex=axis)
    axis2.set_axis_off()

    eq_columns = ("Equal weighted", "Average")
    eq_data = [
        ["Ptf. value", round(eq_pf_value, 4)],
        ["Sharpe ratio", round(eq_sharpe_ratio, 4)],
        ["MDD", round(eq_mdd, 4)],
    ]

    eq_table = axis2.table(
        cellText=eq_data, colLabels=eq_columns, loc="center", cellLoc="center"
    )
    _format_table(eq_table)

    axis3 = divider.append_axes("right", size="30%", pad=0.3, sharex=axis)
    axis3.set_axis_off()

    crypto_columns = ("Assets",)
    crypto_table_data = [[asset] for asset in asset_list]

    crypto_table = axis3.table(
        cellText=crypto_table_data,
        colLabels=crypto_columns,
        loc="center",
        cellLoc="center",
    )
    _format_table(crypto_table)


def _format_table(table):
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(
                fontproperties=FontProperties(weight="bold"), color="white"
            )
            cell.set_facecolor("#4C72B0")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2)


def _plot_histogram(axis, data, title, xlabel):
    num_bins = 15

    sns.distplot(data, bins=num_bins, ax=axis, rug=True, kde=False)
    # axis.hist(data, num_bins, alpha=0.95)
    axis.grid(alpha=0.3)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.set_title(title)


if __name__ == "__main__":

    # session_name = "test_run_with_long_name"
    session_name = "Jiang_et_al._backtest__#1"
    session_name2 = "Jiang_et_al._backtest__#2"
    session_name3 = "Jiang_et_al._backtest__#3"

    make_train_histograms(session_name)
    make_train_histograms(session_name2)
    make_train_histograms(session_name3)
