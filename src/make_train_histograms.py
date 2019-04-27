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

    filtered_history = filter_history_dict(history_dict)
    n_simulations = len(filtered_history)

    if not n_simulations:
        print(f'\nWARNING: Could not create histogram for session {session_name}. No valid test runs found from total {len(history_dict)}\n')
        return

    backtest_stats = aggregate_backtest_stats(filtered_history)

    fig, axes = plt.subplots(nrows=5, ncols=2)

    # width, height
    fig.set_size_inches(16.6, 23.4)

    gs = axes[0][0].get_gridspec()
    axes[0][0].remove()
    axes[0][1].remove()

    table_ax = fig.add_subplot(gs[0:2])

    _plot_histogram_metadata_table(
        table_ax, n_simulations, session_name, backtest_stats
    )

    _plot_histogram(
        axes[1][0],
        backtest_stats["dynamic_pf_values"],
        "Dynamic agent: Distribution of Portfolio Values",
        "Portfolio value",
    )
    _plot_histogram(
        axes[1][1],
        backtest_stats["static_pf_values"],
        "Static agent: Distribution of Portfolio Values",
        "Portfolio value",
    )

    _plot_histogram(
        axes[2][0],
        backtest_stats["dynamic_sharpe_ratios"],
        "Dynamic agent: Distribution of Sharpe Ratios",
        "Sharpe ratio",
    )
    _plot_histogram(
        axes[2][1],
        backtest_stats["static_sharpe_ratios"],
        "Static agent: Distribution of Sharpe Ratios",
        "Sharpe ratio",
    )

    _plot_histogram(
        axes[3][0],
        backtest_stats["dynamic_mdds"],
        "Dynamic agent: Distribution of Maximum Drawdowns",
        "Maximum drawdown",
    )
    _plot_histogram(
        axes[3][1],
        backtest_stats["static_mdds"],
        "Static agent: Distribution of Maximum Drawdowns",
        "Maximum drawdown",
    )

    _plot_histogram(
        axes[4][0],
        backtest_stats["crypto_weight_averages"],
        "Both agents: Distribution of the average of weights",
        "Average weight",
    )
    _plot_histogram(
        axes[4][1],
        backtest_stats["crypto_weight_std_devs"],
        "Both agents: Distribution of the standard deviation of weights",
        "Stdev of weights",
    )
    # _plot_histogram(
    #     axes[5][0],
    #     backtest_stats["cash_investments"],
    #     "Both agents: Distribution of BTC weights",
    #     "BTC weight",
    # )

    output_path = os.path.join(HISTOGRAM_OUTPUT_DIR, f"histogram_{session_name}.png")
    plt.subplots_adjust(hspace=0.5)
    print(f"Saving plot to path: {output_path}")
    plt.savefig(output_path, bbox_inches="tight")


def aggregate_backtest_stats(filtered_history):
    dynamic_pf_values = []
    dynamic_mdds = []
    dynamic_sharpe_ratios = []
    dynamic_sharpe_ratios_ann = []

    static_pf_values = []
    static_mdds = []
    static_sharpe_ratios = []
    static_sharpe_ratios_ann = []

    cash_investments = []
    crypto_weight_averages = []
    crypto_weight_std_devs = []

    first_key = list(filtered_history.keys())[0]

    asset_list = filtered_history[first_key]["asset_list"]
    eq_pf_value = filtered_history[first_key]["eq_weight"]["pf_value"]
    eq_sharpe_ratio = filtered_history[first_key]["eq_weight"]["sharpe_ratio"]
    eq_mdd = filtered_history[first_key]["eq_weight"]["mdd"]
    eq_sharpe_ratio_ann = 0

    test_start = "NA"
    test_end = "NA"
    trading_period_length = "NA"

    for timestamp, session_stats in filtered_history.items():

        if "test_start" in session_stats:
            test_start = session_stats["test_start"]
        if "test_end" in session_stats:
            test_end = session_stats["test_end"]
        if "trading_period_length" in session_stats:
            trading_period_length = session_stats["trading_period_length"]
        if "sharpe_ratio_ann" in session_stats["eq_weight"]:
            eq_sharpe_ratio_ann = session_stats["eq_weight"]["sharpe_ratio_ann"]

        dynamic = session_stats["dynamic"]
        static = session_stats["static"]

        initial_weights = session_stats["initial_weights"]

        dynamic_pf_values.append(dynamic["pf_value"])
        dynamic_mdds.append(dynamic["mdd"])
        dynamic_sharpe_ratios.append(dynamic["sharpe_ratio"])

        try:
            dynamic_sharpe_ratios_ann.append(dynamic["sharpe_ratio_ann"])
        except KeyError:
            pass

        static_pf_values.append(static["pf_value"])
        static_mdds.append(static["mdd"])
        static_sharpe_ratios.append(static["sharpe_ratio"])

        try:
            static_sharpe_ratios_ann.append(static["sharpe_ratio_ann"])
        except KeyError:
            pass

        cash_investments.append(initial_weights[0])

        crypto_weights = initial_weights[1:]
        crypto_weight_averages.append(np.mean(crypto_weights))
        crypto_weight_std_devs.append(np.std(crypto_weights))

    return {
        "dynamic_pf_values": dynamic_pf_values,
        "dynamic_mdds": dynamic_mdds,
        "dynamic_sharpe_ratios": dynamic_sharpe_ratios,
        "dynamic_sharpe_ratios_ann": dynamic_sharpe_ratios_ann if dynamic_sharpe_ratios_ann else [42],
        "static_pf_values": static_pf_values,
        "static_mdds": static_mdds,
        "static_sharpe_ratios": static_sharpe_ratios,
        "static_sharpe_ratios_ann": static_sharpe_ratios_ann if static_sharpe_ratios_ann else [42],
        "cash_investments": cash_investments,
        "crypto_weight_averages": crypto_weight_averages,
        "crypto_weight_std_devs": crypto_weight_std_devs,
        "first_key": first_key,
        "asset_list": asset_list,
        "eq_pf_value": eq_pf_value,
        "eq_sharpe_ratio": eq_sharpe_ratio,
        "eq_sharpe_ratio_ann": eq_sharpe_ratio_ann if eq_sharpe_ratio_ann else 42,
        "eq_mdd": eq_mdd,
        "test_start": test_start,
        "test_end": test_end,
        "trading_period_length": trading_period_length,
    }


def filter_history_dict(history_dict):

    filtered_history = {}
    for timestamp, train_data in history_dict.items():

        initial_weights = train_data["initial_weights"]

        # Ignore train runs with negative weight
        if any(value < 0 for value in initial_weights):
            continue

        # Ignore train runs with huge weight
        if any(value > 0.7 for value in initial_weights):
            continue

        filtered_history[timestamp] = train_data

    return filtered_history


def _plot_histogram_metadata_table(axis, n_simulations, session_name, backtest_stats):

    divider = make_axes_locatable(axis)

    # add sample size of simulation

    axis.set_axis_off()

    axis.set_title(
        f"[Simulation statistics] {session_name.replace('_', ' ')}",
        fontdict={"fontsize": 20, "position": (0.0, 0.95)},  # x, y
        horizontalalignment="left",
    )

    dynamic_columns = ("Dynamic agent", "Average", "Stdev")
    dynamic_data = [
        [
            "Ptf. value",
            round(np.mean(backtest_stats["dynamic_pf_values"]), 4),
            round(np.std(backtest_stats["dynamic_pf_values"]), 4),
        ],
        [
            "Sharpe ratio",
            round(np.mean(backtest_stats["dynamic_sharpe_ratios"]), 4),
            round(np.std(backtest_stats["dynamic_sharpe_ratios"]), 4),
        ],
        [
            "Sharpe ratio (ann)",
            round(np.mean(backtest_stats["dynamic_sharpe_ratios_ann"]), 4),
            round(np.std(backtest_stats["dynamic_sharpe_ratios_ann"]), 4),
        ],
        [
            "MDD",
            round(np.mean(backtest_stats["dynamic_mdds"]), 4),
            round(np.std(backtest_stats["dynamic_mdds"]), 4),
        ],
        [
            "Average of weights",
            round(np.mean(backtest_stats["crypto_weight_averages"]), 4),
            round(np.std(backtest_stats["crypto_weight_averages"]), 4),
        ],
        [
            "Stdev of weights",
            round(np.mean(backtest_stats["crypto_weight_std_devs"]), 4),
            round(np.std(backtest_stats["crypto_weight_std_devs"]), 4),
        ],
        [
            "Cash weight (BTC)",
            round(np.mean(backtest_stats["cash_investments"]), 4),
            round(np.std(backtest_stats["cash_investments"]), 4),
        ],
    ]

    dynamic_table = axis.table(
        cellText=dynamic_data,
        colLabels=dynamic_columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    _format_table(dynamic_table)

    static_columns = ("Static agent", "Average", "Stdev")
    static_data = [
        [
            "Ptf. value",
            round(np.mean(backtest_stats["static_pf_values"]), 4),
            round(np.std(backtest_stats["static_pf_values"]), 4),
        ],
        [
            "Sharpe ratio",
            round(np.mean(backtest_stats["static_sharpe_ratios"]), 4),
            round(np.std(backtest_stats["static_sharpe_ratios"]), 4),
        ],
        [
            "Sharpe ratio (ann)",
            round(np.mean(backtest_stats["static_sharpe_ratios_ann"]), 4),
            round(np.std(backtest_stats["static_sharpe_ratios_ann"]), 4),
        ],
        [
            "MDD",
            round(np.mean(backtest_stats["static_mdds"]), 4),
            round(np.std(backtest_stats["static_mdds"]), 4),
        ],
        [
            "Average of weights",
            round(np.mean(backtest_stats["crypto_weight_averages"]), 4),
            round(np.std(backtest_stats["crypto_weight_averages"]), 4),
        ],
        [
            "Stdev of weights",
            round(np.mean(backtest_stats["crypto_weight_std_devs"]), 4),
            round(np.std(backtest_stats["crypto_weight_std_devs"]), 4),
        ],
        [
            "Cash weight (BTC)",
            round(np.mean(backtest_stats["cash_investments"]), 4),
            round(np.std(backtest_stats["cash_investments"]), 4),
        ],
    ]

    axis1 = divider.append_axes("right", size="100%", pad=0.15, sharex=axis)
    axis1.set_axis_off()

    static_table = axis1.table(
        cellText=static_data,
        colLabels=static_columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    _format_table(static_table)

    axis2 = divider.append_axes("right", size="55%", pad=0.15, sharex=axis)
    axis2.set_axis_off()

    eq_columns = ("Equal weighted", "Average")
    eq_data = [
        ["Ptf. value", round(backtest_stats["eq_pf_value"], 4)],
        ["Sharpe ratio", round(backtest_stats["eq_sharpe_ratio"], 4)],
        ["Sharpe ratio (ann)", round(
            backtest_stats["eq_sharpe_ratio_ann"], 4)],
        ["MDD", round(backtest_stats["eq_mdd"], 4)],
    ]

    eq_table = axis2.table(
        cellText=eq_data,
        colLabels=eq_columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.6, 0.4],
    )
    _format_table(eq_table)

    axis3 = divider.append_axes("right", size="55%", pad=0.15, sharex=axis)
    axis3.set_axis_off()

    crypto_columns = ("Parameter", "Value")
    crypto_table_data = [
        ["No. of simulations", n_simulations],
        ["No. of assets", len(backtest_stats["asset_list"])],
        ["Start date", backtest_stats["test_start"]],
        ["End date", backtest_stats["test_end"]],
        ["Trading period", backtest_stats["trading_period_length"]],
    ]

    crypto_table = axis3.table(
        cellText=crypto_table_data,
        colLabels=crypto_columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.6, 0.4],
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
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))


if __name__ == "__main__":

    for backtest_json_fn in os.listdir(JSON_OUTPUT_DIR):

        session_name = backtest_json_fn.replace(
            'train_history_', "").replace(".json", "")

        print(session_name)

        make_train_histograms(session_name)
