import argparse

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

PERIOD_INDECES = {
    "5min": 0,
    "15min": 1,
    "30min": 2,
    "2h": 3,
    "4h": 4,
    "1d": 5,
}

PERIODS = ["15min", "30min", "2h", "4h", "1d"]


SESSION_NAME_START_INDECES = {
    "calm": 0,
    "awake": 5,
    "ripple": 10,
    "ether": 15,
    "high": 20,
    "rock": 25,
    "recent": 30,
}

# Choose which eq weight hour to use for each backtest, default 2h
SESSION_NAME_EQ_INDECES = {
    "calm": 1,
    "awake": 3,
    "ripple": 4,
    "ether": 2,
    "high": 2,
    "rock": 3,
    "recent": 1,
}

BACKTEST_AGGR_CSV_FP = 'backtests_doc_ready.csv'

BACKTEST_AGGR_PLOTS_FP = 'backtest_aggr_plots/'

if not os.path.exists(BACKTEST_AGGR_PLOTS_FP):
    os.mkdir(BACKTEST_AGGR_PLOTS_FP)


def main(hack_equal):

    backtest_dict = _make_backtest_dict(hack_equal)

    # Make report for each backtest
    for backtest_name, backtest_stats in backtest_dict.items():

        fig, axes = plt.subplots(nrows=3, ncols=3)

        # width, height
        # fig.set_size_inches(16.6, 8)
        fig.set_size_inches(15.5, 11)

        gs = axes[2][0].get_gridspec()
        axes[0][0].remove()
        axes[0][1].remove()
        axes[0][2].remove()

        axes[2][0].remove()
        axes[2][1].remove()
        axes[2][2].remove()

        # make table similar to train reports
        # pf value variance
        table_ax = fig.add_subplot(gs[0:3])

        relative_ax = fig.add_subplot(3, 1, 3)

        _make_backtest_summary_table(table_ax, backtest_name, backtest_stats)

        # make plots 3 x 2 (indicators x [static, dynamic]
        _plot_line(
            axes[1][0],
            backtest_stats["dynamic"]["pf_value"],
            "Average Portfolio values",
            None,
            "Portfolio value",
            "Dynamic",
        )
        _plot_line(
            axes[1][0],
            backtest_stats["static"]["pf_value"],
            "Average Portfolio values",
            None,
            "Portfolio value",
            "Static",
        )
        _plot_line(
            axes[1][0],
            backtest_stats["equal"]["pf_value"],
            "Average Portfolio values",
            None,
            "Portfolio value",
            "Equal",
        )

        axes[1][0].yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        _plot_line(
            axes[1][1],
            backtest_stats["dynamic"]["mdd"],
            "Average Maximum drawdowns",
            None,
            "Maximum drawdown",
            "Dynamic",
        )
        _plot_line(
            axes[1][1],
            backtest_stats["static"]["mdd"],
            "Average Maximum drawdowns",
            None,
            "Maximum drawdown",
            "Static",
        )
        _plot_line(
            axes[1][1],
            backtest_stats["equal"]["mdd"],
            "Average Maximum drawdowns",
            None,
            "Maximum drawdown",
            "Equal",
        )

        _plot_line(
            axes[1][2],
            backtest_stats["dynamic"]["sharpe"],
            "Average Sharpe ratios",
            None,
            "Sharpe ratio",
            "Dynamic",
        )

        _plot_line(
            axes[1][2],
            backtest_stats["static"]["sharpe"],
            "Average Sharpe ratios",
            None,
            "Sharpe ratio",
            "Static",
        )
        _plot_line(
            axes[1][2],
            backtest_stats["equal"]["sharpe"],
            "Average Sharpe ratios",
            None,
            "Sharpe ratio",
            "Equal",
        )

        divider = make_axes_locatable(relative_ax)
        _plot_line(
            relative_ax,
            backtest_stats["dynamic_over_static"]["pf_value"],
            "Impact of trading action: Added Ptf. value",
            None,
            None,
            "Relative ptf. value",
        )

        relative_ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        relative_ax2 = divider.append_axes(
            "right", size="100%", pad=0.7, sharex=relative_ax)

        _plot_line(
            relative_ax2,
            backtest_stats["dynamic_over_static"]["std_dev"],
            "Impact of trading action: Added Stdev",
            None,
            None,
            "Relative stdev",
        )
        relative_ax2.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        output_path = os.path.join(BACKTEST_AGGR_PLOTS_FP, f"backtest_aggr_plot_{backtest_name}.png")
        print(f"Saving plot to path: {output_path}")
        plt.subplots_adjust(hspace=0.35, wspace=.3)
        plt.savefig(output_path, bbox_inches="tight")


def _make_backtest_summary_table(axis, session_name, backtest_stats):
    axis.set_axis_off()

    for key in SESSION_NAME_START_INDECES.keys():
        if key in session_name.lower():
            start_idx = SESSION_NAME_START_INDECES[key]

    simulation_nos = list(range(start_idx, start_idx + 6))

    meta_header = axis.table(
        cellText=[['']],
        colLabels=[session_name],
        loc='center',
        bbox=[
            0.0,  # x offset
            0.54,  # y offset
            0.2,  # width
            0.3  # height
        ],
    )

    meta_table_data = [
        [simulation_nos[1], "15min", backtest_stats["no_of_simulations"][1]],
        [simulation_nos[2], "30min", backtest_stats["no_of_simulations"][2]],
        [simulation_nos[3], "2h", backtest_stats["no_of_simulations"][3]],
        [simulation_nos[4], "4h", backtest_stats["no_of_simulations"][4]],
        [simulation_nos[5], "1d", backtest_stats["no_of_simulations"][5]],
    ]

    meta_columns = ("#", "Period", "Simulations")

    meta_table = axis.table(
        cellText=meta_table_data,
        colLabels=meta_columns,
        loc='center',
        bbox=[
            0.0,
            0,  # -0.35,
            0.2,
            0.7
        ],
        cellLoc="center",
        colWidths=[0.2, 0.3, 0.5]

    )
    meta_table.auto_set_font_size(False)
    meta_table.set_fontsize(10)

    _format_table(meta_table)
    _format_table(meta_header)

    summary_columns = (
        "Ptf. value", "MDD", "Sharpe", "Ptf. value", "MDD", "Sharpe",  "Ptf. value", "MDD", "Sharpe")

    summary_data = []

    for i in range(5):
        summary_data.append(
            [
                str(round(backtest_stats["dynamic"][
                    "pf_value"][i] * 100, 2)) + "%",
                backtest_stats["dynamic"]["mdd"][i],
                backtest_stats["dynamic"]["sharpe"][i],
                str(round(backtest_stats["static"][
                    "pf_value"][i] * 100, 2)) + "%",
                backtest_stats["static"]["mdd"][i],
                backtest_stats["static"]["sharpe"][i],
                str(round(backtest_stats["equal"][
                    "pf_value"][i] * 100, 2)) + "%",
                backtest_stats["equal"]["mdd"][i],
                backtest_stats["equal"]["sharpe"][i],
            ]
        )

    # Create custom widths for cells
    no_width = 0.03
    period_width = 0.05
    simulations_width = 0.09

    others_width = [
        (1 - (no_width + period_width + simulations_width)) / 9] * 9
    col_widths = [no_width, period_width, simulations_width]
    col_widths.extend(others_width)

    # add extra header for agent types

    summary_header = axis.table(
        cellText=[[''] * 3],
        colLabels=['Dynamic agent',
                   'Static agent', "Equal weighted"],
        loc='center',
        bbox=[
            0.2,  # x offset
            0.54,  # y offset
            0.8,  # width
            0.3  # height
        ],
    )

    summary_table = axis.table(
        cellText=summary_data,
        colLabels=summary_columns,
        loc='center',
        bbox=[
            0.2,
            0,  # -0.35,
            0.8,
            0.7
        ],
        cellLoc="center"

    )

    _format_table(summary_header)
    _format_table(summary_table)


def _format_table(table):
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(
                fontproperties=FontProperties(weight="bold"), color="white",

            )
            cell.set_facecolor("#4C72B0")


def _plot_line(axis, data, title, xlabel, ylabel, label=None):

    axis.plot(data, label=label)

    axis.grid(alpha=0.3)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(9)

    if label:
        axis.legend()


def _make_backtest_dict(hack_equal):

    aggr_backtest_dict = {}

    """
    1. open csv
    2. separate each backtest to own dict

    {"awakening":
        "dynamic": {
            "pf_value": pd series / idx 5 min, 15min, 30 min,  1h 2h, 4h
            "mdd": pd series
            "sharpe": pd series
        },
        "static": {
            "pf_value": pd series / idx 5 min, 15min, 30 min,  1h 2h, 4h
            "mdd": pd series
            "sharpe": pd series
        }

        "equal": {
            "pf_value": 2h mean
            "mdd": 2h mean
            "sharpe": 2h mean
        }

     }

    """
    with open(BACKTEST_AGGR_CSV_FP, 'r') as file:

        swag_reader = csv.reader(file, delimiter=',')

        for row_idx, row in enumerate(swag_reader):
            # skip header
            if row_idx == 0:
                continue

            backtest_name = row[1]

            backtest_period = row[3]
            period_idx = PERIOD_INDECES[backtest_period]

            backtest_stats = aggr_backtest_dict.get(
                backtest_name,
                {
                    "dynamic": {
                        "pf_value": [0] * 6,
                        "mdd": [0] * 6,
                        "sharpe": [0] * 6,
                    },
                    "static": {
                        "pf_value": [0] * 6,
                        "mdd": [0] * 6,
                        "sharpe": [0] * 6,
                    },
                    "equal": {
                        "pf_value": [0] * 6,
                        "mdd": [0] * 6,
                        "sharpe": [0] * 6,
                    },
                    "dynamic_over_static": {
                        "pf_value": [0] * 6,
                        "std_dev": [0] * 6,
                    },
                    "no_of_simulations": [0] * 6
                }
            )

            backtest_stats["no_of_simulations"][period_idx] = row[4]

            backtest_stats["dynamic"]["pf_value"][
                period_idx] = float(row[5]) - 1
            backtest_stats["dynamic"]["mdd"][period_idx] = row[6]
            backtest_stats["dynamic"]["sharpe"][period_idx] = row[7]

            backtest_stats["static"]["pf_value"][
                period_idx] = float(row[8]) - 1
            backtest_stats["static"]["mdd"][period_idx] = row[9]
            backtest_stats["static"]["sharpe"][period_idx] = row[10]

            dyn_stdev = float(row[14])
            stat_stdev = float(row[15])

            backtest_stats['dynamic_over_static']["pf_value"][
                period_idx] = (float(row[5]) / float(row[8])) - 1

            backtest_stats['dynamic_over_static'][
                "std_dev"][period_idx] = (dyn_stdev / stat_stdev) - 1

            if not hack_equal:

                backtest_stats["equal"]["pf_value"][
                    period_idx] = float(row[11]) - 1
                backtest_stats["equal"]["mdd"][period_idx] = row[12]
                backtest_stats["equal"]["sharpe"][period_idx] = row[13]

            else:
                for sess_name_part in SESSION_NAME_EQ_INDECES.keys():
                    if sess_name_part in backtest_name.lower():
                        eq_period_idx = SESSION_NAME_EQ_INDECES[sess_name_part]

                if eq_period_idx == period_idx:
                    backtest_stats["equal"]["pf_value"] = [
                        float(row[11]) - 1] * 6
                    backtest_stats["equal"]["mdd"] = [row[12]] * 6
                    backtest_stats["equal"]["sharpe"] = [row[13]] * 6

            aggr_backtest_dict[backtest_name] = backtest_stats

    for bt_stats in aggr_backtest_dict.values():

        for label in ["dynamic", "static", "equal"]:

            bt_stats[label]["pf_value"] = pd.Series(
                bt_stats[label]["pf_value"][1:],
                index=PERIODS,
                dtype=float
            )
            bt_stats[label]["mdd"] = pd.Series(
                bt_stats[label]["mdd"][1:],
                index=PERIODS,
                dtype=float
            )
            bt_stats[label]["sharpe"] = pd.Series(
                bt_stats[label]["sharpe"][1:],
                index=PERIODS,
                dtype=float
            )

        bt_stats["dynamic_over_static"]["pf_value"] = pd.Series(
            bt_stats["dynamic_over_static"]["pf_value"][1:],
            index=PERIODS,
            dtype=float
        )
        bt_stats["dynamic_over_static"]["std_dev"] = pd.Series(
            bt_stats["dynamic_over_static"]["std_dev"][1:],
            index=PERIODS,
            dtype=float
        )

    return aggr_backtest_dict


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("-he", "--hack_equal",
                        action="store_true", default=False)
    ARGS = PARSER.parse_args()

    if not ARGS.hack_equal:
        print(
            '\nWARNING: Equal weights is not hacked! Please ensure this is not document build!\n')

    else:
        print(
            '\nUsing hacky document build! Please ensure correct thresholds\n')

    main(ARGS.hack_equal)
