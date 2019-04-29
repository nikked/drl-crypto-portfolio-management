import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint

from make_backtest_aggregation_table import BACKTEST_AGGR_CSV_FP

PERIOD_INDECES = {
    "5min": 0,
    "15min": 1,
    "30min": 2,
    "2h": 3,
    "4h": 4,
    "1d": 5,
}


BACKTEST_AGGR_PLOTS_FP = 'backtest_aggr_plots/'

if not os.path.exists(BACKTEST_AGGR_PLOTS_FP):
    os.mkdir(BACKTEST_AGGR_PLOTS_FP)


def main():

    backtest_dict = _make_backtest_dict()

    # Make report for each backtest
    for backtest_name, backtest_stats in backtest_dict.items():

        fig, axes = plt.subplots(nrows=2, ncols=3)
        # width, height
        # fig.set_size_inches(16.6, 8)
        fig.set_size_inches(12.5, 6)

        gs = axes[0][0].get_gridspec()
        axes[0][0].remove()
        axes[0][1].remove()
        axes[0][2].remove()

        # make table similar to train reports
        # pf value variance
        table_ax = fig.add_subplot(gs[0:3])

        _make_backtest_summary_table(table_ax, backtest_name)

        # make plots 3 x 2 (indicators x [static, dynamic]
        _plot_line(
            axes[1][0],
            backtest_stats["dynamic"]["pf_value"],
            "Average Portfolio values",
            "Period lengths",
            "Portfolio value",
            "Dynamic"
        )
        _plot_line(
            axes[1][0],
            backtest_stats["static"]["pf_value"],
            "Average Portfolio values",
            "Period lengths",
            "Portfolio value",
            "Static"
        )
        _plot_line(
            axes[1][0],
            backtest_stats["equal"]["pf_value"],
            "Average Portfolio values",
            "Period lengths",
            "Portfolio value",
            "Equal"
        )

        _plot_line(
            axes[1][1],
            backtest_stats["dynamic"]["mdd"],
            "Average Maximum drawdowns",
            "Period lengths",
            "Maximum drawdown",
            "Dynamic"
        )
        _plot_line(
            axes[1][1],
            backtest_stats["static"]["mdd"],
            "Average Maximum drawdowns",
            "Period lengths",
            "Maximum drawdown",
            "Static",
        )
        _plot_line(
            axes[1][1],
            backtest_stats["equal"]["mdd"],
            "Average Maximum drawdowns",
            "Period lengths",
            "Maximum drawdown",
            "Equal",
        )

        _plot_line(
            axes[1][2],
            backtest_stats["dynamic"]["sharpe"],
            "Average Sharpe ratios",
            "Period lengths",
            "Sharpe ratio",
            "Dynamic"

        )

        _plot_line(
            axes[1][2],
            backtest_stats["static"]["sharpe"],
            "Average Sharpe ratios",
            "Period lengths",
            "Sharpe ratio",
            "Static",
        )
        _plot_line(
            axes[1][2],
            backtest_stats["equal"]["sharpe"],
            "Average Sharpe ratios",
            "Period lengths",
            "Sharpe ratio",
            "Equal",
        )

        output_path = os.path.join(BACKTEST_AGGR_PLOTS_FP, f"backtest_aggr_plot_{backtest_name}.png")
        print(f"Saving plot to path: {output_path}")
        plt.savefig(output_path, bbox_inches="tight")


def _make_backtest_summary_table(axis, session_name):
    axis.set_axis_off()

    axis.set_title(
        session_name,
        fontdict={"fontsize": 20, "position": (0.0, 0.85)},  # x, y
        horizontalalignment="left",
    )

    summary_columns = (
        "#", "Period", "Simulations", "Ptf. value", "MDD", "Sharpe", "Ptf. value", "MDD", "Sharpe",  "Ptf. value", "MDD", "Sharpe")

    summary_data = [
        [0] * 12,
        [0] * 12,
        [0] * 12,
        [0] * 12,
        [0] * 12,
        [0] * 12,
        [0] * 12,
    ]

    no_width = 0.03
    period_width = 0.05
    simulations_width = 0.09

    others_width = [
        (1 - (no_width + period_width + simulations_width)) / 9] * 9

    col_widths = [no_width, period_width, simulations_width]

    col_widths.extend(others_width)

    summary_table = axis.table(
        cellText=summary_data,
        colLabels=summary_columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths
    )


def _plot_line(axis, data, title, xlabel, ylabel, label="makkispekkis"):
    num_bins = 15

    sns.lineplot(data=data,
                 ax=axis, legend="full", label=label)
    # axis.hist(data, num_bins, alpha=0.95)
    axis.grid(alpha=0.3)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    # axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    axis.set_xticklabels(data.index)


def _make_backtest_dict():

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
                    }
                }
            )

            backtest_stats["dynamic"]["pf_value"][period_idx] = row[5]
            backtest_stats["dynamic"]["mdd"][period_idx] = row[6]
            backtest_stats["dynamic"]["sharpe"][period_idx] = row[7]

            backtest_stats["static"]["pf_value"][period_idx] = row[8]
            backtest_stats["static"]["mdd"][period_idx] = row[9]
            backtest_stats["static"]["sharpe"][period_idx] = row[10]

            # Take the 2h values for equal weight
            if period_idx == 3:
                backtest_stats["equal"]["pf_value"] = [row[11]] * 6
                backtest_stats["equal"]["mdd"] = [row[12]] * 6
                backtest_stats["equal"]["sharpe"] = [row[13]] * 6

            aggr_backtest_dict[backtest_name] = backtest_stats

    pd_series_idx = ["5min", "15min", "30min", "2h", "4h", "1d"]

    for bt_stats in aggr_backtest_dict.values():

        for label in ["dynamic", "static", "equal"]:

            bt_stats[label]["pf_value"] = pd.Series(
                bt_stats[label]["pf_value"],
                index=pd_series_idx,
                dtype=float
            )
            bt_stats[label]["mdd"] = pd.Series(
                bt_stats[label]["mdd"],
                index=pd_series_idx,
                dtype=float
            )
            bt_stats[label]["sharpe"] = pd.Series(
                bt_stats[label]["sharpe"],
                index=pd_series_idx,
                dtype=float
            )

    return aggr_backtest_dict


if __name__ == "__main__":
    main()
