import numpy as np
import os
import json
from pprint import pprint
import csv

from src.make_train_histograms import filter_history_dict, aggregate_backtest_stats


JSON_OUTPUT_DIR = "train_jsons/"
BACKTEST_AGGR_CSV_FP = 'backtest_aggregated.csv'

BACKTEST_NROS = {
    "Calm_before_the_storm": {
        "5min": 1,
        "15min": 2,
        "30min": 3,
        "2h": 4,
        "4h": 5,
        "1d": 6,
    },
    "Awakening": {
        "5min": 7,
        "15min": 8,
        "30min": 9,
        "2h": 10,
        "4h": 11,
        "1d": 12,
    },
    "Ripple_bull_run": {
        "5min": 13,
        "15min": 14,
        "30min": 15,
        "2h": 16,
        "4h": 17,
        "1d": 18,
    },
    "Ethereum_valley": {
        "5min": 19,
        "15min": 20,
        "30min": 21,
        "2h": 22,
        "4h": 23,
        "1d": 24,
    },
    "All-time_high": {
        "5min": 25,
        "15min": 26,
        "30min": 27,
        "2h": 28,
        "4h": 29,
        "1d": 30,
    },
    "Rock_bottom": {
        "5min": 31,
        "15min": 32,
        "30min": 33,
        "2h": 34,
        "4h": 35,
        "1d": 36,
    },
    "Recent": {
        "5min": 37,
        "15min": 38,
        "30min": 39,
        "2h": 40,
        "4h": 41,
        "1d": 42,
    }
}

MEGA_TABLE_COLS = [
    'Backtest no.',
    'Backtest name',
    'Date range',
    'Trading period',
    'No. of simulations',

    'PF value (dynamic)',
    'MDD (dynamic)',
    'Sharpe (dynamic)',
    # 'Sharpe, ann. (dynamic)',

    'PF value (static)',
    'MDD (static)',
    'Sharpe (static)',
    # 'Sharpe, ann. (static)',

    'PF value (eq)',
    'MDD (eq)',
    'Sharpe (eq)',
    # 'Sharpe, ann. (eq)',
]


def make_backtest_aggregation_table():

    backtest_dicts = {}

    for backtest_json_fn in os.listdir(JSON_OUTPUT_DIR):
        backtest_json_fp = os.path.join(JSON_OUTPUT_DIR, backtest_json_fn)
        with open(backtest_json_fp, 'r') as file:
            backtest_name = backtest_json_fn.replace(
                ".json", "").replace("train_history_", "")
            backtest_dicts[backtest_name] = json.load(file)

    with open(BACKTEST_AGGR_CSV_FP, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')

        csv_writer.writerow(MEGA_TABLE_COLS)

        collected_backtests = {}

        for backtest_name, backtest_dict in backtest_dicts.items():

            if "Dynamic_agent" in backtest_name:
                continue

            key_stats = _extract_key_stats(backtest_name, backtest_dict)

            if key_stats:
                backtest_name_nice = " ".join(backtest_name.split("_")[:-1])
                trading_period_length = key_stats[1]

                if backtest_name_nice in collected_backtests:
                    collected_backtests[backtest_name_nice][
                        trading_period_length] = key_stats

                else:
                    collected_backtests[backtest_name_nice] = {
                        trading_period_length: key_stats
                    }

                backtest_nro = BACKTEST_NROS[backtest_name_nice.replace(
                    " ", '_')][key_stats[1]]
                csv_writer.writerow([backtest_nro, backtest_name_nice, *key_stats])

    _make_individual_tables_for_backtests(collected_backtests, MEGA_TABLE_COLS)

    return backtest_dicts


def _extract_key_stats(backtest_name, backtest_dict):

    filtered_history = filter_history_dict(backtest_dict, backtest_name, move_valid_to_own_dir = True)

    n_simulations = len(filtered_history)

    if not n_simulations:
        print(f'\nWARNING: Could not create histogram for session {backtest_name}. No valid test runs found from total {len(backtest_dict)}\n')
        return []

    backtest_stats = aggregate_backtest_stats(
        filtered_history)

    return [
        f"{backtest_stats['test_start']} to {backtest_stats['test_end']}",
        backtest_stats["trading_period_length"],
        n_simulations,
        np.round(np.mean(backtest_stats["dynamic_pf_values"]), 4),
        np.round(np.mean(backtest_stats["dynamic_mdds"]), 4),
        np.round(np.mean(backtest_stats["dynamic_sharpe_ratios"]), 4),
        # np.round(np.mean(backtest_stats["dynamic_sharpe_ratios_ann"]), 4),


        np.round(np.mean(backtest_stats["static_pf_values"]), 4),
        np.round(np.mean(backtest_stats["static_mdds"]), 4),
        np.round(np.mean(backtest_stats["static_sharpe_ratios"]), 4),
        # np.round(np.mean(backtest_stats["static_sharpe_ratios_ann"]), 4),


        np.round(backtest_stats["eq_pf_value"], 4),
        np.round(backtest_stats["eq_mdd"], 4),
        np.round(backtest_stats["eq_sharpe_ratio"], 4),
        # np.round(backtest_stats["eq_sharpe_ratio_ann"], 4),
    ]


def _make_individual_tables_for_backtests(collected_backtests, MEGA_TABLE_COLS):

    comparable_stats = {}

    for backtest_name, aggr_stats in collected_backtests.items():

        baktest_stats = {
            "mdds"
            "sharpe_ratios"
            "pf_values"
        }
        for time_period, time_period_stats in aggr_stats.items():
            pass

    # import pdb; pdb.set_trace()
    pass


if __name__ == "__main__":
    make_backtest_aggregation_table()
