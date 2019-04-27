import numpy as np
import os
import json
from pprint import pprint
import csv

from src.make_train_histograms import filter_history_dict, aggregate_backtest_stats


JSON_OUTPUT_DIR = "train_jsons/"

BACKTEST_NROS = {
    "Calm_before_the_storm": {
        "5min": 1,
        "15min": 2,
        "30min": 3
    },
    "Awakening": {
        "5min": 4,
        "15min": 5,
        "30min": 6
    },
    "Ripple_bull_run": {
        "5min": 7,
        "15min": 8,
        "30min": 9
    },
    "Ethereum_valley": {
        "5min": 10,
        "15min": 11,
        "30min": 12
    },
    "All-time_high": {
        "5min": 13,
        "15min": 14,
        "30min": 15
    },
    "Rock_bottom": {
        "5min": 16,
        "15min": 17,
        "30min": 18
    },
    "Recent": {
        "5min": 19,
        "15min": 20,
        "30min": 21
    }
}


def make_backtest_aggregation_table():

    backtest_dicts = {}

    for backtest_json_fn in os.listdir(JSON_OUTPUT_DIR):
        backtest_json_fp = os.path.join(JSON_OUTPUT_DIR, backtest_json_fn)
        with open(backtest_json_fp, 'r') as file:
            backtest_name = backtest_json_fn.replace(
                ".json", "").replace("train_history_", "")
            backtest_dicts[backtest_name] = json.load(file)

    with open('backtest_aggregated.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')

        csv_writer.writerow([
            'Backtest no.',
            'Backtest name'.replace("_", " "),
            'Date range',
            'Trading period',
            'No. of simulations',

            'PF value (dynamic)',
            'MDD (dynamic)',
            'Sharpe (dynamic)',
            'Sharpe, ann. (dynamic)',

            'PF value (static)',
            'MDD (static)',
            'Sharpe (static)',
            'Sharpe, ann. (static)',

            'PF value (eq)',
            'MDD (eq)',
            'Sharpe (eq)',
            'Sharpe, ann. (eq)',
        ])

        for backtest_name, backtest_dict in backtest_dicts.items():
            print(backtest_name)

            key_stats = _extract_key_stats(backtest_name, backtest_dict)

            if key_stats:
                backtest_name = " ".join(backtest_name.split("_")[:-1])
                backtest_nro = BACKTEST_NROS[backtest_name.replace(
                    " ", '_')][key_stats[1]]
                csv_writer.writerow([backtest_nro, backtest_name, *key_stats])

    return backtest_dicts


def _extract_key_stats(backtest_name, backtest_dict):

    filtered_history = filter_history_dict(backtest_dict)

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
        np.round(np.mean(backtest_stats["dynamic_sharpe_ratios_ann"]), 4),


        np.round(np.mean(backtest_stats["static_pf_values"]), 4),
        np.round(np.mean(backtest_stats["static_mdds"]), 4),
        np.round(np.mean(backtest_stats["static_sharpe_ratios"]), 4),
        np.round(np.mean(backtest_stats["static_sharpe_ratios_ann"]), 4),


        np.round(backtest_stats["eq_pf_value"], 4),
        np.round(backtest_stats["eq_mdd"], 4),
        np.round(backtest_stats["eq_sharpe_ratio"], 4),
        np.round(backtest_stats["eq_sharpe_ratio_ann"], 4),
    ]


if __name__ == "__main__":
    make_backtest_aggregation_table()
