import numpy as np
import os
import json
from pprint import pprint
import csv

from src.make_train_histograms import filter_history_dict, aggregate_backtest_stats


JSON_OUTPUT_DIR = "train_jsons/"


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
            'Backtest name'.replace("_", " "),
            'PF value (dynamic)',
            'PF value (static)',
            'PF value (eq)',
            'MDD (dynamic)',
            'MDD (static)',
            'MDD (eq)',
            'Sharpe (dynamic)',
            'Sharpe (static)',
            'Sharpe (eq)',
        ])

        for backtest_name, backtest_dict in backtest_dicts.items():

            key_stats = _extract_key_stats(backtest_dict)
            csv_writer.writerow([backtest_name, *key_stats])

    return backtest_dicts


def _extract_key_stats(backtest_dict):

    filtered_history = filter_history_dict(backtest_dict)

    n_simulations = len(filtered_history)

    dynamic_pf_values, dynamic_mdds, dynamic_sharpe_ratios, static_pf_values, static_mdds, static_sharpe_ratios, cash_investments, crypto_weight_averages, crypto_weight_std_devs, first_key, asset_list, eq_pf_value, eq_sharpe_ratio, eq_mdd = aggregate_backtest_stats(
        filtered_history)

    return [
        np.round(np.mean(dynamic_pf_values), 4),
        np.round(np.mean(static_pf_values), 4),
        np.round(eq_pf_value, 4),
        np.round(np.mean(dynamic_mdds), 4),
        np.round(np.mean(static_mdds), 4),
        np.round(eq_mdd, 4),
        np.round(np.mean(dynamic_sharpe_ratios), 4),
        np.round(np.mean(static_sharpe_ratios), 4),
        np.round(eq_sharpe_ratio, 4),
    ]


if __name__ == "__main__":
    make_backtest_aggregation_table()
