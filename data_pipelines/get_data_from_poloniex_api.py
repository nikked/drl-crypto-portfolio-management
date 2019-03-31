import sys
import argparse
import os
import time
from datetime import datetime
from pprint import pprint

import pandas as pd


from src.params import PERIOD_LENGTHS


# https://docs.poloniex.com/#returntradehistory-public
# https://github.com/jyunfan/poloniex-data/blob/master/getdata.py

FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}"
COLUMNS = [
    "date",
    "high",
    "low",
    "open",
    "close",
    "volume",
    "quoteVolume",
    "weightedAverage",
]

DATA_DIR = "crypto_data/"


def download_crypto_portfolio_data(start_date, end_date, period_length, pairs):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    for pair in pairs:
        download_crypto_data(pair, start_date, end_date, period_length)


def download_crypto_data(crypto_pair, start_date, end_date, period_length):
    print(f"Downloading {crypto_pair}")
    output_fn = "{}_{}-{}_{}.csv".format(
        crypto_pair, start_date, end_date, period_length
    )

    output_dir = os.path.join(DATA_DIR, f"{crypto_pair}", f"{start_date}-{end_date}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_fp = os.path.join(output_dir, output_fn)

    if os.path.isfile(output_fp):
        print(f"Skipping. File {output_fp} has already been downloaded.")

    else:
        get_data_from_poloniex(
            output_fp, crypto_pair, start_date, end_date, period_length
        )
        time.sleep(0.3)
    print()


def print_all_pairs():
    pairs_df = pd.read_json("https://poloniex.com/public?command=return24hVolume")
    pprint([pair for pair in pairs_df.columns])


def get_data_from_poloniex(output_fp, pair, start_date, end_date, period_length):
    start_epoch = int(datetime.strptime(start_date, "%Y%m%d").timestamp())
    end_epoch = int(datetime.strptime(end_date, "%Y%m%d").timestamp())

    period_length_in_secs = PERIOD_LENGTHS[period_length]

    url = FETCH_URL.format(pair, start_epoch, end_epoch, period_length_in_secs)
    print("Get {} from {} to {}".format(pair, start_epoch, end_epoch))

    price_history_df = pd.read_json(url, convert_dates=False)

    if price_history_df["date"].iloc[-1] == 0:
        print("No data.")

    with open(output_fp, "w") as file:
        print("Saving file {}".format(output_fp))
        price_history_df.to_csv(file, index=False, columns=COLUMNS)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "-sd",
        "--start_date",
        type=str,
        default="20170601",
        help="date in format YYYYMMDD",
    )
    PARSER.add_argument(
        "-ed",
        "--end_date",
        type=str,
        default="20171231",
        help="date in format YYYYMMDD",
    )
    PARSER.add_argument(
        "-pl", "--period_length", type=str, default="1d", help="Trade period length"
    )

    PARSER.add_argument(
        "-pp",
        "--print_all_pairs",
        help="print_pairs",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-tp", "--top_pairs", help="top pairs", default=False, action="store_true"
    )

    ARGS = PARSER.parse_args()

    PAIRS = ["BTC_ETH"]

    if ARGS.top_pairs:
        PAIRS = [
            "BTC_BCH",
            "BTC_DASH",
            "BTC_EOS",
            "BTC_ETH",
            "BTC_LTC",
            "BTC_XMR",
            "BTC_XRP",
            "BTC_DOGE",
            "BTC_ETC",
            "USDT_BTC",
        ]

    if ARGS.print_all_pairs:
        print_all_pairs()
        sys.exit(0)

    if not ARGS.start_date or not ARGS.end_date:
        print("Please provide start and end dates as kwargs")
        sys.exit(0)

    download_crypto_portfolio_data(
        ARGS.start_date, ARGS.end_date, ARGS.period_length, PAIRS
    )
