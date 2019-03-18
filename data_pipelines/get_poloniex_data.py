import sys
import argparse
import os
import time
from datetime import datetime

import pandas as pd

# https://docs.poloniex.com/#returntradehistory-public
# https://github.com/jyunfan/poloniex-data/blob/master/getdata.py

FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}"
DATA_DIR = "data_temp"
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

START_DATE = 1388534400
END_DATE = 1388834400


UNIX_EPOCH_DATES = {"2017": 1483228800, "2018": 1514764800, "2019": 1546300800}

PERIOD_LENGTHS = {
    "5min:": 300,
    "15min": 900,
    "30min": 1800,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


def main(start_date, end_date, period_length, pairs):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    for pair in pairs:
        get_data_own(pair, start_date, end_date, period_length)
        time.sleep(2)


def get_all_pairs():
    pairs_df = pd.read_json("https://poloniex.com/public?command=return24hVolume")
    pairs = [pair for pair in pairs_df.columns if pair.startswith("BTC")]
    return pairs


def get_data_own(pair, start_date, end_date, period_length):
    start_epoch = int(datetime.strptime(ARGS.start_date, "%Y%m%d").timestamp())
    end_epoch = int(datetime.strptime(ARGS.end_date, "%Y%m%d").timestamp())

    filename = "{}_{}-{}_{}.csv".format(pair, start_date, end_date, period_length)

    datafile = os.path.join(DATA_DIR, filename)

    period_length_in_secs = PERIOD_LENGTHS[period_length]

    url = FETCH_URL.format(pair, start_epoch, end_epoch, period_length_in_secs)
    print("Get {} from {} to {}".format(pair, start_epoch, end_epoch))

    df = pd.read_json(url, convert_dates=False)

    if df["date"].iloc[-1] == 0:
        print("No data.")
        return

    with open(datafile, "w") as file:
        print("Saving file {}".format(filename))
        df.to_csv(file, index=False, columns=COLUMNS)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "-sd", "--start_date", type=str, default=None, help="date in format YYYYMMDD"
    )
    PARSER.add_argument(
        "-ed", "--end_date", type=str, default=None, help="date in format YYYYMMDD"
    )
    PARSER.add_argument(
        "-p", "--period", type=str, default="30min", help="Trade period length"
    )

    ARGS = PARSER.parse_args()

    if not ARGS.start_date or not ARGS.end_date:
        print("Please provide start and end dates as kwargs")
        sys.exit(0)

    PAIRS = ["BTC_ETH"]

    main(ARGS.start_date, ARGS.end_date, ARGS.period, PAIRS)
