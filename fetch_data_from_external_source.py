import argparse

from data_pipelines import data_pipe, data_pipe_poloniex


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "-c",
        "--crypto_data",
        action="store_true",
        help="Fetch cryptocurrency data",
        default=False,
    )

    PARSER.add_argument(
        "-s",
        "--stock_data",
        action="store_true",
        help="Fetch stock data",
        default=False,
    )

    ARGS = PARSER.parse_args()

    if ARGS.stock_data:
        data_pipe.main()

    elif ARGS.crypto_data:
        data_pipe_poloniex.main()

    else:
        print("Please choose a datasource. Use the -h kw for help.")
