import time
import argparse

import numpy as np

from src.train_rl_algorithm import train_rl_algorithm
from src.test_rl_algorithm import test_rl_algorithm
from src.analysis import analysis
from src.environment import TradeEnv

from src.params import (
    LENGTH_TENSOR,
    PF_INITIAL_VALUE,
    TRADING_COST,
    INTEREST_RATE,
    RATIO_TRAIN,
    RATIO_VAL,
)

from data_pipelines import data_pipe


# CRYPTO_DATA_FP = "./data/np_data/inputCrypto.npy"
# STOCK_DATA_FP = "./data/np_data/input.npy"

DEFAULT_TRADE_ENV_ARGS = {
    "path": None,
    "window_length": LENGTH_TENSOR,
    "portfolio_value": PF_INITIAL_VALUE,
    "trading_cost": TRADING_COST,
    "interest_rate": INTEREST_RATE,
    "train_size": RATIO_TRAIN,
    "data": None,
}


def main(**cli_options):  # pylint: disable=too-many-locals

    no_of_assets = cli_options["no_of_assets"]

    start_time = time.time()

    trade_env_args, asset_list, data_features = _get_data_for_trade_envs(
        max_no_of_assets=no_of_assets
    )

    nb_feature_map = data_features["nb_feature_map"]
    trading_period = data_features["trading_period"]

    total_steps_train, total_steps_val, total_steps_test = _get_train_val_test_steps(
        trading_period
    )

    weights_equal = np.array(np.array([1 / (no_of_assets + 1)] * (no_of_assets + 1)))
    weights_single = np.array(np.array([1] + [0.0] * no_of_assets))

    # Creation of the trading environment
    env, env_eq, env_s, action_fu, env_fu = _get_train_environments(
        no_of_assets, trade_env_args
    )

    # Agent training
    actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s = train_rl_algorithm(
        cli_options["interactive_session"],
        env,
        env_eq,
        env_s,
        action_fu,
        env_fu,
        trade_env_args,
        weights_equal,
        weights_single,
        asset_list,
        total_steps_train,
        total_steps_val,
        nb_feature_map,
        no_of_assets,
        cli_options["gpu_device"],
        cli_options["verbose"],
    )

    # Agent evaluation
    p_list, p_list_eq, p_list_fu, p_list_s, w_list = test_rl_algorithm(
        actor,
        state_fu,
        done_fu,
        env,
        env_eq,
        env_s,
        action_fu,
        env_fu,
        total_steps_train,
        total_steps_val,
        total_steps_test,
        weights_equal,
        weights_single,
        no_of_assets,
    )

    # Analysis
    analysis(
        p_list,
        p_list_eq,
        p_list_s,
        p_list_fu,
        w_list,
        list_final_pf,
        list_final_pf_eq,
        list_final_pf_s,
        "stocks",
        total_steps_train,
        total_steps_val,
        no_of_assets,
        asset_list,
    )

    end_time = time.time()
    train_time_secs = round(end_time - start_time, 1)

    print("\nTraining completed")
    print(f"Process took {train_time_secs} seconds")


def _get_data_for_trade_envs(max_no_of_assets=3):
    dataset, asset_names = data_pipe.main(
        count_of_stocks=max_no_of_assets,
        max_count_of_periods=100)

    trade_env_args = DEFAULT_TRADE_ENV_ARGS
    trade_env_args["data"] = dataset

    data_features = {
        "nb_stocks": dataset.shape[1],
        "nb_feature_map": dataset.shape[0],
        "trading_period": dataset.shape[2],
    }

    print("\nStarting training for {} assets".format(len(asset_names)))
    print(asset_names)

    return trade_env_args, asset_names, data_features


def _get_train_val_test_steps(trading_period):
    # Total number of steps for pre-training in the training set
    total_steps_train = int(RATIO_TRAIN * trading_period)

    # Total number of steps for pre-training in the validation set
    total_steps_val = int(RATIO_VAL * trading_period)

    # Total number of steps for the test
    total_steps_test = trading_period - total_steps_train - total_steps_val

    return total_steps_train, total_steps_val, total_steps_test


def _get_train_environments(nb_stocks, trade_env_args):

    # environment for trading of the agent
    # this is the agent trading environment (policy network agent)

    env = TradeEnv(**trade_env_args)

    # environment for equiweighted
    # this environment is set up for an agent who only plays an equiweithed
    # portfolio (baseline)
    env_eq = TradeEnv(**trade_env_args)

    # environment secured (only money)
    # this environment is set up for an agentwho plays secure, keeps its money

    env_s = TradeEnv(**trade_env_args)

    # full on one stock environment
    # these environments are set up for agents who play only on one stock
    action_fu = list()
    env_fu = list()

    for i in range(nb_stocks):
        action = np.array([0] * (i + 1) + [1] + [0] * (nb_stocks - (i + 1)))
        action_fu.append(action)

        env_fu_i = TradeEnv(**trade_env_args)

        env_fu.append(env_fu_i)

    return env, env_eq, env_s, action_fu, env_fu


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "-i",
        "--interactive_session",
        action="store_true",
        help="Plot interactively with matplotlib",
        default=False,
    )

    PARSER.add_argument(
        "-c",
        "--crypto_data",
        action="store_true",
        help="Use cryptocurrency data",
        default=False,
    )

    PARSER.add_argument(
        "-g", "--gpu_device", type=int, help="Choose GPU device number", default=None
    )
    PARSER.add_argument(
        "-n",
        "--no_of_assets",
        type=int,
        help="Choose how many assets are trained",
        default=5,
    )
    PARSER.add_argument(
        "-v",
        "--verbose",
        help="Print train vectors",
        default=False,
        action="store_true",
    )

    ARGS = PARSER.parse_args()

    if ARGS.verbose:
        print("\nVerbose session. Alot of vectors will be printed below.\n")

    main(
        interactive_session=ARGS.interactive_session,
        crypto_data=ARGS.crypto_data,
        gpu_device=ARGS.gpu_device,
        verbose=ARGS.verbose,
        no_of_assets=ARGS.no_of_assets,
    )
