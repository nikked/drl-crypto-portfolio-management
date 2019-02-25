import time
import argparse
from pprint import pprint

import numpy as np

from src.train_rl_algorithm import train_rl_algorithm
from src.test_rl_algorithm import test_rl_algorithm
from src import plot_training_results
from src.environment import TradeEnv

from src.params import (
    PF_INITIAL_VALUE,
    TRADING_COST,
    INTEREST_RATE,
    RATIO_TRAIN,
    RATIO_VAL,
    WINDOW_LENGTH,
)

from data_pipelines import data_pipe


# CRYPTO_DATA_FP = "./data/np_data/inputCrypto.npy"
# STOCK_DATA_FP = "./data/np_data/input.npy"

DEFAULT_TRADE_ENV_ARGS = {
    "path": None,
    "window_length": WINDOW_LENGTH,
    "portfolio_value": PF_INITIAL_VALUE,
    "trading_cost": TRADING_COST,
    "interest_rate": INTEREST_RATE,
    "train_size": RATIO_TRAIN,
    "data": None,
}


def main(**cli_options):  # pylint: disable=too-many-locals

    print("\nStarting training process with the following options:")
    pprint(cli_options)

    start_time = time.time()

    # Creation of the trading environment
    print("\n")
    trade_envs, asset_list, step_counts = _initialize_trade_envs(
        window_length=cli_options["window_length"],
        no_of_assets=cli_options["no_of_assets"],
        max_no_of_training_periods=cli_options["max_no_of_training_periods"],
    )

    # Agent training
    actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s = train_rl_algorithm(
        cli_options["window_length"],
        cli_options["n_episodes"],
        cli_options["n_batches"],
        cli_options["interactive_session"],
        trade_envs,
        asset_list,
        step_counts,
        cli_options["gpu_device"],
        cli_options["verbose"],
    )

    # Agent evaluation
    p_list, p_list_eq, p_list_fu, p_list_s, w_list = test_rl_algorithm(
        cli_options["window_length"], actor, state_fu, done_fu, trade_envs, step_counts
    )

    end_time = time.time()
    train_time_secs = round(end_time - start_time, 1)

    print("\nTraining completed")
    print(f"Process took {train_time_secs} seconds")

    if cli_options["plot_results"]:
        plot_training_results(
            cli_options["window_length"],
            cli_options["n_batches"],
            p_list,
            p_list_eq,
            p_list_s,
            p_list_fu,
            w_list,
            list_final_pf,
            list_final_pf_eq,
            list_final_pf_s,
            "stocks",
            asset_list,
        )


def _initialize_trade_envs(
    window_length=10, no_of_assets=3, max_no_of_training_periods=10000
):
    dataset, asset_names = data_pipe.main(
        count_of_stocks=no_of_assets, max_count_of_periods=max_no_of_training_periods
    )

    trade_env_args = DEFAULT_TRADE_ENV_ARGS
    trade_env_args["data"] = dataset
    trade_env_args["window_length"] = window_length

    trading_periods = dataset.shape[2]

    print("Trading periods: {}".format(dataset.shape[2]))
    print("Starting training for {} assets".format(len(asset_names)))
    print(asset_names)

    train_envs = _get_train_environments(no_of_assets, trade_env_args)

    step_counts = _get_train_val_test_steps(trading_periods)

    return train_envs, asset_names, step_counts


def _get_train_environments(no_of_assets, trade_env_args):

    # environment for trading of the agent
    # this is the agent trading environment (policy network agent)

    env = TradeEnv(**trade_env_args)

    # environment for equally weighted
    # this environment is set up for an agent who only plays an equally weithed
    # portfolio (baseline)
    env_eq = TradeEnv(**trade_env_args)

    # environment secured (only money)
    # this environment is set up for an agent who plays secure, keeps its money
    env_s = TradeEnv(**trade_env_args)

    # full on one stock environment
    # these environments are set up for agents who play only on one stock
    action_fu = list()
    env_fu = list()

    for i in range(no_of_assets):
        action = np.array([0] * (i + 1) + [1] + [0] * (no_of_assets - (i + 1)))
        action_fu.append(action)

        env_fu_i = TradeEnv(**trade_env_args)

        env_fu.append(env_fu_i)

    trade_envs = {
        "policy_network": env,
        "equal_weighted": env_eq,
        "only_cash": env_s,
        "full_on_one_stocks": env_fu,
        "action_fu": action_fu,
        "args": trade_env_args,
    }

    return trade_envs


def _get_train_val_test_steps(trading_period):
    # Total number of steps for pre-training in the training set
    total_steps_train = int(RATIO_TRAIN * trading_period)

    # Total number of steps for pre-training in the validation set
    total_steps_val = int(RATIO_VAL * trading_period)

    # Total number of steps for the test
    total_steps_test = trading_period - total_steps_train - total_steps_val

    step_counts = {
        "train": total_steps_train,
        "test": total_steps_test,
        "validation": total_steps_val,
    }

    return step_counts


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "-pr",
        "--plot_results",
        action="store_true",
        help="Plot aftermath analysis",
        default=False,
    )
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
        "-nb",
        "--no_of_batches",
        type=int,
        help="Choose how many batches are trained",
        default=10,
    )
    PARSER.add_argument(
        "-ne",
        "--no_of_episodes",
        type=int,
        help="Choose how many episodes are trained",
        default=2,
    )
    PARSER.add_argument(
        "-wl", "--window_length", type=int, help="Choose window length", default=10
    )
    PARSER.add_argument(
        "-tp",
        "--max_no_of_training_periods",
        type=int,
        help="Set upper limit for training periods",
        default=10000,
    )
    PARSER.add_argument(
        "-v",
        "--verbose",
        help="Print train vectors",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-t", "--test_mode", help="Quick testrun", default=False, action="store_true"
    )

    ARGS = PARSER.parse_args()

    if ARGS.verbose:
        print("\nVerbose session. Alot of vectors will be printed below.\n")

    if ARGS.test_mode:
        main(
            interactive_session=False,
            crypto_data=False,
            gpu_device=None,
            verbose=False,
            no_of_assets=2,
            max_no_of_training_periods=10000,
            plot_results=False,
            n_episodes=1,
            n_batches=1,
            window_length=400,
        )

    else:
        main(
            interactive_session=ARGS.interactive_session,
            crypto_data=ARGS.crypto_data,
            gpu_device=ARGS.gpu_device,
            verbose=ARGS.verbose,
            no_of_assets=ARGS.no_of_assets,
            max_no_of_training_periods=ARGS.max_no_of_training_periods,
            plot_results=ARGS.plot_results,
            n_episodes=ARGS.no_of_episodes,
            n_batches=ARGS.no_of_batches,
            window_length=ARGS.window_length,
        )
