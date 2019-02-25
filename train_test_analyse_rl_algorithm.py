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

from data_pipelines import data_pipe, data_pipe_poloniex


DEFAULT_TRADE_ENV_ARGS = {
    "path": None,
    "window_length": WINDOW_LENGTH,
    "portfolio_value": PF_INITIAL_VALUE,
    "trading_cost": TRADING_COST,
    "interest_rate": INTEREST_RATE,
    "train_size": RATIO_TRAIN,
    "data": None,
}


def main(**train_configs):

    print("\nStarting training process with the following options:")
    pprint(train_configs)

    start_time = time.time()

    # Creation of the trading environment
    print("\n")
    trade_envs, asset_list, step_counts = _initialize_trade_envs(train_configs)

    # Agent training
    actor, state_fu, done_fu, train_performance_lists = train_rl_algorithm(
        train_configs, trade_envs, asset_list, step_counts
    )

    # Agent evaluation
    test_performance_lists = test_rl_algorithm(
        train_configs, actor, state_fu, done_fu, trade_envs, step_counts
    )

    end_time = time.time()
    train_time_secs = round(end_time - start_time, 1)

    print("\nTraining completed")
    print(f"Process took {train_time_secs} seconds")

    if train_configs["plot_results"]:
        plot_training_results(
            train_configs,
            test_performance_lists,
            train_performance_lists,
            "stocks",
            asset_list,
        )


def _initialize_trade_envs(train_configs):

    if train_configs["crypto_data"]:
        dataset, asset_names = data_pipe_poloniex.main(
            no_of_cryptos=train_configs["no_of_assets"],
            max_count_of_periods=train_configs["max_no_of_training_periods"],
        )

    else:
        dataset, asset_names = data_pipe.main(
            no_of_stocks=train_configs["no_of_assets"],
            max_count_of_periods=train_configs["max_no_of_training_periods"],
        )

    trade_env_args = DEFAULT_TRADE_ENV_ARGS
    trade_env_args["data"] = dataset
    trade_env_args["window_length"] = train_configs["window_length"]

    trading_periods = dataset.shape[2]
    print("Trading periods: {}".format(dataset.shape[2]))
    step_counts = _get_train_val_test_steps(trading_periods)

    print("Starting training for {} assets".format(len(asset_names)))
    print(asset_names)

    train_envs = _get_train_environments(train_configs["no_of_assets"], trade_env_args)

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
        "-na",
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
        "-bs", "--batch_size", type=int, help="Select batch size", default=50
    )
    PARSER.add_argument(
        "-ne",
        "--no_of_episodes",
        type=int,
        help="Choose how many episodes are trained",
        default=2,
    )
    PARSER.add_argument(
        "-wl", "--window_length", type=int, help="Choose window length", default=40
    )
    PARSER.add_argument(
        "-tp",
        "--max_no_of_training_periods",
        type=int,
        help="Set upper limit for training periods",
        default=1000,
    )
    PARSER.add_argument(
        "-pv",
        "--portfolio_initial_value",
        type=int,
        help="Initial cash invested in portfolio",
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

        print("\nStarting rapid test run...")
        main(
            interactive_session=False,
            crypto_data=True,
            gpu_device=None,
            verbose=False,
            no_of_assets=2,
            max_no_of_training_periods=200,
            plot_results=False,
            n_episodes=1,
            n_batches=1,
            window_length=77,
            batch_size=1,
            portfolio_value=10000,
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
            batch_size=ARGS.batch_size,
            portfolio_value=ARGS.portfolio_initial_value,
        )
