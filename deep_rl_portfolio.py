import sys
from datetime import datetime, timedelta
import time
import argparse
from pprint import pprint

from src.train_rl_algorithm import train_rl_algorithm
from src.test_rl_algorithm import test_rl_algorithm
from src.plot_train_results import plot_train_results
from src.make_train_histograms import make_train_histograms
from src.environment import TradeEnv

from src.params import PF_INITIAL_VALUE, TRADING_COST, INTEREST_RATE, WINDOW_LENGTH

from data_pipelines import get_crypto_price_tensors


DEFAULT_TRADE_ENV_ARGS = {
    "window_length": WINDOW_LENGTH,
    "portfolio_value": PF_INITIAL_VALUE,
    "trading_cost": TRADING_COST,
}

TRAIN_BASE_PARAMS = {
    "interactive_session": False,
    "verbose": False,
    "no_of_assets": 11,
    "plot_results": False,
    "n_episodes": 3,
    "n_batches": 20,
    "window_length": 70,
    "batch_size": 50,
    "portfolio_value": 1,
    "validate_during_training": False,
    "ratio_val": 0,
    "max_pf_weight_penalty": 0.5,
}


def main(**train_configs):

    print("\nStarting training process with the following options:")
    pprint(train_configs)

    start_time = time.time()

    print("\n")

    # Creation of the trading environment
    trade_envs, asset_list, train_test_val_steps = _initialize_trade_envs(train_configs)

    # Agent training
    agent, state_fu, done_fu, train_performance_lists = train_rl_algorithm(
        train_configs, trade_envs, asset_list, train_test_val_steps
    )

    # Agent evaluation
    test_performance_lists = test_rl_algorithm(
        train_configs, agent, state_fu, done_fu, trade_envs, train_test_val_steps
    )

    end_time = time.time()
    train_time_secs = round(end_time - start_time, 1)

    print("\nTraining completed")
    print(f"Process took {train_time_secs} seconds")

    plot_train_results(
        train_configs,
        train_performance_lists,
        test_performance_lists,
        asset_list,
        train_time_secs,
        train_test_val_steps,
    )

    print("\nAggregating simulation statistics...")
    make_train_histograms(train_configs["train_session_name"])

    pprint("Exiting")


def _initialize_trade_envs(train_configs):

    dataset, asset_names, ratio_train = get_crypto_price_tensors.main(
        no_of_cryptos=train_configs["no_of_assets"],
        start_date=train_configs["start_date"],
        test_start_date=train_configs["test_start_date"],
        end_date=train_configs["end_date"],
        trading_period_length=train_configs["trading_period_length"],
        train_session_name=train_configs["train_session_name"],
    )

    trade_env_args = DEFAULT_TRADE_ENV_ARGS
    trade_env_args["train_size"] = ratio_train
    trade_env_args["data"] = dataset
    trade_env_args["window_length"] = train_configs["window_length"]

    trading_periods = dataset.shape[2]
    print("Trading periods: {}".format(dataset.shape[2]))

    # Determine the step sizes of different datasets
    train_test_val_steps = _get_train_val_test_steps(
        trading_periods, train_configs, ratio_train
    )

    print("Starting training for {} assets".format(len(asset_names)))
    print(asset_names)

    train_envs = _get_train_environments(train_configs["no_of_assets"], trade_env_args)

    return train_envs, asset_names, train_test_val_steps


def _get_train_environments(no_of_assets, trade_env_args):

    # environment for trading of the agent
    # this is the agent trading environment (policy network agent)
    env = TradeEnv(**trade_env_args)

    # environment for trading of the agent
    # that just does the first cash distribution and then holds eq
    env_first_action = TradeEnv(**trade_env_args)

    # environment for equally weighted
    # this environment is set up for an agent who only plays an equally weithed
    # portfolio (baseline)
    env_eq = TradeEnv(**trade_env_args)

    # environment secured (only money)
    # this environment is set up for an agent who plays secure, keeps its money
    env_s = TradeEnv(**trade_env_args)

    # full on one stock environment
    # these environments are set up for agents who play only on one stock
    env_fu = [TradeEnv(**trade_env_args) for asset in range(no_of_assets)]

    trade_envs = {
        "policy_network": env,
        "policy_network_first_step_only": env_first_action,
        "equal_weighted": env_eq,
        "only_cash": env_s,
        "full_on_one_stocks": env_fu,
        "args": trade_env_args,
    }

    return trade_envs


def _get_train_val_test_steps(trading_period, train_configs, ratio_train):

    # Total number of steps for pre-training in the training set
    total_steps_train = int(ratio_train * trading_period) + 1

    # Total number of steps for pre-training in the validation set
    total_steps_val = int(train_configs["ratio_val"] * trading_period)

    # Total number of steps for the test
    total_steps_test = trading_period - total_steps_train - total_steps_val

    train_test_val_steps = {
        "train": total_steps_train,
        "test": total_steps_test,
        "validation": total_steps_val,
    }

    test_start_idx = train_test_val_steps["train"] + train_test_val_steps["validation"]

    print(
        f"Test will start from idx: {train_test_val_steps['train'] + train_test_val_steps['validation']}"
    )

    return train_test_val_steps


def _calculate_start_date(end_date, trading_period_length):

    if trading_period_length in ["2h", "4h", "1d", "30min"]:
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(599)).strftime(
            "%Y%m%d"
        )

    elif trading_period_length == "15min":
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(299)).strftime(
            "%Y%m%d"
        )

    elif trading_period_length == "5min":
        start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(99)).strftime(
            "%Y%m%d"
        )

    return start_date


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
        "-rt",
        "--ratio_train",
        type=float,
        help="Proportional size of train set",
        default=0.95,
    )
    PARSER.add_argument(
        "-rv",
        "--ratio_val",
        type=float,
        help="Proportional size of val set",
        default=0.0,
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
        "-pv",
        "--portfolio_initial_value",
        type=int,
        help="Initial cash invested in portfolio",
        default=100,
    )
    PARSER.add_argument(
        "-v",
        "--verbose",
        help="Print train vectors",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-t", "--test_mode", help="Proper testrun", default=False, action="store_true"
    )
    PARSER.add_argument(
        "-qt",
        "--quick_test_mode",
        help="Quick testrun",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-vdt",
        "--validate_during_training",
        help="Run additional validation during training to monitor overfitting",
        default=False,
        action="store_true",
    )
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
        "-pl",
        "--trading_period_length",
        type=str,
        default="1d",
        help="Trade period length (5min, 15min, 30min, 2h, 4h, 1d)",
    )
    PARSER.add_argument("-demo", "--demo", default=False, action="store_true")
    PARSER.add_argument(
        "-cbts", "--calm_before_the_storm", default=False, action="store_true"
    )
    PARSER.add_argument("-awake", "--awakening", default=False, action="store_true")
    PARSER.add_argument("-xrp", "--ripple_bull_run", default=False, action="store_true")
    PARSER.add_argument("-eth", "--ethereum_valley", default=False, action="store_true")
    PARSER.add_argument("-ath", "--all_time_high", default=False, action="store_true")

    PARSER.add_argument("-rock", "--rock_bottom", default=False, action="store_true")
    PARSER.add_argument("-recent", "--recent", default=False, action="store_true")
    PARSER.add_argument("-nano", "--nano_run", default=False, action="store_true")
    PARSER.add_argument("-pico", "--pico_run", default=False, action="store_true")

    ARGS = PARSER.parse_args()

    if ARGS.verbose:
        print("\nVerbose session. Alot of vectors will be printed below.\n")

    if ARGS.quick_test_mode:
        print("\nStarting rapid test run...")
        main(
            interactive_session=False,
            gpu_device=None,
            verbose=True,
            no_of_assets=5,
            plot_results=False,
            n_episodes=1,
            n_batches=1,
            ratio_train=ARGS.ratio_train,
            ratio_val=ARGS.ratio_val,
            window_length=130,
            batch_size=1,
            portfolio_value=100,
            start_date="20190101",
            test_start_date="20190201",
            end_date="20190301",
            trading_period_length="4h",
            max_pf_weight_penalty=0.7,
            test_mode=True,
            validate_during_training=ARGS.validate_during_training,
            train_session_name="quick_test_run_with_long_name",
        )

    elif ARGS.test_mode:
        print("\nStarting proper test run...")
        main(
            interactive_session=False,
            gpu_device=None,
            verbose=True,
            no_of_assets=7,
            plot_results=False,
            ratio_train=ARGS.ratio_train,
            ratio_val=ARGS.ratio_val,
            n_episodes=1,
            n_batches=1,
            window_length=77,
            batch_size=1,
            portfolio_value=100,
            start_date="20190101",
            test_start_date="20190201",
            end_date="20190301",
            trading_period_length="2h",
            max_pf_weight_penalty=0.7,
            test_mode=True,
            validate_during_training=ARGS.validate_during_training,
            train_session_name="test_run_with_long_name",
        )

    elif ARGS.demo:
        print("\nRunning model: Ripple bull run")

        end_date = "20170427"
        test_start_date = "20170307"
        start_date = "20170101"

        demo_params = {
            **TRAIN_BASE_PARAMS,
            "no_of_assets": 3,
            "n_batches": 10,
            "n_episodes": 1,
        }

        main(
            **demo_params,
            start_date=start_date,
            end_date=end_date,
            test_start_date=test_start_date,
            trading_period_length="4h",
            train_session_name="Dynamic_agent_demo",
            gpu_device=ARGS.gpu_device,
        )

    elif ARGS.calm_before_the_storm:
        print("\nRunning model: Calm_before_the_storm")

        end_date = "20161028"
        start_date = _calculate_start_date(end_date, ARGS.trading_period_length)
        main(
            **TRAIN_BASE_PARAMS,
            start_date=start_date,
            test_start_date="20160907",
            end_date=end_date,
            trading_period_length=ARGS.trading_period_length,
            train_session_name="Calm_before_the_storm_{}".format(
                ARGS.trading_period_length
            ),
            gpu_device=ARGS.gpu_device,
        )
    elif ARGS.awakening:
        print("\nRunning model: Awakening")

        end_date = "20170128"
        start_date = _calculate_start_date(end_date, ARGS.trading_period_length)
        main(
            **TRAIN_BASE_PARAMS,
            start_date=start_date,
            end_date=end_date,
            test_start_date="20161208",
            trading_period_length=ARGS.trading_period_length,
            train_session_name="Awakening_{}".format(ARGS.trading_period_length),
            gpu_device=ARGS.gpu_device,
        )
    elif ARGS.ripple_bull_run:
        print("\nRunning model: Ripple bull run")

        end_date = "20170427"
        start_date = _calculate_start_date(end_date, ARGS.trading_period_length)
        main(
            **TRAIN_BASE_PARAMS,
            start_date=start_date,
            end_date=end_date,
            test_start_date="20170307",
            trading_period_length=ARGS.trading_period_length,
            train_session_name="Ripple_bull_run_{}".format(ARGS.trading_period_length),
            gpu_device=ARGS.gpu_device,
        )
    elif ARGS.ethereum_valley:
        print("\nRunning model ethereum_valley")

        end_date = "20170718"
        start_date = _calculate_start_date(end_date, ARGS.trading_period_length)
        main(
            **TRAIN_BASE_PARAMS,
            start_date=start_date,
            end_date=end_date,
            test_start_date="20170528",
            trading_period_length=ARGS.trading_period_length,
            train_session_name="Ethereum_valley_{}".format(ARGS.trading_period_length),
            gpu_device=ARGS.gpu_device,
        )
    elif ARGS.all_time_high:
        print("\nRunning All time high")

        end_date = "20180113"

        start_date = _calculate_start_date(end_date, ARGS.trading_period_length)

        train_params = {**TRAIN_BASE_PARAMS, "max_pf_weight_penalty": 1.1}
        main(
            **train_params,
            start_date=start_date,
            end_date=end_date,
            test_start_date="20171123",
            trading_period_length=ARGS.trading_period_length,
            train_session_name="All-time_high_{}".format(ARGS.trading_period_length),
            gpu_device=ARGS.gpu_device,
        )

    elif ARGS.rock_bottom:
        print("\nRunning Rock Bottom")

        end_date = "20181231"

        start_date = _calculate_start_date(end_date, ARGS.trading_period_length)
        train_params = {**TRAIN_BASE_PARAMS, "max_pf_weight_penalty": 0.9}
        main(
            **train_params,
            start_date=start_date,
            end_date=end_date,
            test_start_date="20181110",
            trading_period_length=ARGS.trading_period_length,
            train_session_name="Rock_bottom_{}".format(ARGS.trading_period_length),
            gpu_device=ARGS.gpu_device,
        )

    elif ARGS.recent:
        print("\nRunning recent year 2019")

        end_date = "20190426"

        start_date = _calculate_start_date(end_date, ARGS.trading_period_length)
        main(
            **TRAIN_BASE_PARAMS,
            start_date=start_date,
            end_date=end_date,
            test_start_date="20190306",
            trading_period_length=ARGS.trading_period_length,
            train_session_name="Recent_{}".format(ARGS.trading_period_length),
            gpu_device=ARGS.gpu_device,
        )

    else:
        main(
            interactive_session=ARGS.interactive_session,
            gpu_device=ARGS.gpu_device,
            verbose=ARGS.verbose,
            no_of_assets=ARGS.no_of_assets,
            plot_results=ARGS.plot_results,
            n_episodes=ARGS.no_of_episodes,
            ratio_train=ARGS.ratio_train,
            ratio_val=ARGS.ratio_val,
            n_batches=ARGS.no_of_batches,
            window_length=ARGS.window_length,
            batch_size=ARGS.batch_size,
            portfolio_value=ARGS.portfolio_initial_value,
            start_date=ARGS.start_date,
            end_date=ARGS.end_date,
            trading_period_length=ARGS.trading_period_length,
            validate_during_training=ARGS.validate_during_training,
        )
