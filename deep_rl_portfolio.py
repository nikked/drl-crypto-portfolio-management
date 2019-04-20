import time
import argparse
from pprint import pprint

from src.train_rl_algorithm import train_rl_algorithm
from src.test_rl_algorithm import test_rl_algorithm
from src.plot_train_results import plot_train_results
from src.environment import TradeEnv

from src.params import (
    PF_INITIAL_VALUE,
    TRADING_COST,
    INTEREST_RATE,
    WINDOW_LENGTH,
)

from data_pipelines import get_crypto_price_tensors


DEFAULT_TRADE_ENV_ARGS = {
    "window_length": WINDOW_LENGTH,
    "portfolio_value": PF_INITIAL_VALUE,
    "trading_cost": TRADING_COST,
    "interest_rate": INTEREST_RATE,
}

JIANG_BASE_PARAMS = {
    "interactive_session": False,
    "verbose": True,
    "no_of_assets": 8,
    "plot_results": False,
    "n_episodes": 2,
    "n_batches": 10,
    "window_length": 50,
    "batch_size": 50,
    "portfolio_value": 1,
    "validate_during_training": False,
    "trading_period_length": "2h",
}


def main(**train_configs):

    print("\nStarting training process with the following options:")
    pprint(train_configs)

    start_time = time.time()

    print("\n")

    # Creation of the trading environment
    trade_envs, asset_list, train_test_val_steps = _initialize_trade_envs(
        train_configs)

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


def _initialize_trade_envs(train_configs):

    dataset, asset_names = get_crypto_price_tensors.main(
        no_of_cryptos=train_configs["no_of_assets"],
        start_date=train_configs["start_date"],
        end_date=train_configs["end_date"],
        trading_period_length=train_configs["trading_period_length"],
        train_session_name=train_configs['train_session_name']
    )

    trade_env_args = DEFAULT_TRADE_ENV_ARGS
    trade_env_args["train_size"] = train_configs['ratio_train']
    trade_env_args["data"] = dataset
    trade_env_args["window_length"] = train_configs["window_length"]

    trading_periods = dataset.shape[2]
    print("Trading periods: {}".format(dataset.shape[2]))

    # Determine the step sizes of different datasets
    train_test_val_steps = _get_train_val_test_steps(
        trading_periods, train_configs)

    print("Starting training for {} assets".format(len(asset_names)))
    print(asset_names)

    train_envs = _get_train_environments(
        train_configs["no_of_assets"], trade_env_args)

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


def _get_train_val_test_steps(trading_period, train_configs):
    # Total number of steps for pre-training in the training set
    total_steps_train = int(train_configs['ratio_train'] * trading_period)

    # Total number of steps for pre-training in the validation set
    total_steps_val = int(train_configs['ratio_val'] * trading_period)

    # Total number of steps for the test
    total_steps_test = trading_period - total_steps_train - total_steps_val

    train_test_val_steps = {
        "train": total_steps_train,
        "test": total_steps_test,
        "validation": total_steps_val,
    }

    return train_test_val_steps


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
    PARSER.add_argument(
        "-jbt1",
        "--jbt1",
        help="Run model in Jiang's backtest period",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-jbt2",
        "--jbt2",
        help="Run model in Jiang's backtest period",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-jbt3",
        "--jbt3",
        help="Run model in Jiang's backtest period",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-bull",
        "--bull_run_year_2017",
        help="Sita fattii erikoissettii jota on harvoin tarjol; fatimpaa saa edelleen ettii!",
        default=False,
        action="store_true",
    )

    PARSER.add_argument(
        "-bear",
        "--bear_year_2018",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-recent",
        "--recent_yr_2019",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-long",
        "--long_run",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-micro",
        "--micro_run",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-nano",
        "--nano_run",
        default=False,
        action="store_true",
    )
    PARSER.add_argument(
        "-pico",
        "--pico_run",
        default=False,
        action="store_true",
    )

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
            end_date="20190301",
            trading_period_length="4h",
            test_mode=True,
            validate_during_training=ARGS.validate_during_training,
            train_session_name="quick_test_run"
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
            end_date="20190301",
            trading_period_length="2h",
            test_mode=True,
            validate_during_training=ARGS.validate_during_training,
            train_session_name="test_run"
        )

    elif ARGS.jbt1:
        print("\nRunning model for Jiang's back test 1 period")

        OVERRIDE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "ratio_train": 0.918,
            "ratio_val": 0,
        }

        main(
            **OVERRIDE_PARAMS,
            start_date="20150220",
            end_date="20161028",
            train_session_name="Jiang_backtest_period_1",
            gpu_device=ARGS.gpu_device,
        )
    elif ARGS.jbt2:
        print("\nRunning model for Jiang's back test 2 period")

        OVERRIDE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "ratio_train": 0.928,
            "ratio_val": 0,
        }

        main(
            **OVERRIDE_PARAMS,
            start_date="20150220",
            end_date="20170128",
            train_session_name="Jiang_backtest_period_2",
            gpu_device=ARGS.gpu_device,
        )
    elif ARGS.jbt3:
        print("\nRunning model for Jiang's back test 2 period")

        OVERRIDE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "ratio_train": 0.93,
            "ratio_val": 0,
        }

        main(
            **OVERRIDE_PARAMS,
            start_date="20150501",
            end_date="20170427",
            train_session_name="Jiang_backtest_period_3",
            gpu_device=ARGS.gpu_device,
        )
    elif ARGS.bull_run_year_2017:
        print("\nRunning Fattii Erikois Settii")

        BASE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "no_of_assets": 7,
            "start_date": "20170101",
            "end_date": "20171231",
            "train_session_name": "bull_run_year_2017",
            "gpu_device": ARGS.gpu_device,
            "ratio_train": 0.9,
            "ratio_val": 0.0,
            "trading_period_length": "2h",
        }

        main(**BASE_PARAMS)

    elif ARGS.bear_year_2018:
        print("\nRunning bear year 2018")

        BASE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "no_of_assets": 7,
            "start_date": "20180101",
            "end_date": "20181231",
            "train_session_name": "bear_year_2018",
            "gpu_device": ARGS.gpu_device,
            "ratio_train": 0.9,
            "ratio_val": 0.0,
            "trading_period_length": "2h",
        }

        main(**BASE_PARAMS)

    elif ARGS.recent_yr_2019:
        print("\nRunning recent year 2019")

        BASE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "no_of_assets": 7,
            "start_date": "20180101",
            "end_date": "20190418",
            "train_session_name": "recent_yr_2019",
            "gpu_device": ARGS.gpu_device,
            "ratio_train": 0.9,
            "ratio_val": 0.0,
            "trading_period_length": "2h",
        }

        main(**BASE_PARAMS)


    elif ARGS.long_run:

        BASE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "no_of_assets": 7,
            "start_date": "20150220",
            "end_date": "20190418",
            "train_session_name": "long_run",
            "gpu_device": ARGS.gpu_device,
            "ratio_train": 0.9,
            "ratio_val": 0.0,
            "trading_period_length": "4h",
        }

        main(**BASE_PARAMS)

    elif ARGS.micro_run:

        BASE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "no_of_assets": 7,
            "start_date": "20180101",
            "end_date": "20190418",
            "train_session_name": "micro_run",
            "gpu_device": ARGS.gpu_device,
            "ratio_train": 0.9,
            "ratio_val": 0.0,
            "trading_period_length": "30min",
        }

        main(**BASE_PARAMS)

    elif ARGS.nano_run:

        BASE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "no_of_assets": 7,
            "start_date": "20180801",
            "end_date": "20190418",
            "train_session_name": "nano_run",
            "gpu_device": ARGS.gpu_device,
            "ratio_train": 0.9,
            "ratio_val": 0.0,
            "trading_period_length": "15min",
        }

        main(**BASE_PARAMS)

    elif ARGS.pico_run:

        BASE_PARAMS = {
            **JIANG_BASE_PARAMS,
            "no_of_assets": 6,
            "start_date": "20190201",
            "end_date": "20190418",
            "train_session_name": "pico_run",
            "gpu_device": ARGS.gpu_device,
            "ratio_train": 0.9,
            "ratio_val": 0.0,
            "trading_period_length": "5min",
        }

        main(**BASE_PARAMS)

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
