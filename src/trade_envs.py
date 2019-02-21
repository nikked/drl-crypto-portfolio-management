import numpy as np

from environment import TradeEnv
from src.params import (
    path_data,
    n,
    pf_init_train,
    trading_cost,
    interest_rate,
    dict_hp_pb,
    m,
)

# environment for trading of the agent
# this is the agent trading environment (policy network agent)
env = TradeEnv(
    path=path_data,
    window_length=n,
    portfolio_value=pf_init_train,
    trading_cost=trading_cost,
    interest_rate=interest_rate,
    train_size=dict_hp_pb["ratio_train"],
)


# environment for equiweighted
# this environment is set up for an agent who only plays an equiweithed
# portfolio (baseline)
env_eq = TradeEnv(
    path=path_data,
    window_length=n,
    portfolio_value=pf_init_train,
    trading_cost=trading_cost,
    interest_rate=interest_rate,
    train_size=dict_hp_pb["ratio_train"],
)

# environment secured (only money)
# this environment is set up for an agentwho plays secure, keeps its money
env_s = TradeEnv(
    path=path_data,
    window_length=n,
    portfolio_value=pf_init_train,
    trading_cost=trading_cost,
    interest_rate=interest_rate,
    train_size=dict_hp_pb["ratio_train"],
)


# full on one stock environment
# these environments are set up for agents who play only on one stock

action_fu = list()
env_fu = list()


for i in range(m):
    action = np.array([0] * (i + 1) + [1] + [0] * (m - (i + 1)))
    action_fu.append(action)

    env_fu_i = TradeEnv(
        path=path_data,
        window_length=n,
        portfolio_value=pf_init_train,
        trading_cost=trading_cost,
        interest_rate=interest_rate,
        train_size=dict_hp_pb["ratio_train"],
    )

    env_fu.append(env_fu_i)
