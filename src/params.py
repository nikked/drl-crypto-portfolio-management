# can be changed following the type of stocks studied


RATIO_TRAIN = 0.6
RATIO_VAL = 0.2


# Size of mini-batch during training
BATCH_SIZE = 50


# Parameter alpha (i.e. the step size) of the Adam optimization
LEARNING_RATE = 9e-2


PF_INIT_TEST = 10000

# Dict hp net
N_FILTER_1 = 2
N_FILTER_2 = 20
KERNEL1_SIZE = (1, 3)

namesBio = ["JNJ", "PFE", "AMGN", "MDT", "CELG", "LLY"]
namesUtilities = ["XOM", "CVX", "MRK", "SLB", "MMM"]
namesTech = ["FB", "AMZN", "MSFT", "AAPL",
             "T", "VZ", "CMCSA", "IBM", "CRM", "INTC"]
namesCrypto = [
    "ETCBTC",
    "ETHBTC",
    "DOGEBTC",
    "ETHUSDT",
    "BTCUSDT",
    "XRPBTC",
    "DASHBTC",
    "XMRBTC",
    "LTCBTC",
    "ETCETH",
]

# Dicts of the problem
dict_hp_net = {"n_filter_1": 2, "n_filter_2": 20, "kernel1_size": (1, 3)}
dict_hp_pb = {
    "batch_size": BATCH_SIZE,
    "ratio_train": RATIO_TRAIN,
    "ratio_val": RATIO_VAL,
    "length_tensor": 10,
    "ratio_greedy": 0.8,
    "ratio_regul": 0.1,
}
dict_hp_opt = {"regularization": 1e-8, "learning": 9e-2}
dict_fin = {
    "trading_cost": 0.25 / 100,
    "interest_rate": 0.02 / 250,
    "cash_bias_init": 0.7,
}
dict_train = {
    "pf_init_train": 10000,
    "w_init_train": "d",
    "n_episodes": 2,
    "n_batches": 10,
}
dict_test = {"pf_init_test": 10000, "w_init_test": "d"}


# HP of the network

kernel1_size = (1, 3)

# Number of the columns (number of the trading periods) in each input
# price matrix
LENGTH_TENSOR = 10

ratio_greedy = dict_hp_pb["ratio_greedy"]

ratio_regul = dict_hp_pb["ratio_regul"]

# HP of the optimization


# The L2 regularization coefficient applied to network training
regularization = dict_hp_opt["regularization"]


# Finance parameters

trading_cost = dict_fin["trading_cost"]
interest_rate = dict_fin["interest_rate"]
cash_bias_init = dict_fin["cash_bias_init"]

# Training Parameters

pf_init_train = dict_train["pf_init_train"]

n_episodes = dict_train["n_episodes"]
n_batches = dict_train["n_batches"]
