import numpy as np
import tensorflow as tf
from src.environment import TradeEnv


# dataset

# can be changed following the type of stocks studied

# PATH_DATA = './np_data/inputCrypto.npy'
PATH_DATA = "./np_data/input.npy"


RATIO_TRAIN = 0.6
RATIO_VAL = 0.2
BATCH_SIZE = 50

# Size of mini-batch during training
batch_size = 50


# Parameter alpha (i.e. the step size) of the Adam optimization
LEARNING_RATE = 9e-2

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

# determine the length of the data, #features, #stocks
DATA_SOURCE = np.load(PATH_DATA)
trading_period = DATA_SOURCE.shape[2]
nb_feature_map = DATA_SOURCE.shape[0]
nb_stocks = DATA_SOURCE.shape[1]


data_type = PATH_DATA.split("/")[2][5:].split(".")[0]


# HP of the problem
# Total number of steps for pre-training in the training set
total_steps_train = int(RATIO_TRAIN * trading_period)

# Total number of steps for pre-training in the validation set
total_steps_val = int(RATIO_VAL * trading_period)

# Total number of steps for the test
total_steps_test = trading_period - total_steps_train - total_steps_val


# fix parameters of the network
if data_type == "Utilities":
    list_stock = namesUtilities
elif data_type == "Bio":
    list_stock = namesBio
elif data_type == "Tech":
    list_stock = namesTech
elif data_type == "Crypto":
    list_stock = namesCrypto
else:
    list_stock = [i for i in range(nb_stocks)]

# Test Parameters

# dict_test['w_init_test']
w_init_test = np.array(np.array([1] + [0] * nb_stocks))


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

pf_init_test = dict_test["pf_init_test"]


# HP of the network
n_filter_1 = dict_hp_net["n_filter_1"]
n_filter_2 = dict_hp_net["n_filter_2"]
kernel1_size = dict_hp_net["kernel1_size"]

# Number of the columns (number of the trading periods) in each input
# price matrix
n = dict_hp_pb["length_tensor"]

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
