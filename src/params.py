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

# Finance parameters
TRADING_COST = 0.25 / 100
INTEREST_RATE = 0.02 / 250
CASH_BIAS_INIT = 0.7

# HP of the network

# Number of the columns (number of the trading periods) in each input
# price matrix
WINDOW_LENGTH = 10

RATIO_GREEDY = 0.8

RATIO_REGUL = 0.1

# The L2 regularization coefficient applied to network training
REGULARIZATION = 1e-8  # not used ATM

# Training Parameters
PF_INITIAL_VALUE = 10000
N_EPISODES = 2
N_BATCHES = 10


# Only impor
