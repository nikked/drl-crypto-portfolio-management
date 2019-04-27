PERIOD_LENGTHS = {
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}

# Add to CLI
EPSILON_GREEDY_THRESHOLD = 0.8
MAX_PF_WEIGHT_PENALTY = 0.5


# Dict hp net
# POLICY PARAMS
N_FILTER_1 = 5
N_FILTER_2 = 50
KERNEL1_SIZE = (1, 3)


# Parameter alpha (i.e. the step size) of the Adam optimization
LEARNING_RATE = 9e-2

# Finance parameters
TRADING_COST = 0.2 / 100
INTEREST_RATE = 0.00 / 250
CASH_BIAS_INIT = 0.7


# Only imported at CLI

# Training Parameters
PF_INITIAL_VALUE = 10000

# Number of the columns (number of the trading periods) in each input
# price matrix
WINDOW_LENGTH = 10

# not imported
# N_EPISODES = 2

# N_BATCHES = 10
# # Size of mini-batch during training
# BATCH_SIZE = 50

# # The L2 regularization coefficient applied to network training
# REGULARIZATION = 1e-8  # not used ATM
