from gym.utils import seeding
import numpy as np


class TradingEnvironment:  # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
        self,
        window_length=50,
        portfolio_value=10000,
        trading_cost=0.25 / 100,
        interest_rate=0.02 / 250,
        train_size=0.7,
        data=None,
    ):

        self.data = data

        self.portfolio_value = portfolio_value
        self.window_length = window_length
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate

        self.nb_cryptos = self.data.shape[1]
        self.end_train = int((self.data.shape[2] - self.window_length) * train_size)

        self.index = None
        self.state = None
        self.done = False

        self.initialize_environment()

    def initialize_environment(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_crypto_prices(self, x_prices, window_length):
        return x_prices[:, :, window_length - self.window_length : window_length]

    def get_crypto_returns(self):
        return np.array([1 + self.interest_rate] + self.data[-1, :, self.index].tolist())

    def reset_environment(self, w_init, p_init, index=0):
        self.state = (self.get_crypto_prices(self.data, self.window_length), w_init, p_init)
        self.index = index
        self.done = False

        return self.state, self.done

    def step(self, weights_before_step, adjust_portfolio=True):
        old_weights = self.state[1]
        old_ptf_value = self.state[2]

        if not adjust_portfolio:
            weights_before_step = old_weights

        cost = (
            old_ptf_value * np.linalg.norm((weights_before_step - old_weights), ord=1) * self.trading_cost
        )

        value_after_tx_costs = old_ptf_value * weights_before_step - np.array([cost] + [0] * self.nb_cryptos)

        new_crypto_values = value_after_tx_costs * self.get_crypto_returns()

        new_ptf_value = np.sum(new_crypto_values)

        new_weights = new_crypto_values / new_ptf_value

        step_reward = (new_ptf_value - old_ptf_value) / old_ptf_value

        new_index = self.index + 1

        self.state = (self.get_crypto_prices(self.data, new_index), new_weights, new_ptf_value)

        if new_index >= self.end_train:
            self.done = True

        self.index = new_index

        return self.state, step_reward, self.done
