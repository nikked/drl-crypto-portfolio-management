import tensorflow as tf
import numpy as np

from src.params import (
    trading_cost,
    interest_rate,
    n_filter_1,
    n_filter_2,
    nb_feature_map,
    kernel1_size,
    cash_bias_init,
    ratio_regul,
    LEARNING_RATE
)

OPTIMIZER = tf.train.AdamOptimizer(LEARNING_RATE)


# define neural net \pi_\phi(s) as a class


class Policy(object):
    """
    This class is used to instanciate the policy network agent

    """

    def __init__(
        self,
        m,
        n,
        sess,
        w_eq,
        trading_cost=trading_cost,
        interest_rate=interest_rate,
        n_filter_1=n_filter_1,
        n_filter_2=n_filter_2,
    ):

        # parameters
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate
        self.n_filter_1 = n_filter_1
        self.n_filter_2 = n_filter_2
        self.n = n
        self.m = m

        with tf.variable_scope("Inputs"):

            # Placeholder

            # tensor of the prices
            self.X_t = tf.placeholder(
                tf.float32, [None, nb_feature_map, self.m, self.n]
            )  # The Price tensor
            # weights at the previous time step
            self.W_previous = tf.placeholder(tf.float32, [None, self.m + 1])
            # portfolio value at the previous time step
            self.pf_value_previous = tf.placeholder(tf.float32, [None, 1])
            # vector of Open(t+1)/Open(t)
            self.dailyReturn_t = tf.placeholder(tf.float32, [None, self.m])

            # self.pf_value_previous_eq = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope("Policy_Model"):

            # variable of the cash bias
            bias = tf.get_variable(
                "cash_bias",
                shape=[1, 1, 1, 1],
                initializer=tf.constant_initializer(cash_bias_init),
            )
            # shape of the tensor == batchsize
            shape_X_t = tf.shape(self.X_t)[0]
            # trick to get a "tensor size" for the cash bias
            self.cash_bias = tf.tile(bias, tf.stack([shape_X_t, 1, 1, 1]))
            # print(self.cash_bias.shape)

            with tf.variable_scope("Conv1"):
                # first layer on the X_t tensor
                # return a tensor of depth 2
                self.conv1 = tf.layers.conv2d(
                    inputs=tf.transpose(self.X_t, perm=[0, 3, 2, 1]),
                    activation=tf.nn.relu,
                    filters=self.n_filter_1,
                    strides=(1, 1),
                    kernel_size=kernel1_size,
                    padding="same",
                )

            with tf.variable_scope("Conv2"):

                # feature maps
                self.conv2 = tf.layers.conv2d(
                    inputs=self.conv1,
                    activation=tf.nn.relu,
                    filters=self.n_filter_2,
                    strides=(self.n, 1),
                    kernel_size=(1, self.n),
                    padding="same",
                )

            with tf.variable_scope("Tensor3"):
                # w from last periods
                # trick to have good dimensions
                w_wo_c = self.W_previous[:, 1:]
                w_wo_c = tf.expand_dims(w_wo_c, 1)
                w_wo_c = tf.expand_dims(w_wo_c, -1)
                self.tensor3 = tf.concat([self.conv2, w_wo_c], axis=3)

            with tf.variable_scope("Conv3"):
                # last feature map WITHOUT cash bias
                self.conv3 = tf.layers.conv2d(
                    inputs=self.conv2,
                    activation=tf.nn.relu,
                    filters=1,
                    strides=(self.n_filter_2 + 1, 1),
                    kernel_size=(1, 1),
                    padding="same",
                )

            with tf.variable_scope("Tensor4"):
                # last feature map WITH cash bias
                self.tensor4 = tf.concat([self.cash_bias, self.conv3], axis=2)
                # we squeeze to reduce and get the good dimension
                self.squeezed_tensor4 = tf.squeeze(self.tensor4, [1, 3])

            with tf.variable_scope("Policy_Output"):
                # softmax layer to obtain weights
                self.action = tf.nn.softmax(self.squeezed_tensor4)

            with tf.variable_scope("Reward"):
                # computation of the reward
                # please look at the chronological map to understand
                constant_return = tf.constant(
                    1 + self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, self.dailyReturn_t], axis=1)
                Vprime_t = self.action * self.pf_value_previous
                Vprevious = self.W_previous * self.pf_value_previous

                # this is just a trick to get the good shape for cost
                constant = tf.constant(1.0, shape=[1])

                cost = (
                    self.trading_cost
                    * tf.norm(Vprime_t - Vprevious, ord=1, axis=1)
                    * constant
                )

                cost = tf.expand_dims(cost, 1)

                zero = tf.constant(
                    np.array([0.0] * m).reshape(1, m), shape=[1, m], dtype=tf.float32
                )

                vec_zero = tf.tile(zero, tf.stack([shape_X_t, 1]))
                vec_cost = tf.concat([cost, vec_zero], axis=1)

                Vsecond_t = Vprime_t - vec_cost

                V_t = tf.multiply(Vsecond_t, y_t)
                self.portfolioValue = tf.norm(V_t, ord=1)
                self.instantaneous_reward = (
                    self.portfolioValue - self.pf_value_previous
                ) / self.pf_value_previous

            with tf.variable_scope("Reward_Equiweighted"):
                constant_return = tf.constant(
                    1 + self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat([cash_return, self.dailyReturn_t], axis=1)

                V_eq = w_eq * self.pf_value_previous
                V_eq_second = tf.multiply(V_eq, y_t)

                self.portfolioValue_eq = tf.norm(V_eq_second, ord=1)

                self.instantaneous_reward_eq = (
                    self.portfolioValue_eq - self.pf_value_previous
                ) / self.pf_value_previous

            with tf.variable_scope("Max_weight"):
                self.max_weight = tf.reduce_max(self.action)
                print(self.max_weight.shape)

            with tf.variable_scope("Reward_adjusted"):

                self.adjested_reward = (
                    self.instantaneous_reward
                    - self.instantaneous_reward_eq
                    - ratio_regul * self.max_weight
                )

        # objective function
        # maximize reward over the batch
        # min(-r) = max(r)
        self.train_op = OPTIMIZER.minimize(-self.adjested_reward)

        # some bookkeeping
        self.optimizer = OPTIMIZER
        self.sess = sess

    def compute_W(self, X_t_, W_previous_):
        """
        This function returns the action the agent takes 
        given the input tensor and the W_previous

        It is a vector of weight

        """

        return self.sess.run(
            tf.squeeze(self.action),
            feed_dict={self.X_t: X_t_, self.W_previous: W_previous_},
        )

    def train(self, X_t_, W_previous_, pf_value_previous_, dailyReturn_t_):
        """
        This function trains the neural network
        maximizing the reward 
        the input is a batch of the differents values
        """
        self.sess.run(
            self.train_op,
            feed_dict={
                self.X_t: X_t_,
                self.W_previous: W_previous_,
                self.pf_value_previous: pf_value_previous_,
                self.dailyReturn_t: dailyReturn_t_,
            },
        )
