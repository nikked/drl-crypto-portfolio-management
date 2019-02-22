import tensorflow as tf
import numpy as np
from src.Policy import Policy

from src.params import (
    m,
    n,
    optimizer,
    trading_cost,
    interest_rate,
    w_eq,
    w_s,
    pf_init_train,
    ratio_greedy,
    sample_bias,
    batch_size,
    w_init_train,
    n_episodes,
    n_batches,
    total_steps_train,
    DEFAULT_TRADE_ENV_ARGS
)

from src.eval_perf import eval_perf
from environment import TradeEnv
from src.PVM import PVM


def train_rl_algorithm(interactive_session: bool,
                       env, env_eq, env_s, action_fu, env_fu):

    ############# TRAINING #####################
    ###########################################
    tf.reset_default_graph()

    # sess
    sess = tf.Session()

    # initialize networks
    actor = Policy(
        m, n, sess, optimizer, trading_cost=trading_cost, interest_rate=interest_rate
    )  # policy initialization

    # initialize tensorflow graphs
    sess.run(tf.global_variables_initializer())

    list_final_pf = list()
    list_final_pf_eq = list()
    list_final_pf_s = list()

    list_final_pf_fu = list()
    state_fu = [0] * m
    done_fu = [0] * m

    pf_value_t_fu = [0] * m

    for i in range(m):
        list_final_pf_fu.append(list())

    ###### Train #####
    for e in range(n_episodes):
        print("Start Episode", e)
        if e == 0:
            eval_perf("Before Training", actor, interactive_session)
        print("Episode:", e)
        # init the PVM with the training parameters
        memory = PVM(
            m,
            sample_bias,
            total_steps=total_steps_train,
            batch_size=batch_size,
            w_init=w_init_train,
        )

        for nb in range(n_batches):
            # draw the starting point of the batch
            i_start = memory.draw()

            # reset the environment with the weight from PVM at the starting point
            # reset also with a portfolio value with initial portfolio value
            state, done = env.reset(memory.get_W(
                i_start), pf_init_train, t=i_start)
            state_eq, done_eq = env_eq.reset(w_eq, pf_init_train, t=i_start)
            state_s, done_s = env_s.reset(w_s, pf_init_train, t=i_start)

            for i in range(m):
                state_fu[i], done_fu[i] = env_fu[i].reset(
                    action_fu[i], pf_init_train, t=i_start
                )

            list_X_t, list_W_previous, list_pf_value_previous, list_dailyReturn_t = (
                [],
                [],
                [],
                [],
            )
            list_pf_value_previous_eq, list_pf_value_previous_s = [], []
            list_pf_value_previous_fu = list()
            for i in range(m):
                list_pf_value_previous_fu.append(list())

            for bs in range(batch_size):

                # load the different inputs from the previous loaded state
                X_t = state[0].reshape([-1] + list(state[0].shape))
                W_previous = state[1].reshape([-1] + list(state[1].shape))
                pf_value_previous = state[2]

                if np.random.rand() < ratio_greedy:
                    # print('go')
                    # computation of the action of the agent
                    action = actor.compute_W(X_t, W_previous)
                else:
                    action = _get_random_action(m)

                # given the state and the action, call the environment to go one
                # time step later
                state, reward, done = env.step(action)
                state_eq, reward_eq, done_eq = env_eq.step(w_eq)
                state_s, reward_s, done_s = env_s.step(w_s)

                for i in range(m):
                    state_fu[i], _, done_fu[i] = env_fu[i].step(action_fu[i])

                # get the new state
                X_next = state[0]
                W_t = state[1]
                pf_value_t = state[2]

                pf_value_t_eq = state_eq[2]
                pf_value_t_s = state_s[2]

                for i in range(m):
                    pf_value_t_fu[i] = state_fu[i][2]

                # let us compute the returns
                dailyReturn_t = X_next[-1, :, -1]
                # update into the PVM
                memory.update(i_start + bs, W_t)
                # store elements
                list_X_t.append(X_t.reshape(state[0].shape))
                list_W_previous.append(W_previous.reshape(state[1].shape))
                list_pf_value_previous.append([pf_value_previous])
                list_dailyReturn_t.append(dailyReturn_t)

                list_pf_value_previous_eq.append(pf_value_t_eq)
                list_pf_value_previous_s.append(pf_value_t_s)

                for i in range(m):
                    list_pf_value_previous_fu[i].append(pf_value_t_fu[i])

                if bs == batch_size - 1:
                    list_final_pf.append(pf_value_t)
                    list_final_pf_eq.append(pf_value_t_eq)
                    list_final_pf_s.append(pf_value_t_s)
                    for i in range(m):
                        list_final_pf_fu[i].append(pf_value_t_fu[i])

            #             #printing
            #             if bs==0:
            #                 print('start', i_start)
            #                 print('PF_start', round(pf_value_previous,0))

            #             if bs==batch_size-1:
            #                 print('PF_end', round(pf_value_t,0))
            #                 print('weight', W_t)

            list_X_t = np.array(list_X_t)
            list_W_previous = np.array(list_W_previous)
            list_pf_value_previous = np.array(list_pf_value_previous)
            list_dailyReturn_t = np.array(list_dailyReturn_t)

            # for each batch, train the network to maximize the reward
            actor.train(
                list_X_t, list_W_previous, list_pf_value_previous, list_dailyReturn_t
            )
        eval_perf(e, actor, interactive_session)

    return actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s


# random action function


def _get_random_action(m):
    random_vec = np.random.rand(m + 1)
    return random_vec / np.sum(random_vec)
