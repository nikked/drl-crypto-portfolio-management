import tensorflow as tf
import numpy as np
from src.Policy import Policy
import matplotlib.pyplot as plt

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
    batch_size,
    n_episodes,
    n_batches,
    dict_hp_pb,
    w_init_test,
    pf_init_test,
    total_steps_train,
    total_steps_train,
    total_steps_val,
    list_stock,
    DEFAULT_TRADE_ENV_ARGS
)

from src.environment import TradeEnv
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
            _eval_perf("Before Training", actor, interactive_session)
        print("Episode:", e)
        # init the PVM with the training parameters

        w_init_train = np.array(np.array([1] + [0] * m))  # dict_train['w_init_train']

        memory = PVM(
            m,
            total_steps_train,
            batch_size,
            w_init_train,
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
        _eval_perf(e, actor, interactive_session)

    return actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s


# random action function


def _get_random_action(m):
    random_vec = np.random.rand(m + 1)
    return random_vec / np.sum(random_vec)


def _eval_perf(e, actor, render_plots):
    """
    This function evaluates the performance of the different types of agents.


    """
    list_weight_end_val = list()
    list_pf_end_training = list()
    list_pf_min_training = list()
    list_pf_max_training = list()
    list_pf_mean_training = list()
    list_pf_dd_training = list()

    #######TEST#######
    # environment for trading of the agent
    env_eval = TradeEnv(**DEFAULT_TRADE_ENV_ARGS)

    # initialization of the environment
    state_eval, done_eval = env_eval.reset(
        w_init_test, pf_init_test, t=total_steps_train
    )

    # first element of the weight and portfolio value
    p_list_eval = [pf_init_test]
    w_list_eval = [w_init_test]

    for k in range(total_steps_train, total_steps_train + total_steps_val - int(n / 2)):
        X_t = state_eval[0].reshape([-1] + list(state_eval[0].shape))
        W_previous = state_eval[1].reshape([-1] + list(state_eval[1].shape))
        pf_value_previous = state_eval[2]
        # compute the action
        action = actor.compute_W(X_t, W_previous)
        # step forward environment
        state_eval, reward_eval, done_eval = env_eval.step(action)

        X_next = state_eval[0]
        W_t_eval = state_eval[1]
        pf_value_t_eval = state_eval[2]

        dailyReturn_t = X_next[-1, :, -1]
        # print('current portfolio value', round(pf_value_previous,0))
        # print('weights', W_previous)
        p_list_eval.append(pf_value_t_eval)
        w_list_eval.append(W_t_eval)

    list_weight_end_val.append(w_list_eval[-1])
    list_pf_end_training.append(p_list_eval[-1])
    list_pf_min_training.append(np.min(p_list_eval))
    list_pf_max_training.append(np.max(p_list_eval))
    list_pf_mean_training.append(np.mean(p_list_eval))

    list_pf_dd_training.append(_get_max_draw_down(p_list_eval))

    print("End of test PF value:", round(p_list_eval[-1]))
    print("Min of test PF value:", round(np.min(p_list_eval)))
    print("Max of test PF value:", round(np.max(p_list_eval)))
    print("Mean of test PF value:", round(np.mean(p_list_eval)))
    print("Max Draw Down of test PF value:",
          round(_get_max_draw_down(p_list_eval)))
    print("End of test weights:", w_list_eval[-1])

    if render_plots:
        plt.title("Portfolio evolution (validation set) episode {}".format(e))
        plt.plot(p_list_eval, label="Agent Portfolio Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()
        plt.title("Portfolio weights (end of validation set) episode {}".format(e))
        plt.bar(np.arange(m + 1), list_weight_end_val[-1])
        plt.xticks(np.arange(m + 1), ["Money"] + list_stock, rotation=45)
        plt.show()

    names = ["Money"] + list_stock
    w_list_eval = np.array(w_list_eval)

    if render_plots:
        for j in range(m + 1):
            plt.plot(w_list_eval[:, j],
                     label="Weight Stock {}".format(names[j]))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
        plt.show()


def _get_max_draw_down(xs):
    xs = np.array(xs)
    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
    j = np.argmax(xs[:i])  # start of period

    return xs[j] - xs[i]
