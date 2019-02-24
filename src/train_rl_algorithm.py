import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.params import (  # pylint: disable=ungrouped-imports
    LENGTH_TENSOR,
    TRADING_COST,
    INTEREST_RATE,
    PF_INITIAL_VALUE,
    RATIO_GREEDY,
    BATCH_SIZE,
    N_EPISODES,
    N_BATCHES,
    PF_INIT_TEST,
)

from src.policy import Policy
from src.environment import TradeEnv
from src.pvm import PVM


def train_rl_algorithm(  # pylint: disable= too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    interactive_session: bool,
    env,
    env_eq,
    env_s,
    action_fu,
    env_fu,
    trade_env_args,
    weights_equal,
    weights_single,
    asset_list,
    total_steps_train,
    total_steps_val,
    nb_feature_map,
    nb_stocks,
    gpu_device,
    print_verbose,
):

    ############# TRAINING #####################
    ###########################################
    tf.reset_default_graph()

    # sess
    sess = tf.Session()

    # initialize networks
    actor = Policy(
        nb_stocks,
        LENGTH_TENSOR,
        sess,
        weights_equal,
        nb_feature_map,
        trading_cost=TRADING_COST,
        interest_rate=INTEREST_RATE,
        gpu_device=gpu_device,
    )  # policy initialization

    # initialize tensorflow graphs
    sess.run(tf.global_variables_initializer())

    list_final_pf = list()
    list_final_pf_eq = list()
    list_final_pf_s = list()

    list_final_pf_fu = list()
    state_fu = [0] * nb_stocks
    done_fu = [0] * nb_stocks

    pf_value_t_fu = [0] * nb_stocks

    for i in range(nb_stocks):
        list_final_pf_fu.append(list())

    ###### Train #####
    for no_episode in range(N_EPISODES):  # pylint: disable= too-many-nested-blocks
        print("Start Episode", no_episode)
        if no_episode == 0:
            _eval_perf(
                "Before Training",
                actor,
                interactive_session,
                trade_env_args,
                asset_list,
                total_steps_train,
                total_steps_val,
                nb_stocks,
            )
        print("Episode:", no_episode)
        # init the PVM with the training parameters

        # dict_train['w_init_train']
        w_init_train = np.array(np.array([1] + [0] * nb_stocks))

        memory = PVM(total_steps_train, BATCH_SIZE, w_init_train)

        for _ in range(N_BATCHES):
            # draw the starting point of the batch
            i_start = memory.draw()

            # reset the environment with the weight from PVM at the starting point
            # reset also with a portfolio value with initial portfolio value
            state, _ = env.reset(memory.get_w(i_start), PF_INITIAL_VALUE, index=i_start)
            state_eq, _ = env_eq.reset(weights_equal, PF_INITIAL_VALUE, index=i_start)
            state_s, _ = env_s.reset(weights_single, PF_INITIAL_VALUE, index=i_start)

            for i in range(nb_stocks):
                state_fu[i], done_fu[i] = env_fu[i].reset(
                    action_fu[i], PF_INITIAL_VALUE, index=i_start
                )

            list_x_t, list_w_previous, list_pf_value_previous, list_daily_return_t = (
                [],
                [],
                [],
                [],
            )
            list_pf_value_previous_eq, list_pf_value_previous_s = [], []
            list_pf_value_previous_fu = list()
            for i in range(nb_stocks):
                list_pf_value_previous_fu.append(list())

            for batch_item in range(BATCH_SIZE):

                # load the different inputs from the previous loaded state
                x_t = state[0].reshape([-1] + list(state[0].shape))
                w_previous = state[1].reshape([-1] + list(state[1].shape))
                pf_value_previous = state[2]

                if np.random.rand() < RATIO_GREEDY:
                    # print('go')
                    # computation of the action of the agent
                    action = actor.compute_w(x_t, w_previous)
                else:
                    action = _get_random_action(nb_stocks)

                # given the state and the action, call the environment to go one
                # time step later
                state, _, _ = env.step(action)
                state_eq, _, _ = env_eq.step(weights_equal)
                state_s, _, _ = env_s.step(weights_single)

                for i in range(nb_stocks):
                    state_fu[i], _, done_fu[i] = env_fu[i].step(action_fu[i])

                # get the new state
                x_next = state[0]
                w_t = state[1]
                pf_value_t = state[2]

                pf_value_t_eq = state_eq[2]
                pf_value_t_s = state_s[2]

                for i in range(nb_stocks):
                    pf_value_t_fu[i] = state_fu[i][2]

                # let us compute the returns
                daily_return_t = x_next[-1, :, -1]
                # update into the PVM
                memory.update(i_start + batch_item, w_t)
                # store elements
                list_x_t.append(x_t.reshape(state[0].shape))
                list_w_previous.append(w_previous.reshape(state[1].shape))
                list_pf_value_previous.append([pf_value_previous])
                list_daily_return_t.append(daily_return_t)

                list_pf_value_previous_eq.append(pf_value_t_eq)
                list_pf_value_previous_s.append(pf_value_t_s)

                for i in range(nb_stocks):
                    list_pf_value_previous_fu[i].append(pf_value_t_fu[i])

                if batch_item == BATCH_SIZE - 1:
                    list_final_pf.append(pf_value_t)
                    list_final_pf_eq.append(pf_value_t_eq)
                    list_final_pf_s.append(pf_value_t_s)
                    for i in range(nb_stocks):
                        list_final_pf_fu[i].append(pf_value_t_fu[i])

                        if print_verbose:
                            # printing
                            if batch_item == 0:
                                print("start", i_start)
                                print("PF_start", round(pf_value_previous, 0))

                            if batch_item == BATCH_SIZE - 1:
                                print("PF_end", round(pf_value_t, 0))
                                print("weight", w_t)

            list_x_t = np.array(list_x_t)
            list_w_previous = np.array(list_w_previous)
            list_pf_value_previous = np.array(list_pf_value_previous)
            list_daily_return_t = np.array(list_daily_return_t)

            # for each batch, train the network to maximize the reward
            actor.train(
                list_x_t, list_w_previous, list_pf_value_previous, list_daily_return_t
            )
        _eval_perf(
            no_episode,
            actor,
            interactive_session,
            trade_env_args,
            asset_list,
            total_steps_train,
            total_steps_val,
            nb_stocks,
        )

    return actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s


# random action function


def _get_random_action(nb_stocks):
    random_vec = np.random.rand(nb_stocks + 1)
    return random_vec / np.sum(random_vec)


def _eval_perf(  # pylint: disable= too-many-arguments, too-many-locals
    no_episode,
    actor,
    render_plots,
    trade_env_args,
    asset_list,
    total_steps_train,
    total_steps_val,
    nb_stocks,
):
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
    env_eval = TradeEnv(**trade_env_args)

    w_init_test = np.array(np.array([1] + [0] * nb_stocks))

    # initialization of the environment
    state_eval, _ = env_eval.reset(w_init_test, PF_INIT_TEST, index=total_steps_train)

    # first element of the weight and portfolio value
    p_list_eval = [PF_INIT_TEST]
    w_list_eval = [w_init_test]

    for _ in range(
        total_steps_train, total_steps_train + total_steps_val - int(LENGTH_TENSOR / 2)
    ):
        x_t = state_eval[0].reshape([-1] + list(state_eval[0].shape))
        w_previous = state_eval[1].reshape([-1] + list(state_eval[1].shape))
        # pf_value_previous = state_eval[2]
        # compute the action
        action = actor.compute_w(x_t, w_previous)
        # step forward environment
        state_eval, _, _ = env_eval.step(action)

        # x_next = state_eval[0]
        w_t_eval = state_eval[1]
        pf_value_t_eval = state_eval[2]

        # daily_return_t = x_next[-1, :, -1]
        # print('current portfolio value', round(pf_value_previous,0))
        # print('weights', w_previous)
        p_list_eval.append(pf_value_t_eval)
        w_list_eval.append(w_t_eval)

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
    print("Max Draw Down of test PF value:", round(_get_max_draw_down(p_list_eval)))
    print("End of test weights:", w_list_eval[-1])

    if render_plots:
        plt.title("Portfolio evolution (validation set) episode {}".format(no_episode))
        plt.plot(p_list_eval, label="Agent Portfolio Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()
        plt.title(
            "Portfolio weights (end of validation set) episode {}".format(no_episode)
        )
        plt.bar(np.arange(nb_stocks + 1), list_weight_end_val[-1])
        plt.xticks(np.arange(nb_stocks + 1), ["Money"] + asset_list, rotation=45)
        plt.show()

    names = ["Money"] + asset_list
    w_list_eval = np.array(w_list_eval)

    if render_plots:
        for j in range(nb_stocks + 1):
            plt.plot(w_list_eval[:, j], label="Weight Stock {}".format(names[j]))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
        plt.show()


def _get_max_draw_down(p_list_eval):
    p_list_eval = np.array(p_list_eval)

    # end of the period
    i = np.argmax(
        np.maximum.accumulate(p_list_eval) - p_list_eval  # pylint: disable=no-member
    )
    j = np.argmax(p_list_eval[:i])  # start of period

    return p_list_eval[j] - p_list_eval[i]
