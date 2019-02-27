import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.params import (  # pylint: disable=ungrouped-imports
    TRADING_COST,
    INTEREST_RATE,
    RATIO_GREEDY,
)

from src.policy import Policy
from src.environment import TradeEnv
from src.pvm import PVM


def train_rl_algorithm(  # pylint: disable= too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    train_options, trade_envs, asset_list, step_counts
):
    print("\nStarting to train deep reinforcement learning algorithm...")

    no_of_assets = len(asset_list)

    total_steps_train = step_counts["train"]
    total_steps_val = step_counts["validation"]

    nb_feature_map = trade_envs["args"]["data"].shape[0]

    weights_equal = np.array(np.array([1 / (no_of_assets + 1)] * (no_of_assets + 1)))
    weights_single = np.array(np.array([1] + [0.0] * no_of_assets))

    tf.reset_default_graph()

    sess = tf.Session()

    actor = Policy(
        no_of_assets,
        train_options,
        sess,
        weights_equal,
        nb_feature_map,
        trading_cost=TRADING_COST,
        interest_rate=INTEREST_RATE,
    )

    print("\nInitializing tensorflow graphs")
    sess.run(tf.global_variables_initializer())

    list_final_pf = list()
    list_final_pf_eq = list()
    list_final_pf_s = list()

    list_final_pf_fu = list()
    state_fu = [0] * no_of_assets
    done_fu = [0] * no_of_assets

    pf_value_t_fu = [0] * no_of_assets

    for i in range(no_of_assets):
        list_final_pf_fu.append(list())

    # Run training episodes
    for n_episode in range(
        train_options["n_episodes"]
    ):  # pylint: disable= too-many-nested-blocks
        print("\nStarting reinforcement learning episode", n_episode + 1)
        if n_episode == 0:
            _eval_perf(
                train_options,
                "Before Training",
                actor,
                trade_envs["args"],
                asset_list,
                total_steps_train,
                total_steps_val,
                no_of_assets,
            )
        # init the PVM with the training parameters

        # dict_train['w_init_train']
        w_init_train = np.array(np.array([1] + [0] * no_of_assets))

        memory = PVM(total_steps_train, train_options["batch_size"], w_init_train)

        for idx in range(train_options["n_batches"]):

            if train_options["verbose"]:
                print(
                    "\nTraining batch: {}/{}".format(
                        idx + 1, train_options["n_batches"]
                    )
                )

            # draw the starting point of the batch
            i_start = memory.draw()

            # reset the environment with the weight from PVM at the starting point
            # reset also with a portfolio value with initial portfolio value
            state, _ = trade_envs["policy_network"].reset(
                memory.get_w(i_start), train_options["portfolio_value"], index=i_start
            )
            state_eq, _ = trade_envs["equal_weighted"].reset(
                weights_equal, train_options["portfolio_value"], index=i_start
            )
            state_s, _ = trade_envs["only_cash"].reset(
                weights_single, train_options["portfolio_value"], index=i_start
            )

            for i in range(no_of_assets):
                state_fu[i], done_fu[i] = trade_envs["full_on_one_stocks"][i].reset(
                    trade_envs["action_fu"][i],
                    train_options["portfolio_value"],
                    index=i_start,
                )

            list_x_t, list_w_previous, list_pf_value_previous, list_daily_return_t = (
                [],
                [],
                [],
                [],
            )
            list_pf_value_previous_eq, list_pf_value_previous_s = [], []
            list_pf_value_previous_fu = list()
            for i in range(no_of_assets):
                list_pf_value_previous_fu.append(list())

            for batch_item in range(train_options["batch_size"]):
                pf_value_previous = state[2]
                x_t, w_previous = _train_batch_item(
                    actor,
                    state,
                    no_of_assets,
                    trade_envs,
                    weights_equal,
                    weights_single,
                    state_fu,
                    done_fu,
                )

                # get the new state
                x_next = state[0]
                w_t = state[1]
                pf_value_t = state[2]

                pf_value_t_eq = state_eq[2]
                pf_value_t_s = state_s[2]

                for i in range(no_of_assets):
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

                for i in range(no_of_assets):
                    list_pf_value_previous_fu[i].append(pf_value_t_fu[i])

                if batch_item == train_options["batch_size"] - 1:
                    list_final_pf.append(pf_value_t)
                    list_final_pf_eq.append(pf_value_t_eq)
                    list_final_pf_s.append(pf_value_t_s)
                    for i in range(no_of_assets):
                        list_final_pf_fu[i].append(pf_value_t_fu[i])

                    if train_options["verbose"]:
                        if batch_item == 0:
                            print("start", i_start)
                            print("PF_start", round(pf_value_previous, 0))

                        if batch_item == train_options["batch_size"] - 1:
                            print("Ptf value: ", round(pf_value_t, 0))
                            print("Ptf weights: ", w_t)

            list_x_t = np.array(list_x_t)
            list_w_previous = np.array(list_w_previous)
            list_pf_value_previous = np.array(list_pf_value_previous)
            list_daily_return_t = np.array(list_daily_return_t)

            # for each batch, train the network to maximize the reward
            actor.train(
                list_x_t, list_w_previous, list_pf_value_previous, list_daily_return_t
            )
        _eval_perf(
            train_options,
            n_episode,
            actor,
            trade_envs["args"],
            asset_list,
            total_steps_train,
            total_steps_val,
            no_of_assets,
        )

    train_performance_lists = {
        "policy_network": list_final_pf,
        "equal_weighted": list_final_pf_eq,
        "single_asset": list_final_pf,
    }

    return actor, state_fu, done_fu, train_performance_lists


def _train_batch_item(
    actor,
    state,
    no_of_assets,
    trade_envs,
    weights_equal,
    weights_single,
    state_fu,
    done_fu,
):
    # load the different inputs from the previous loaded state
    x_t = state[0].reshape([-1] + list(state[0].shape))
    w_previous = state[1].reshape([-1] + list(state[1].shape))

    if np.random.rand() < RATIO_GREEDY:
        # print('go')
        # computation of the action of the agent
        action = actor.compute_w(x_t, w_previous)
    else:
        action = _get_random_action(no_of_assets)

    # given the state and the action, call the environment to go one
    # time step later
    trade_envs["policy_network"].step(action)
    trade_envs["equal_weighted"].step(weights_equal)
    trade_envs["only_cash"].step(weights_single)

    for i in range(no_of_assets):
        state_fu[i], _, done_fu[i] = trade_envs["full_on_one_stocks"][i].step(
            trade_envs["action_fu"][i]
        )

    return x_t, w_previous


def _get_random_action(no_of_assets):
    random_vec = np.random.rand(no_of_assets + 1)
    return random_vec / np.sum(random_vec)


def _eval_perf(  # pylint: disable= too-many-arguments, too-many-locals
    train_options,
    n_episode,
    actor,
    trade_env_args,
    asset_list,
    total_steps_train,
    total_steps_val,
    no_of_assets,
):
    """
    This function evaluates the performance of the different types of agents.


    """

    print("\nEvaluating agent performance")
    list_weight_end_val = list()
    list_pf_end_training = list()
    list_pf_min_training = list()
    list_pf_max_training = list()
    list_pf_mean_training = list()
    list_pf_dd_training = list()

    #######TEST#######
    # environment for trading of the agent
    env_eval = TradeEnv(**trade_env_args)

    w_init_test = np.array(np.array([1] + [0] * no_of_assets))

    # initialization of the environment
    state_eval, _ = env_eval.reset(
        w_init_test, train_options["portfolio_value"], index=total_steps_train
    )

    # first element of the weight and portfolio value
    p_list_eval = [train_options["portfolio_value"]]
    w_list_eval = [w_init_test]

    for _ in tqdm(
        range(
            total_steps_train,
            total_steps_train
            + total_steps_val
            - int(train_options["window_length"] / 2),
        )
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

        p_list_eval.append(pf_value_t_eval)
        w_list_eval.append(w_t_eval)

    list_weight_end_val.append(w_list_eval[-1])
    list_pf_end_training.append(p_list_eval[-1])
    list_pf_min_training.append(np.min(p_list_eval))
    list_pf_max_training.append(np.max(p_list_eval))
    list_pf_mean_training.append(np.mean(p_list_eval))

    list_pf_dd_training.append(_get_max_draw_down(p_list_eval))

    print("\nPerformance report:")
    print("End of test PF value:", round(p_list_eval[-1]))
    print("Min of test PF value:", round(np.min(p_list_eval)))
    print("Max of test PF value:", round(np.max(p_list_eval)))
    print("Mean of test PF value:", round(np.mean(p_list_eval)))
    print("Max Draw Down of test PF value:", round(_get_max_draw_down(p_list_eval)))
    print("End of test weights:", w_list_eval[-1])
    print("\n")

    if train_options["interactive_session"]:
        plt.title("Portfolio evolution (validation set) episode {}".format(n_episode))
        plt.plot(p_list_eval, label="Agent Portfolio Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()
        plt.title(
            "Portfolio weights (end of validation set) episode {}".format(n_episode)
        )
        plt.bar(np.arange(no_of_assets + 1), list_weight_end_val[-1])
        plt.xticks(np.arange(no_of_assets + 1), ["Money"] + asset_list, rotation=45)
        plt.show()

    names = ["Money"] + asset_list
    w_list_eval = np.array(w_list_eval)

    if train_options["interactive_session"]:
        for j in range(no_of_assets + 1):
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
