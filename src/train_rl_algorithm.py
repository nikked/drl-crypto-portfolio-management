import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.params import TRADING_COST, INTEREST_RATE, EPSILON_GREEDY_THRESHOLD

from src.policy import Policy
from src.environment import TradeEnv
from src.pvm import PVM


def train_rl_algorithm(train_options, trade_envs, asset_list, train_test_split):
    print("\nStarting to train deep reinforcement learning algorithm...")

    tf.reset_default_graph()
    sess = tf.Session()

    print("\nInitializing Agent CNN with Tensorflow")
    benchmark_weights = _initialize_benchmark_weights(train_options["no_of_assets"])
    nb_feature_map = trade_envs["args"]["data"].shape[0]
    agent = Policy(
        train_options["no_of_assets"],
        train_options,
        sess,
        benchmark_weights["equal"],
        nb_feature_map,
        trading_cost=TRADING_COST,
        interest_rate=INTEREST_RATE,
    )

    print("\nInitializing tensorflow graphs")
    sess.run(tf.global_variables_initializer())

    train_performance_lists = {
        "policy_network": [],
        "equal_weighted": [],
        "only_cash": [],
        "single_asset": [list() for item in range(train_options["no_of_assets"])],
    }

    # Run training episodes
    env_states = None
    for n_episode in range(train_options["n_episodes"]):

        print("\nStarting reinforcement learning episode", n_episode + 1)
        if n_episode == 0 and train_options["validate_during_training"]:
            _test_and_report_progress(
                train_options,
                "Before Training",
                agent,
                trade_envs["args"],
                asset_list,
                train_test_split,
            )

        env_states = _train_episode(
            train_options, trade_envs, train_test_split, agent, train_performance_lists
        )

        if train_options["validate_during_training"]:
            _test_and_report_progress(
                train_options,
                n_episode,
                agent,
                trade_envs["args"],
                asset_list,
                train_test_split,
            )

    return (
        agent,
        env_states["single_assets_states"],
        env_states["single_assets_done"],
        train_performance_lists,
    )


def _initialize_benchmark_weights(no_of_assets):

    benchmark_weights = {
        "equal": np.array(np.array([1 / (no_of_assets + 1)] * (no_of_assets + 1))),
        "only_cash": np.array(np.array([1] + [0.0] * no_of_assets)),
        "single_assets": np.eye(no_of_assets + 1, dtype=int),
    }

    benchmark_weights["equal"] = np.array(
        np.array([1 / (no_of_assets + 1)] * (no_of_assets + 1))
    )
    benchmark_weights["only_cash"] = np.array(np.array([1] + [0.0] * no_of_assets))

    return benchmark_weights


def _train_episode(
    train_options, trade_envs, train_test_split, agent, train_performance_lists
):

    benchmark_weights = _initialize_benchmark_weights(train_options["no_of_assets"])

    # init the PVM with the training parameters
    w_init_train = np.array(np.array([1] + [0] * train_options["no_of_assets"]))

    memory = PVM(train_test_split["train"], train_options["batch_size"], w_init_train)

    env_states = None

    for idx in range(train_options["n_batches"]):

        if train_options["verbose"]:
            print("\nTraining batch: {}/{}".format(idx + 1, train_options["n_batches"]))
        env_states = _train_batch(
            agent,
            train_performance_lists,
            train_options,
            memory,
            trade_envs,
            benchmark_weights,
        )

    return env_states


def _test_and_report_progress(  # pylint: disable=too-many-arguments
    train_options, n_episode, agent, trade_env_args, asset_list, train_test_split
):
    """
    This function evaluates the performance of the different types of agents.

    """

    print("\nEvaluating agent performance")

    policy_performance_tracker = {
        "list_weight_end_val": [],
        "list_pf_end_training": [],
        "list_pf_min_training": [],
        "list_pf_max_training": [],
        "list_pf_mean_training": [],
        "list_pf_dd_training": [],
    }

    w_list_eval, p_list_eval = _validate_agent_performance(
        train_options, trade_env_args, train_test_split, agent
    )

    policy_performance_tracker["list_weight_end_val"].append(w_list_eval[-1])
    policy_performance_tracker["list_pf_end_training"].append(p_list_eval[-1])
    policy_performance_tracker["list_pf_min_training"].append(np.min(p_list_eval))
    policy_performance_tracker["list_pf_max_training"].append(np.max(p_list_eval))
    policy_performance_tracker["list_pf_mean_training"].append(np.mean(p_list_eval))

    policy_performance_tracker["list_pf_dd_training"].append(
        _get_max_draw_down(p_list_eval)
    )

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
        plt.bar(
            np.arange(train_options["no_of_assets"] + 1),
            policy_performance_tracker["list_weight_end_val"][-1],
        )
        plt.xticks(
            np.arange(train_options["no_of_assets"] + 1),
            ["Money"] + asset_list,
            rotation=45,
        )
        plt.show()

    names = ["Money"] + asset_list
    w_list_eval = np.array(w_list_eval)

    if train_options["interactive_session"]:
        for j in range(train_options["no_of_assets"] + 1):
            plt.plot(w_list_eval[:, j], label="Weight Stock {}".format(names[j]))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
        plt.show()


def _validate_agent_performance(train_options, trade_env_args, train_test_split, agent):
    # environment for trading of the agent
    env_eval = TradeEnv(**trade_env_args)

    w_init_test = np.array(np.array([1] + [0] * train_options["no_of_assets"]))

    # initialization of the environment
    state_eval, _ = env_eval.reset(
        w_init_test, train_options["portfolio_value"], index=train_test_split["train"]
    )

    # first element of the weight and portfolio value
    p_list_eval = [train_options["portfolio_value"]]
    w_list_eval = [w_init_test]

    # Using validation set
    for _ in tqdm(
        range(
            train_test_split["train"],
            train_test_split["train"] + train_test_split["validation"],
        )
    ):
        x_t = state_eval[0].reshape([-1] + list(state_eval[0].shape))
        w_previous = state_eval[1].reshape([-1] + list(state_eval[1].shape))

        # compute the action
        action = agent.compute_w(x_t, w_previous)
        # step forward environment
        state_eval, _, _ = env_eval.step(action)

        w_t_eval = state_eval[1]
        pf_value_t_eval = state_eval[2]

        p_list_eval.append(pf_value_t_eval)
        w_list_eval.append(w_t_eval)

    return w_list_eval, p_list_eval


def _get_max_draw_down(p_list_eval):
    p_list_eval = np.array(p_list_eval)

    # end of the period
    i = np.argmax(  # pylint: disable=no-member
        np.maximum.accumulate(p_list_eval)  # pylint: disable=no-member
        - p_list_eval  # pylint: disable=no-member
    )
    j = np.argmax(p_list_eval[:i])  # start of period

    return p_list_eval[j] - p_list_eval[i]


def _train_batch(  # pylint: disable=too-many-arguments
    agent, train_performance_lists, train_options, memory, trade_envs, benchmark_weights
):

    no_of_assets = train_options["no_of_assets"]
    single_asset_pf_values_t = [0] * no_of_assets

    # draw the starting point of the batch
    i_start = memory.draw()

    env_states = _reset_memory_states(
        train_options, trade_envs, memory, i_start, benchmark_weights
    )

    train_session_tracker = _initialize_train_session_tracker(no_of_assets)

    for batch_item in range(train_options["batch_size"]):

        _train_batch_item(
            env_states,
            agent,
            trade_envs,
            benchmark_weights,
            single_asset_pf_values_t,
            memory,
            i_start,
            batch_item,
            train_session_tracker,
            train_options,
            train_performance_lists,
        )

    # for each batch, train the network to maximize the reward
    agent.train(
        np.array(train_session_tracker["policy_x_t"]),
        np.array(train_session_tracker["policy_prev_weights"]),
        np.array(train_session_tracker["policy_prev_value"]),
        np.array(train_session_tracker["policy_daily_return_t"]),
    )

    return env_states


def _reset_memory_states(train_options, trade_envs, memory, i_start, benchmark_weights):
    # reset the environment with the weight from PVM at the starting point
    # reset also with a portfolio value with initial portfolio value
    state, policy_done = trade_envs["policy_network"].reset(
        memory.get_w(i_start), train_options["portfolio_value"], index=i_start
    )
    state_eq, equal_done = trade_envs["equal_weighted"].reset(
        benchmark_weights["equal"], train_options["portfolio_value"], index=i_start
    )
    state_s, cash_done = trade_envs["only_cash"].reset(
        benchmark_weights["only_cash"], train_options["portfolio_value"], index=i_start
    )

    state_single_assets = [0] * train_options["no_of_assets"]
    done_single_assets = [0] * train_options["no_of_assets"]

    full_on_one_weights = np.eye(train_options["no_of_assets"] + 1, dtype=int)

    for i in range(train_options["no_of_assets"]):
        state_single_assets[i], done_single_assets[i] = trade_envs[
            "full_on_one_stocks"
        ][i].reset(
            full_on_one_weights[i + 1, :],
            train_options["portfolio_value"],
            index=i_start,
        )

    return {
        "policy_network": {"state": state, "done": policy_done},
        "equal_weighted": {"state": state_eq, "done": equal_done},
        "only_cash": {"state": state_s, "done": cash_done},
        "single_assets_states": state_single_assets,
        "single_assets_done": done_single_assets,
    }


def _initialize_train_session_tracker(no_of_assets):
    single_asset_prev_values = list()
    for _ in range(no_of_assets):
        single_asset_prev_values.append(list())

    train_session_tracker = {
        "policy_x_t": [],
        "policy_prev_weights": [],
        "policy_prev_value": [],
        "policy_daily_return_t": [],
        "equal_prev_value": [],
        "cash_prev_value": [],
        "single_asset_prev_values": single_asset_prev_values,
    }

    return train_session_tracker


def _train_batch_item(  # pylint: disable=too-many-arguments, too-many-locals
    env_states,
    agent,
    trade_envs,
    benchmark_weights,
    single_asset_pf_values_t,
    memory,
    i_start,
    batch_item,
    train_session_tracker,
    train_options,
    train_performance_lists,
):

    pf_value_previous = env_states["policy_network"]["state"][2]

    x_t, w_previous = _take_train_step(
        agent, env_states, train_options["no_of_assets"], trade_envs, benchmark_weights
    )

    new_state = _update_state(
        env_states, single_asset_pf_values_t, train_options["no_of_assets"]
    )

    # let us compute the returns
    print(
        "\n******\nNOTE THIS BUG, SOMETIMES X_NEXT IS EMPTY CAUSING THE TRAIN TO CRASH"
    )
    print(new_state["x_next"])
    daily_return_t = new_state["x_next"][-1, :, -1]

    # update into the PVM
    memory.update(i_start + batch_item, new_state["w_t"])

    # store elements
    train_session_tracker["policy_x_t"].append(
        x_t.reshape(env_states["policy_network"]["state"][0].shape)
    )
    train_session_tracker["policy_prev_weights"].append(
        w_previous.reshape(env_states["policy_network"]["state"][1].shape)
    )
    train_session_tracker["policy_prev_value"].append([pf_value_previous])
    train_session_tracker["policy_daily_return_t"].append(daily_return_t)

    train_session_tracker["equal_prev_value"].append(new_state["pf_value_t_eq"])
    train_session_tracker["cash_prev_value"].append(new_state["pf_value_t_s"])

    for i in range(train_options["no_of_assets"]):
        train_session_tracker["single_asset_prev_values"][i].append(
            single_asset_pf_values_t[i]
        )

    if batch_item == train_options["batch_size"] - 1:
        _handle_after_last_item_of_batch(
            train_performance_lists,
            new_state,
            train_options["no_of_assets"],
            single_asset_pf_values_t,
        )

        if train_options["verbose"]:
            if batch_item == 0:
                print("start", i_start)
                print("PF_start", round(pf_value_previous, 0))

            if batch_item == train_options["batch_size"] - 1:
                print("Ptf value: ", round(new_state["pf_value_t"], 0))
                print("Ptf weights: ", new_state["w_t"])


def _take_train_step(agent, env_states, no_of_assets, trade_envs, benchmark_weights):

    # load the different inputs from the previous loaded state
    x_t = env_states["policy_network"]["state"][0].reshape(
        [-1] + list(env_states["policy_network"]["state"][0].shape)
    )
    w_previous = env_states["policy_network"]["state"][1].reshape(
        [-1] + list(env_states["policy_network"]["state"][1].shape)
    )

    if np.random.rand() < EPSILON_GREEDY_THRESHOLD:
        # computation of the action of the agent
        action = agent.compute_w(x_t, w_previous)
    else:
        action = _get_random_action(no_of_assets)

    # given the state and the action, call the environment to go one
    # time step later
    env_states["policy_network"]["state"], _, _ = trade_envs["policy_network"].step(
        action
    )
    env_states["equal_weighted"]["state"], _, _ = trade_envs["equal_weighted"].step(
        benchmark_weights["equal"]
    )
    env_states["only_cash"]["state"], _, _ = trade_envs["only_cash"].step(
        benchmark_weights["only_cash"]
    )

    for i in range(no_of_assets):
        env_states["single_assets_states"][i], _, env_states["single_assets_done"][
            i
        ] = trade_envs["full_on_one_stocks"][i].step(
            benchmark_weights["single_assets"][i + 1, :]
        )

    return x_t, w_previous


def _update_state(env_states, single_asset_pf_values_t, no_of_assets):

    # get the new state
    x_next = env_states["policy_network"]["state"][0]
    w_t = env_states["policy_network"]["state"][1]
    pf_value_t = env_states["policy_network"]["state"][2]

    pf_value_t_eq = env_states["equal_weighted"]["state"][2]
    pf_value_t_s = env_states["only_cash"]["state"][2]

    for i in range(no_of_assets):
        single_asset_pf_values_t[i] = env_states["single_assets_states"][i][2]

    new_state = {
        "x_next": x_next,
        "w_t": w_t,
        "pf_value_t": pf_value_t,
        "pf_value_t_eq": pf_value_t_eq,
        "pf_value_t_s": pf_value_t_s,
    }

    return new_state


def _get_random_action(no_of_assets):
    random_vec = np.random.rand(no_of_assets + 1)
    return random_vec / np.sum(random_vec)


def _handle_after_last_item_of_batch(
    train_performance_lists, new_state, no_of_assets, single_asset_pf_values_t
):
    train_performance_lists["policy_network"].append(new_state["pf_value_t"])
    train_performance_lists["equal_weighted"].append(new_state["pf_value_t_eq"])
    train_performance_lists["only_cash"].append(new_state["pf_value_t_s"])
    for i in range(no_of_assets):
        train_performance_lists["single_asset"][i].append(single_asset_pf_values_t[i])
