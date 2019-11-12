import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.params import TRADING_COST, INTEREST_RATE, EPSILON_GREEDY_THRESHOLD

from src.cnn_policy import CNNPolicy


def train_rl_algorithm(train_options, trade_envs, asset_list, train_test_split):
    print("\nStarting to train deep reinforcement learning algorithm...")

    tf.reset_default_graph()
    sess = tf.Session()

    print("\nInitializing Agent CNN with Tensorflow")
    benchmark_weights = _initialize_benchmark_weights(train_options["no_of_assets"])
    nb_feature_map = trade_envs["args"]["data"].shape[0]
    agent = CNNPolicy(
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

    env_states = None
    for n_episode in range(train_options["n_episodes"]):

        print("\nStarting reinforcement learning episode", n_episode + 1)

        env_states = _train_episode(
            train_options, trade_envs, train_test_split, agent, train_performance_lists
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

    w_init_train = np.array(np.array([1] + [0] * train_options["no_of_assets"]))

    memory = np.transpose(np.array([w_init_train] * train_test_split["train"]))

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


def _train_batch(  # pylint: disable=too-many-arguments
    agent, train_performance_lists, train_options, memory, trade_envs, benchmark_weights
):

    no_of_assets = train_options["no_of_assets"]
    single_asset_pf_values_t = [0] * no_of_assets

    i_start = train_options["window_length"]

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

    agent.train(
        np.array(train_session_tracker["policy_x_t"]),
        np.array(train_session_tracker["policy_prev_weights"]),
        np.array(train_session_tracker["policy_prev_value"]),
        np.array(train_session_tracker["policy_daily_return_t"]),
    )

    return env_states


def _reset_memory_states(train_options, trade_envs, memory, i_start, benchmark_weights):
    state, policy_done = trade_envs["policy_network"].reset_environment(
        memory[:, i_start], train_options["portfolio_value"], index=i_start
    )

    state_eq, equal_done = trade_envs["equal_weighted"].reset_environment(
        benchmark_weights["equal"], train_options["portfolio_value"], index=i_start
    )
    state_s, cash_done = trade_envs["only_cash"].reset_environment(
        benchmark_weights["only_cash"], train_options["portfolio_value"], index=i_start
    )

    state_single_assets = [0] * train_options["no_of_assets"]
    done_single_assets = [0] * train_options["no_of_assets"]

    full_on_one_weights = np.eye(train_options["no_of_assets"] + 1, dtype=int)

    for i in range(train_options["no_of_assets"]):
        state_single_assets[i], done_single_assets[i] = trade_envs[
            "full_on_one_stocks"
        ][i].reset_environment(
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

    daily_return_t = new_state["x_next"][-1, :, -1]

    memory[:, i_start + batch_item] = new_state["w_t"]

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
                print("Ptf value: ", round(new_state["pf_value_t"], 2))
                print("Ptf weights: ", new_state["w_t"])


def _take_train_step(agent, env_states, no_of_assets, trade_envs, benchmark_weights):

    x_t = env_states["policy_network"]["state"][0].reshape(
        [-1] + list(env_states["policy_network"]["state"][0].shape)
    )
    w_previous = env_states["policy_network"]["state"][1].reshape(
        [-1] + list(env_states["policy_network"]["state"][1].shape)
    )

    if np.random.rand() < EPSILON_GREEDY_THRESHOLD:
        action = agent.compute_new_ptf_weights(x_t, w_previous)
    else:
        action = _get_random_action(no_of_assets)

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
