import numpy as np


def test_rl_algorithm(  # pylint:  disable=too-many-arguments, too-many-locals
    train_options, actor, state_fu, done_fu, trade_envs, train_test_split
):

    print("\nTesting algorithm performance with test set")
    no_of_assets = len(state_fu)

    weights_equal = np.array(np.array([1 / (no_of_assets + 1)] * (no_of_assets + 1)))
    weights_single = np.array(np.array([1] + [0.0] * no_of_assets))

    w_init_test = np.array(np.array([1] + [0] * no_of_assets))

    # initialization of the environment
    state, _ = trade_envs["policy_network"].reset(
        w_init_test, train_options["portfolio_value"], index=train_test_split["train"]
    )

    state_eq, _ = trade_envs["equal_weighted"].reset(
        weights_equal, train_options["portfolio_value"], index=train_test_split["train"]
    )
    state_s, _ = trade_envs["only_cash"].reset(
        weights_single,
        train_options["portfolio_value"],
        index=train_test_split["train"],
    )

    full_on_one_weights = np.eye(no_of_assets + 1, dtype=int)
    for i in range(no_of_assets):
        state_fu[i], done_fu[i] = trade_envs["full_on_one_stocks"][i].reset(
            full_on_one_weights[i + 1, :],
            train_options["portfolio_value"],
            index=train_test_split["train"],
        )

    # first element of the weight and portfolio value
    p_list = [train_options["portfolio_value"]]
    w_list = [w_init_test]

    p_list_eq = [train_options["portfolio_value"]]
    p_list_s = [train_options["portfolio_value"]]

    p_list_fu = list()
    for i in range(no_of_assets):
        p_list_fu.append([train_options["portfolio_value"]])

    pf_value_t_fu = [0] * no_of_assets

    # Using test set
    for k in range(
        train_test_split["train"]
        + train_test_split["validation"]
        - int(train_options["window_length"] / 2),
        train_test_split["train"]
        + train_test_split["validation"]
        + train_test_split["test"]
        - train_options["window_length"],
    ):
        x_current = state[0].reshape([-1] + list(state[0].shape))
        w_previous = state[1].reshape([-1] + list(state[1].shape))
        pf_value_previous = state[2]
        # compute the action
        action = actor.compute_w(x_current, w_previous)
        # step forward environment
        state, _, _ = trade_envs["policy_network"].step(action)
        state_eq, _, _ = trade_envs["equal_weighted"].step(weights_equal)
        state_s, _, _ = trade_envs["only_cash"].step(weights_single)

        for i in range(no_of_assets):
            state_fu[i], _, done_fu[i] = trade_envs["full_on_one_stocks"][i].step(
                full_on_one_weights[i + 1, :]
            )

        # x_next = state[0]
        w_current = state[1]
        pf_value_t = state[2]

        pf_value_t_eq = state_eq[2]
        pf_value_t_s = state_s[2]
        for i in range(no_of_assets):
            pf_value_t_fu[i] = state_fu[i][2]

        # dailyReturn_t = x_next[-1, :, -1]
        if k % 20 == 0:
            print("Ptf value: ", round(pf_value_previous, 0))
            print("Ptf weights: ", w_previous[0])
        p_list.append(pf_value_t)
        w_list.append(w_current)

        p_list_eq.append(pf_value_t_eq)
        p_list_s.append(pf_value_t_s)
        for i in range(no_of_assets):
            p_list_fu[i].append(pf_value_t_fu[i])

        # here to breack the loop/not in original code
        if (
            k
            == train_test_split["train"]
            + train_test_split["validation"]
            - int(train_options["window_length"] / 2)
            + 100
        ):
            break

    test_performance_lists = {
        "p_list": p_list,
        "p_list_eq": p_list_eq,
        "p_list_fu": p_list_fu,
        "p_list_s": p_list_s,
        "w_list": w_list,
    }

    return test_performance_lists
