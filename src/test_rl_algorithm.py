import numpy as np

from src.params import PF_INIT_TEST


def test_rl_algorithm(  # pylint:  disable=too-many-arguments, too-many-locals
    window_length, actor, state_fu, done_fu, trade_envs, set_step_counts
):

    no_of_assets = len(state_fu)

    total_steps_train = set_step_counts["train"]
    total_steps_val = set_step_counts["validation"]
    total_steps_test = set_step_counts["test"]

    env_policy_network = trade_envs["policy_network"]
    env_equal_weighted = trade_envs["equal_weighted"]
    env_only_cash = trade_envs["only_cash"]
    env_full_on_one_stocks = trade_envs["full_on_one_stocks"]
    action_fu = trade_envs["action_fu"]

    weights_equal = np.array(np.array([1 / (no_of_assets + 1)] * (no_of_assets + 1)))
    weights_single = np.array(np.array([1] + [0.0] * no_of_assets))

    w_init_test = np.array(np.array([1] + [0] * no_of_assets))

    # initialization of the environment
    state, _ = env_policy_network.reset(
        w_init_test, PF_INIT_TEST, index=total_steps_train
    )

    state_eq, _ = env_equal_weighted.reset(
        weights_equal, PF_INIT_TEST, index=total_steps_train
    )
    state_s, _ = env_only_cash.reset(
        weights_single, PF_INIT_TEST, index=total_steps_train
    )

    for i in range(no_of_assets):
        state_fu[i], done_fu[i] = env_full_on_one_stocks[i].reset(
            action_fu[i], PF_INIT_TEST, index=total_steps_train
        )

    # first element of the weight and portfolio value
    p_list = [PF_INIT_TEST]
    w_list = [w_init_test]

    p_list_eq = [PF_INIT_TEST]
    p_list_s = [PF_INIT_TEST]

    p_list_fu = list()
    for i in range(no_of_assets):
        p_list_fu.append([PF_INIT_TEST])

    pf_value_t_fu = [0] * no_of_assets

    for k in range(
        total_steps_train + total_steps_val - int(window_length / 2),
        total_steps_train + total_steps_val + total_steps_test - window_length,
    ):
        x_current = state[0].reshape([-1] + list(state[0].shape))
        w_previous = state[1].reshape([-1] + list(state[1].shape))
        pf_value_previous = state[2]
        # compute the action
        action = actor.compute_w(x_current, w_previous)
        # step forward environment
        state, _, _ = env_policy_network.step(action)
        state_eq, _, _ = env_equal_weighted.step(weights_equal)
        state_s, _, _ = env_only_cash.step(weights_single)

        for i in range(no_of_assets):
            state_fu[i], _, done_fu[i] = env_full_on_one_stocks[i].step(action_fu[i])

        # x_next = state[0]
        w_current = state[1]
        pf_value_t = state[2]

        pf_value_t_eq = state_eq[2]
        pf_value_t_s = state_s[2]
        for i in range(no_of_assets):
            pf_value_t_fu[i] = state_fu[i][2]

        # dailyReturn_t = x_next[-1, :, -1]
        if k % 20 == 0:
            print("current portfolio value", round(pf_value_previous, 0))
            print("weights", w_previous)
        p_list.append(pf_value_t)
        w_list.append(w_current)

        p_list_eq.append(pf_value_t_eq)
        p_list_s.append(pf_value_t_s)
        for i in range(no_of_assets):
            p_list_fu[i].append(pf_value_t_fu[i])

        # here to breack the loop/not in original code
        if k == total_steps_train + total_steps_val - int(window_length / 2) + 100:
            break

    return p_list, p_list_eq, p_list_fu, p_list_s, w_list
