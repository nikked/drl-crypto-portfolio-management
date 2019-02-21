import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.params import (
    ratio_greedy,
    batch_size,
    learning,
    n_episodes,
    kernel1_size,
    n,
    m,
    n_batches,
    ratio_regul,
    total_steps_train,
    total_steps_val,
    path_data

)


def analysis(
        p_list,
        p_list_eq,
        p_list_s,
        p_list_fu,
        w_list,
        list_final_pf,
        list_final_pf_eq,
        list_final_pf_s
):

    path = "individual_stocks_5yr/"
    times = pd.read_csv(path + "A_data.csv").date
    test_start_day = total_steps_train + total_steps_val - int(n / 2) + 10
    times = list(times[test_start_day:])

    data_type = path_data.split('/')[2][5:].split('.')[0]
    namesBio = ['JNJ',
                'PFE', 'AMGN', 'MDT', 'CELG', 'LLY']
    namesUtilities = ['XOM', 'CVX', 'MRK', 'SLB', 'MMM']
    namesTech = ['FB', 'AMZN', 'MSFT', 'AAPL',
                 'T', 'VZ', 'CMCSA', 'IBM', 'CRM', 'INTC']

    if data_type == 'Utilities':
        list_stock = namesUtilities
    elif data_type == 'Bio':
        list_stock = namesBio
    elif data_type == 'Tech':
        list_stock = namesTech
    else:
        list_stock = [i for i in range(m)]

    plt.title('Portfolio Value (Test Set) {}: {}, {}, {}, {}, {}, {}, {}, {}'.format(
        data_type, batch_size, learning, ratio_greedy, n_episodes, n, kernel1_size, n_batches, ratio_regul))
    plt.plot(p_list, label='Agent Portfolio Value')
    plt.plot(p_list_eq, label='Equi-weighted Portfolio Value')
    plt.plot(p_list_s, label='Secured Portfolio Value')
    for i in range(m):
        plt.plot(
            p_list_fu[i], label='Full Stock {} Portfolio Value'.format(list_stock[i]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    # In[ ]:

    names = ['Money'] + list_stock
    w_list = np.array(w_list)
    for j in range(m + 1):
        plt.plot(w_list[:, j], label='Weight Stock {}'.format(names[j]))
        plt.title('Weight evolution during testing')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
    plt.show()

    # In[ ]:

    plt.plot(np.array(p_list) - np.array(p_list_eq))

    # In[ ]:

    index1 = 0
    index2 = -1

    plt.plot(list_final_pf[index1:index2], label='Agent Portfolio Value')
    plt.plot(list_final_pf_eq[index1:index2], label='Baseline Portfolio Value')
    plt.plot(list_final_pf_s[index1:index2], label='Secured Portfolio Value')
    plt.legend()
    plt.show()

    # In[ ]:

    plt.plot((np.array(list_final_pf) - np.array(list_final_pf_eq)))
