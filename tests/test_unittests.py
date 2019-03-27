import unittest

import deep_rl_portfolio


class TestTrainTestAnalyse(unittest.TestCase):
    def setUp(self):
        self.cli_options = {
            "stock_data": False,
            "gpu_device": None,
            "interactive_session": False,
            "no_of_assets": 2,
            "plot_results": False,
            "verbose": False,
            "n_batches": 1,
            "n_episodes": 1,
            "window_length": 120,
            "batch_size": 10,
            "portfolio_value": 10000,
            "start_date": "20190101",
            "end_date": "20190301",
            "trading_period_length": "2h",
            "test_mode": True,
        }

    def test_crypto_train_completes_fully(self):
        deep_rl_portfolio.main(**self.cli_options)
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
