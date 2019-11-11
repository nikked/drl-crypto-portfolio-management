import unittest

import deep_rl_portfolio


class TestTrainTestAnalyse(unittest.TestCase):
    def setUp(self):
        self.cli_options = {
            "interactive_session": False,
            "gpu_device": None,
            "verbose": True,
            "no_of_assets": 7,
            "plot_results": False,
            "ratio_train": 0.5,
            "ratio_val": 0.0,
            "n_episodes": 1,
            "n_batches": 1,
            "window_length": 77,
            "batch_size": 1,
            "portfolio_value": 100,
            "start_date": "20190101",
            "test_start_date": "20190201",
            "end_date": "20190301",
            "trading_period_length": "2h",
            "max_pf_weight_penalty": 0.7,
            "test_mode": True,
            "validate_during_training": False,
            "train_session_name": "test_run_with_long_name",
        }

    def test_crypto_train_completes_fully(self):
        deep_rl_portfolio.main(**self.cli_options)
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
