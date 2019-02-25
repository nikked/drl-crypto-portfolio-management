import unittest

import train_test_analyse_rl_algorithm


class TestTrainTestAnalyse(unittest.TestCase):
    def setUp(self):
        self.cli_options = {
            "crypto_data": False,
            "gpu_device": None,
            "interactive_session": False,
            "max_no_of_training_periods": 1000,
            "no_of_assets": 2,
            "plot_results": False,
            "verbose": False,
            "n_batches": 1,
            "n_episodes": 1,
            "window_length": 300,
            "batch_size": 10,
            "portfolio_value": 10000,
        }

    def test_stock_train_completes_fully(self):

        train_test_analyse_rl_algorithm.main(**self.cli_options)

        self.assertEqual(True, True)

    def test_crypto_train_completes_fully(self):

        crypto_options = self.cli_options
        crypto_options["crypto_data"] = True

        train_test_analyse_rl_algorithm.main(**crypto_options)

        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
