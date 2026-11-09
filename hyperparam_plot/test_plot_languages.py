import unittest

import numpy as np

from hyperparam_plot import plot_batch_hyperparams
from hyperparam_plot import plot_hyperparams


class HyperparameterPlotTests(unittest.TestCase):
    def test_language_argument_defaults_to_english_and_accepts_chinese(self):
        for module in (plot_hyperparams, plot_batch_hyperparams):
            self.assertEqual(module.parse_args([]).lang, "en")
            self.assertEqual(module.parse_args(["--lang", "zh"]).lang, "zh")

    def test_beta_experiment_uses_confirmed_five_run_data(self):
        np.testing.assert_array_equal(
            plot_hyperparams.HYPERPARAM_VALUES,
            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        )
        np.testing.assert_array_equal(
            plot_hyperparams.VGG8_ACCURACY_RUNS,
            np.array(
                [
                    [71.84, 71.91, 72.08, 71.75, 71.89],
                    [71.89, 71.82, 71.45, 71.90, 71.34],
                    [71.54, 72.19, 71.60, 71.88, 71.96],
                    [71.17, 71.50, 71.61, 71.59, 71.44],
                    [71.33, 71.50, 71.30, 71.61, 71.56],
                    [70.99, 71.07, 71.19, 70.80, 70.53],
                ]
            ),
        )
        np.testing.assert_array_equal(
            plot_hyperparams.RESNET8X4_ACCURACY_RUNS,
            np.array(
                [
                    [73.51, 73.06, 73.38, 73.33, 73.14],
                    [73.31, 73.25, 73.17, 73.46, 73.41],
                    [73.43, 73.38, 73.12, 73.19, 73.10],
                    [72.82, 72.86, 73.31, 73.22, 73.05],
                    [72.93, 72.57, 72.69, 72.98, 72.75],
                    [72.15, 72.37, 71.99, 72.34, 71.99],
                ]
            ),
        )

    def test_batch_experiment_uses_confirmed_five_run_data(self):
        np.testing.assert_array_equal(
            plot_batch_hyperparams.BATCH_VALUES,
            np.array([16, 32, 64, 128, 256]),
        )
        np.testing.assert_array_equal(
            plot_batch_hyperparams.VGG8_ACCURACY_RUNS,
            np.array(
                [
                    [71.05, 70.85, 70.79, 70.32, 70.66],
                    [71.59, 71.61, 71.74, 71.80, 71.50],
                    [71.89, 71.82, 71.45, 71.90, 71.34],
                    [71.95, 71.78, 71.68, 71.23, 71.18],
                    [70.82, 70.65, 70.62, 71.14, 70.73],
                ]
            ),
        )
        np.testing.assert_array_equal(
            plot_batch_hyperparams.RESNET8X4_ACCURACY_RUNS,
            np.array(
                [
                    [72.11, 72.18, 72.32, 72.43, 72.20],
                    [72.99, 72.98, 73.32, 73.26, 73.44],
                    [73.31, 73.25, 73.17, 73.46, 73.41],
                    [73.06, 72.61, 72.70, 73.02, 73.19],
                    [72.34, 71.94, 72.29, 71.86, 72.14],
                ]
            ),
        )

    def test_beta_student_baseline_values(self):
        self.assertEqual(plot_hyperparams.VGG8_BASELINE_ACCURACY, 70.542)
        self.assertEqual(plot_hyperparams.RESNET8X4_BASELINE_ACCURACY, 72.50)

    def test_output_names_and_axis_labels(self):
        self.assertEqual(
            plot_hyperparams.OUTPUT_PATHS["VGG8"]["en"].name,
            "inarank_beta_hyperparams.pdf",
        )
        self.assertEqual(
            plot_hyperparams.OUTPUT_PATHS["VGG8"]["zh"].name,
            "inarank_beta_hyperparams_zh.pdf",
        )
        self.assertEqual(
            plot_hyperparams.OUTPUT_PATHS["ResNet8×4"]["en"].name,
            "inarank_beta_hyperparams_resnet8x4.pdf",
        )
        self.assertEqual(
            plot_hyperparams.OUTPUT_PATHS["ResNet8×4"]["zh"].name,
            "inarank_beta_hyperparams_resnet8x4_zh.pdf",
        )
        self.assertEqual(
            plot_batch_hyperparams.OUTPUT_PATHS["VGG8"]["en"].name,
            "batch_size_hyperparams_vgg8_en.pdf",
        )
        self.assertEqual(
            plot_batch_hyperparams.OUTPUT_PATHS["VGG8"]["zh"].name,
            "batch_size_hyperparams_vgg8.pdf",
        )
        self.assertEqual(
            plot_batch_hyperparams.OUTPUT_PATHS["ResNet8×4"]["en"].name,
            "batch_size_hyperparams_resnet8x4_en.pdf",
        )
        self.assertEqual(
            plot_batch_hyperparams.OUTPUT_PATHS["ResNet8×4"]["zh"].name,
            "batch_size_hyperparams_resnet8x4.pdf",
        )
        self.assertEqual(
            plot_hyperparams.AXIS_LABELS["en"],
            (r"Value of $\beta$", "Accuracy (%)"),
        )
        self.assertEqual(
            plot_hyperparams.AXIS_LABELS["zh"],
            (r"$\beta$ 的取值", "准确率（%）"),
        )
        self.assertEqual(
            plot_batch_hyperparams.AXIS_LABELS["en"],
            ("Batch size", "Accuracy (%)"),
        )
        self.assertEqual(
            plot_batch_hyperparams.AXIS_LABELS["zh"],
            ("批大小", "准确率（%）"),
        )

    def test_beta_models_have_separate_baselines_and_outputs(self):
        configs = {config["label"]: config for config in plot_hyperparams.PLOT_CONFIGS}

        self.assertEqual(configs["VGG8"]["baseline_accuracy"], 70.542)
        self.assertEqual(configs["ResNet8×4"]["baseline_accuracy"], 72.50)
        self.assertIs(configs["VGG8"]["output_paths"], plot_hyperparams.OUTPUT_PATHS["VGG8"])
        self.assertIs(
            configs["ResNet8×4"]["output_paths"],
            plot_hyperparams.OUTPUT_PATHS["ResNet8×4"],
        )

    def test_mean_and_sample_standard_deviation(self):
        beta_means, beta_stds = plot_hyperparams.mean_and_std(
            plot_hyperparams.VGG8_ACCURACY_RUNS
        )
        self.assertAlmostEqual(beta_means[0], 71.894)
        self.assertAlmostEqual(beta_stds[0], 0.120954537)

        batch_means, batch_stds = plot_batch_hyperparams.mean_and_std(
            plot_batch_hyperparams.RESNET8X4_ACCURACY_RUNS
        )
        self.assertAlmostEqual(batch_means[2], 73.320)
        self.assertAlmostEqual(batch_stds[2], 0.117473401)

    def test_batch_models_have_separate_baseline_runs_and_outputs(self):
        configs = {config["label"]: config for config in plot_batch_hyperparams.PLOT_CONFIGS}
        for label, counts in (("VGG8", [3, 3, 5, 3, 3]), ("ResNet8×4", [3, 3, 4, 3, 3])):
            config = configs[label]
            self.assertEqual([len(runs) for runs in config["baseline_runs"]], counts)
            self.assertIs(config["output_paths"], plot_batch_hyperparams.OUTPUT_PATHS[label])

    def test_batch_statistics_support_unequal_repeat_counts(self):
        means, stds = plot_batch_hyperparams.mean_and_std([[1, 3], [2, 4, 6]])

        np.testing.assert_allclose(means, [2, 4])
        np.testing.assert_allclose(stds, [np.sqrt(2), 2])

    def test_chinese_style_keeps_times_new_roman_as_the_global_font(self):
        for module in (plot_hyperparams, plot_batch_hyperparams):
            module.configure_style("zh")
            self.assertEqual(module.plt.rcParams["font.family"], ["serif"])
            self.assertEqual(module.plt.rcParams["font.serif"][0], "Times New Roman")

    def test_only_chinese_axis_labels_use_songti_font_fallbacks(self):
        for module in (plot_hyperparams, plot_batch_hyperparams):
            fig, ax = module.plt.subplots()
            module.set_axis_labels(ax, "zh")
            self.assertEqual(
                ax.xaxis.label.get_fontfamily()[:3],
                ["SimSun", "Songti SC", "STSong"],
            )
            self.assertEqual(
                ax.yaxis.label.get_fontfamily()[:3],
                ["SimSun", "Songti SC", "STSong"],
            )
            module.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
