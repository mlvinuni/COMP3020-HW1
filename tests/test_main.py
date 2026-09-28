"""Tests for provided I/O and student preprocessing/evaluation contracts."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from main import (
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SEMESTER1_COLUMNS, SEMESTER2_COLUMNS,
    CourseBaseline, ID3Preprocessor, KNNPreprocessor, capacity_metrics,
    feature_sets, group_metrics, load_dataset, prepare_cohorts,
    select_advising_list, split_row_ids,
)


DATASET = Path(__file__).resolve().parents[1] / "dataset.csv"


class TestProvidedScaffold(unittest.TestCase):
    """These tests pass before students implement the TODOs."""

    def test_local_csv_schema_and_bom(self):
        df = load_dataset(DATASET)
        self.assertEqual(df.shape, (4424, 35))
        self.assertIn("Marital status", df.columns)
        self.assertEqual(df.index.name, "row_id")
        self.assertEqual(df.index.tolist(), list(range(4424)))
        self.assertEqual(df["Target"].value_counts().to_dict(),
                         {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794})
        self.assertEqual(int(df.isna().sum().sum()), 0)

    def test_feature_type_constants_cover_local_predictors(self):
        df = load_dataset(DATASET)
        numeric, nominal = set(NUMERIC_COLUMNS), set(CATEGORICAL_COLUMNS)
        self.assertFalse(numeric & nominal)
        self.assertEqual(numeric | nominal, set(df.columns) - {"Target"})


class TestCohortsAndSplits(unittest.TestCase):
    def test_cohorts_preserve_ids_and_do_not_leak_target(self):
        df = pd.DataFrame({"Course": [1, 2, 3, 4],
                           "Target": ["Graduate", "Enrolled", "Dropout", "Graduate"]},
                          index=pd.Index([10, 40, 20, 70], name="row_id"))
        original = df.copy(deep=True)
        X, y, enrolled = prepare_cohorts(df)
        self.assertEqual(X.index.tolist(), [10, 20, 70])
        self.assertEqual(y.index.tolist(), X.index.tolist())
        self.assertEqual(y.tolist(), [0, 1, 0])
        self.assertEqual(enrolled.index.tolist(), [40])
        self.assertEqual(X.columns.tolist(), ["Course"])
        self.assertEqual(enrolled.columns.tolist(), ["Course"])
        pd.testing.assert_frame_equal(df, original)

    def test_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            prepare_cohorts(pd.DataFrame({"Course": [1], "Target": ["Unknown"]}))

    def test_nested_timeline_excludes_future_and_identifiers(self):
        cols = load_dataset(DATASET).columns.tolist() + ["row_id"]
        sets = feature_sets(cols)
        self.assertEqual(set(sets), {"enrollment", "semester1", "semester2"})
        self.assertEqual([len(sets[k]) for k in ["enrollment", "semester1", "semester2"]], [22, 28, 34])
        self.assertLess(set(sets["enrollment"]), set(sets["semester1"]))
        self.assertLess(set(sets["semester1"]), set(sets["semester2"]))
        self.assertEqual(set(sets["semester1"]) - set(sets["enrollment"]), set(SEMESTER1_COLUMNS))
        self.assertEqual(set(sets["semester2"]) - set(sets["semester1"]), set(SEMESTER2_COLUMNS))
        for features in sets.values():
            self.assertEqual(features, [c for c in cols if c in set(features)])
            self.assertNotIn("Target", features)
            self.assertNotIn("row_id", features)

    def test_shared_split_sizes_disjointness_stratification_and_seed(self):
        y = pd.Series([0] * 50 + [1] * 50, index=np.arange(100) * 7 + 1000)
        parts = split_row_ids(y, seed=42)
        repeated = split_row_ids(y, seed=42)
        self.assertEqual(set(parts), {"train", "validation", "test"})
        self.assertEqual([len(parts[k]) for k in ["train", "validation", "test"]], [60, 20, 20])
        all_ids = np.concatenate(list(parts.values()))
        self.assertEqual(len(set(all_ids)), 100)
        self.assertEqual(set(all_ids), set(y.index))
        for name, ids in parts.items():
            np.testing.assert_array_equal(ids, repeated[name])
            self.assertAlmostEqual(y.loc[ids].mean(), 0.5)
        different = split_row_ids(y, seed=19)
        self.assertNotEqual(set(parts["test"]), set(different["test"]))


class TestPreprocessing(unittest.TestCase):
    def test_id3_training_quantiles_and_out_of_range_values(self):
        train = pd.DataFrame({"age": [0.0, 10.0, 20.0, 30.0], "code": [24, 25, 23, 24]},
                             index=[8, 2, 7, 4])
        original = train.copy(deep=True)
        prep = ID3Preprocessor(["age"], n_bins=3)
        self.assertIs(prep.fit(train), prep)
        np.testing.assert_allclose(prep.bin_edges_["age"], [10, 20])
        query = pd.DataFrame({"age": [-100.0, 10.0, 20.0, 100.0], "code": [999, 25, 23, 24]},
                             index=[104, 102, 101, 103])
        transformed = prep.transform(query)
        self.assertEqual(transformed["age"].tolist(), [0, 1, 2, 2])
        pd.testing.assert_series_equal(transformed["code"], query["code"])
        self.assertEqual(transformed.index.tolist(), query.index.tolist())
        np.testing.assert_allclose(prep.bin_edges_["age"], [10, 20])  # no query refit
        pd.testing.assert_frame_equal(train, original)
        self.assertEqual(query["age"].tolist(), [-100, 10, 20, 100])

    def test_id3_constant_and_duplicate_quantiles(self):
        train = pd.DataFrame({"constant": [4] * 5, "zero_heavy": [0, 0, 0, 0, 10]})
        prep = ID3Preprocessor(list(train.columns), n_bins=5).fit(train)
        self.assertEqual(len(prep.bin_edges_["constant"]), 0)
        edges = prep.bin_edges_["zero_heavy"]
        self.assertEqual(len(edges), len(np.unique(edges)))
        self.assertTrue(np.all((edges > 0) & (edges < 10)))
        self.assertTrue((prep.transform(train)["constant"] == 0).all())

    def test_knn_population_scaling_uses_training_only(self):
        train = pd.DataFrame({"x": [0.0, 2.0], "constant": [5.0, 5.0], "code": [24, 25]},
                             index=[9, 3])
        original = train.copy(deep=True)
        prep = KNNPreprocessor(["x", "constant"])
        self.assertIs(prep.fit(train), prep)
        self.assertAlmostEqual(prep.means_["x"], 1)
        self.assertAlmostEqual(prep.scales_["x"], 1)
        self.assertAlmostEqual(prep.scales_["constant"], 1)
        transformed = prep.transform(train)
        np.testing.assert_allclose(transformed["x"], [-1, 1])
        np.testing.assert_allclose(transformed["constant"], [0, 0])
        pd.testing.assert_series_equal(transformed["code"], train["code"])
        query = pd.DataFrame({"x": [100.0], "constant": [7.0], "code": [999]}, index=[42])
        result = prep.transform(query)
        np.testing.assert_allclose(result["x"], [99])
        np.testing.assert_allclose(result["constant"], [2])
        self.assertEqual(result["code"].tolist(), [999])
        self.assertEqual(result.index.tolist(), [42])
        self.assertAlmostEqual(prep.means_["x"], 1)
        pd.testing.assert_frame_equal(train, original)

    def test_transform_before_fit_and_schema_change_rejected(self):
        X = pd.DataFrame({"x": [1.0, 2.0], "code": [1, 2]})
        for prep in [ID3Preprocessor(["x"]), KNNPreprocessor(["x"])]:
            with self.subTest(preprocessor=type(prep).__name__):
                with self.assertRaises(ValueError):
                    prep.transform(X)
                prep.fit(X)
                with self.assertRaises(ValueError):
                    prep.transform(X[["code", "x"]])

    def test_missing_training_values_rejected(self):
        X = pd.DataFrame({"x": [1.0, np.nan]})
        for prep in [ID3Preprocessor(["x"]), KNNPreprocessor(["x"])]:
            with self.subTest(preprocessor=type(prep).__name__):
                with self.assertRaises(ValueError):
                    prep.fit(X)


class TestAdvisingEvaluation(unittest.TestCase):
    def setUp(self):
        self.ids = [20, 10, 30, 40, 50, 60, 70, 80, 90, 100]
        self.risks = pd.Series([0.9, 0.9, 0.2, 0.1, 0, 0, 0, 0, 0, 0], index=self.ids)
        self.y = pd.Series([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=self.ids)

    def test_exact_budget_cutoff_tie_and_metrics(self):
        selected = select_advising_list(self.risks)
        self.assertTrue(pd.api.types.is_bool_dtype(selected))
        self.assertEqual(selected.index.tolist(), self.ids)
        self.assertEqual(selected[selected].index.tolist(), [10])
        metrics = capacity_metrics(self.y, self.risks)
        for key, expected in {"n": 10, "budget": 1, "dropouts": 2, "reached": 1,
                              "precision": 1, "recall": 0.5, "missed_rate": 0.5,
                              "lift": 5, "cutoff_score": 0.9, "cutoff_tied": 2,
                              "cutoff_selected": 1}.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(metrics[key], expected)

    def test_budget_floors_fraction_and_selects_at_least_one(self):
        risks = pd.Series(np.linspace(0, 1, 19), index=np.arange(19))
        self.assertEqual(int(select_advising_list(risks).sum()), 1)
        self.assertEqual(int(select_advising_list(risks, capacity=0.20).sum()), 3)
        self.assertEqual(int(select_advising_list(risks.iloc[:3]).sum()), 1)

    def test_no_actual_dropouts_has_undefined_recall_and_lift(self):
        metrics = capacity_metrics(self.y * 0, self.risks)
        self.assertEqual(metrics["precision"], 0)
        for name in ["recall", "missed_rate", "lift"]:
            self.assertIsNone(metrics[name])

    def test_group_missed_rate_uses_global_selection_and_dropout_denominator(self):
        y = pd.Series([1, 1, 0, 1, 0], index=[5, 2, 8, 4, 9])
        selected = pd.Series([True, False, False, False, False], index=y.index)
        groups = pd.Series(["a", "a", "a", "b", "c"], index=y.index)
        table = group_metrics(y, selected, groups)
        self.assertEqual(table.index.name, "group")
        for group, values in {
            "a": {"n": 3, "dropouts": 2, "selected": 1, "reached": 1,
                  "selection_rate": 1 / 3, "missed_rate": 0.5},
            "b": {"n": 1, "dropouts": 1, "selected": 0, "reached": 0,
                  "selection_rate": 0, "missed_rate": 1},
        }.items():
            for key, value in values.items():
                with self.subTest(group=group, metric=key):
                    self.assertAlmostEqual(table.loc[group, key], value)
        self.assertTrue(pd.isna(table.loc["c", "missed_rate"]))

    def test_course_baseline_training_only_and_unseen_course(self):
        courses = pd.Series([1, 1, 2, 2, 2], index=[5, 1, 7, 2, 9])
        y = pd.Series([0, 1, 1, 1, 1], index=courses.index)
        model = CourseBaseline()
        self.assertIs(model.fit(courses, y), model)
        query = pd.Series([2, 999, 1], index=[60, 40, 20])
        np.testing.assert_allclose(model.predict_risk(query), [1, 0.8, 0.5])
        np.testing.assert_allclose(model.predict_risk(query), [1, 0.8, 0.5])

    def test_invalid_risks_and_capacity_rejected(self):
        for risks in [pd.Series([], dtype=float), pd.Series([np.nan]),
                      pd.Series([1.1]), pd.Series([-0.1]), pd.Series([0.1, 0.2], index=[1, 1])]:
            with self.subTest(risks=risks.tolist()):
                with self.assertRaises(ValueError):
                    select_advising_list(risks)
        for capacity in [0, 1.1]:
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    select_advising_list(self.risks, capacity=capacity)

    def test_misaligned_evaluation_indexes_rejected(self):
        with self.assertRaises(ValueError):
            capacity_metrics(self.y.iloc[::-1], self.risks)
        with self.assertRaises(ValueError):
            group_metrics(self.y, self.risks > 0.5, pd.Series([0] * 10))


if __name__ == "__main__":
    unittest.main()
