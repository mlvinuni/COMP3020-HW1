"""Behavioral tests for student ID3; intentionally fail on the untouched skeleton."""

import unittest

import numpy as np
import pandas as pd

from decision_tree import ID3Classifier


class TestID3Math(unittest.TestCase):
    def test_entropy_empty_pure_and_balanced(self):
        for labels, expected in [([], 0), ([1, 1], 0), ([0, 1], 1)]:
            with self.subTest(labels=labels):
                self.assertAlmostEqual(ID3Classifier.entropy(np.array(labels)), expected)

    def test_entropy_unequal_classes(self):
        # H(3/4, 1/4), independently calculated.
        self.assertAlmostEqual(ID3Classifier.entropy(np.array([0, 0, 0, 1])), 0.8112781244591328)

    def test_information_gain_perfect_split(self):
        values = pd.Series(["a", "a", "b", "b"], index=[7, 4, 9, 2])
        y = pd.Series([0, 0, 1, 1], index=values.index)
        self.assertAlmostEqual(ID3Classifier.information_gain(values, y), 1)

    def test_information_gain_weights_child_sizes(self):
        # Parent H(3/4,1/4); size-1 pure child plus size-3 H(2/3,1/3).
        values = pd.Series(["a", "b", "b", "b"])
        y = pd.Series([0, 0, 0, 1])
        self.assertAlmostEqual(ID3Classifier.information_gain(values, y), 0.12255624891826566)

    def test_constant_feature_has_zero_gain(self):
        self.assertAlmostEqual(
            ID3Classifier.information_gain(pd.Series([7, 7, 7, 7]), pd.Series([0, 0, 1, 1])), 0
        )


class TestID3Model(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"Course": [1, 1, 1, 2, 2, 2]}, index=[10, 30, 20, 50, 40, 60])
        self.y = pd.Series([0, 1, 1, 0, 0, 0], index=self.X.index)

    def test_leaf_risks_labels_and_input_preservation(self):
        original = self.X.copy(deep=True)
        model = ID3Classifier(min_samples_split=2)
        self.assertIs(model.fit(self.X, self.y), model)
        query = pd.DataFrame({"Course": [2, 1]}, index=[101, 100])
        np.testing.assert_allclose(model.predict_risk(query), [0, 2 / 3])
        np.testing.assert_array_equal(model.predict(query), [0, 1])
        pd.testing.assert_frame_equal(self.X, original)

    def test_unseen_root_category_uses_root_distribution(self):
        model = ID3Classifier(min_samples_split=2).fit(self.X, self.y)
        query = pd.DataFrame({"Course": [999]})
        np.testing.assert_allclose(model.predict_risk(query), [1 / 3])
        np.testing.assert_array_equal(model.predict(query), [0])

    def test_unseen_deeper_category_uses_deepest_distribution(self):
        X = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4,
                          "detail": ["p", "p", "p", "q", "p", "p", "q", "q"]})
        y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
        model = ID3Classifier(min_samples_split=2).fit(X, y)
        query = pd.DataFrame({"group": ["a"], "detail": ["unseen"]})
        np.testing.assert_allclose(model.predict_risk(query), [0.25])
        path = model.decision_path(query.iloc[0])
        self.assertEqual([step["feature"] for step in path], ["group", "detail"])
        self.assertEqual(tuple(path[-1]["counts"]), (3, 1))
        self.assertTrue(path[-1]["fallback"])

    def test_decision_path_includes_terminal_leaf(self):
        model = ID3Classifier(min_samples_split=2).fit(self.X, self.y)
        path = model.decision_path(pd.Series({"Course": 1}))
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0]["feature"], "Course")
        self.assertEqual(path[0]["value"], 1)
        self.assertEqual(tuple(path[0]["counts"]), (4, 2))
        self.assertFalse(path[0]["fallback"])
        self.assertIsNone(path[-1]["feature"])
        self.assertIsNone(path[-1]["value"])
        self.assertEqual(tuple(path[-1]["counts"]), (1, 2))
        self.assertFalse(path[-1]["fallback"])

    def test_depth_zero_and_class_tie(self):
        X = pd.DataFrame({"code": [3, 4]})
        model = ID3Classifier(max_depth=0, min_samples_split=2).fit(X, pd.Series([0, 1]))
        np.testing.assert_allclose(model.predict_risk(X), [0.5, 0.5])
        np.testing.assert_array_equal(model.predict(X), [0, 0])
        self.assertEqual(model.tree_stats(), {"depth": 0, "n_nodes": 1, "n_leaves": 1})

    def test_minimum_parent_size_stops_split(self):
        model = ID3Classifier(min_samples_split=10).fit(self.X, self.y)
        np.testing.assert_allclose(model.predict_risk(self.X), np.full(6, 1 / 3))

    def test_zero_gain_stops_instead_of_arbitrary_split(self):
        X = pd.DataFrame({"code": [1, 1, 2, 2]})
        model = ID3Classifier(min_samples_split=2).fit(X, pd.Series([0, 1, 0, 1]))
        self.assertIsNone(model.root_.feature)

    def test_information_gain_tie_uses_input_column_order(self):
        X = pd.DataFrame({"z_first": [1, 1, 2, 2], "a_second": [9, 9, 7, 7]})
        model = ID3Classifier(min_samples_split=2).fit(X, pd.Series([0, 0, 1, 1]))
        self.assertEqual(model.root_.feature, "z_first")
        self.assertEqual(model.tree_stats(), {"depth": 1, "n_nodes": 3, "n_leaves": 2})

    def test_categorical_renumbering_preserves_predictions(self):
        model = ID3Classifier(min_samples_split=2).fit(self.X, self.y)
        renamed = self.X.replace({"Course": {1: 900, 2: -50}})
        other = ID3Classifier(min_samples_split=2).fit(renamed, self.y)
        np.testing.assert_allclose(model.predict_risk(self.X), other.predict_risk(renamed))

    def test_unfitted_prediction_rejected(self):
        with self.assertRaises(ValueError):
            ID3Classifier().predict_risk(self.X)

    def test_invalid_fit_inputs_rejected(self):
        cases = [
            (ID3Classifier(), self.X.iloc[:0], self.y.iloc[:0]),
            (ID3Classifier(), self.X, self.y.replace({1: 2})),
            (ID3Classifier(), self.X, self.y.iloc[::-1]),
            (ID3Classifier(max_depth=-1), self.X, self.y),
            (ID3Classifier(min_samples_split=1), self.X, self.y),
        ]
        for model, X, y in cases:
            with self.subTest(model=model, rows=len(X)):
                with self.assertRaises(ValueError):
                    model.fit(X, y)

    def test_prediction_schema_rejected(self):
        model = ID3Classifier(min_samples_split=2).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.predict_risk(self.X.rename(columns={"Course": "other"}))


if __name__ == "__main__":
    unittest.main()
