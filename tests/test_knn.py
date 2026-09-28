"""Hand-computed kNN distance, voting, and row-ID tests."""

import unittest

import numpy as np
import pandas as pd

from knn import KNNClassifier


class TestKNN(unittest.TestCase):
    def test_mixed_distance_and_training_row_ids(self):
        X = pd.DataFrame({"a": [3.0, 0.0], "b": [4.0, 0.0], "code": [10, 999]}, index=[80, 20])
        y = pd.Series([1, 0], index=X.index)
        model = KNNClassifier(k=2, numeric_columns=["a", "b"], categorical_columns=["code"])
        self.assertIs(model.fit(X, y), model)
        query = pd.DataFrame({"a": [0.0], "b": [0.0], "code": [10]}, index=[100])
        distances, ids = model.kneighbors(query)
        np.testing.assert_allclose(distances, [[1, 5]])
        np.testing.assert_array_equal(ids, [[20, 80]])
        np.testing.assert_allclose(model.predict_risk(query), [0.5])
        np.testing.assert_array_equal(model.predict(query), [0])

    def test_raw_distance_uses_code_magnitude(self):
        X = pd.DataFrame({"a": [3.0], "b": [4.0], "code": [11]}, index=[7])
        model = KNNClassifier(k=1, distance="raw").fit(X, pd.Series([1], index=X.index))
        query = pd.DataFrame({"a": [0.0], "b": [0.0], "code": [10]})
        distances, ids = model.kneighbors(query)
        np.testing.assert_allclose(distances, [[np.sqrt(26)]])
        np.testing.assert_array_equal(ids, [[7]])

    def test_equal_distance_tie_uses_ids_not_labels_or_input_order(self):
        X = pd.DataFrame({"x": [-1.0, 1.0]}, index=[20, 10])
        y = pd.Series([1, 0], index=X.index)
        model = KNNClassifier(k=1, numeric_columns=["x"]).fit(X, y)
        query = pd.DataFrame({"x": [0.0]})
        distances, ids = model.kneighbors(query, n_neighbors=2)
        np.testing.assert_allclose(distances, [[1, 1]])
        np.testing.assert_array_equal(ids, [[10, 20]])
        np.testing.assert_array_equal(model.predict(query), [0])
        reversed_model = KNNClassifier(k=1, numeric_columns=["x"]).fit(X.iloc[::-1], y.iloc[::-1])
        np.testing.assert_array_equal(reversed_model.kneighbors(query)[1], [[10]])

    def test_uniform_votes_use_tuned_k_not_five(self):
        X = pd.DataFrame({"x": [0.0, 2.0, 4.0]}, index=[10, 3, 8])
        y = pd.Series([1, 0, 1], index=X.index)
        model = KNNClassifier(k=3, numeric_columns=["x"]).fit(X, y)
        query = pd.DataFrame({"x": [2.0, 100.0]}, index=[400, 200])
        risks = model.predict_risk(query)
        self.assertEqual(risks.shape, (2,))
        np.testing.assert_allclose(risks, [2 / 3, 2 / 3])
        np.testing.assert_array_equal(model.predict(query), [1, 1])

    def test_k_one_exact_training_match(self):
        X = pd.DataFrame({"x": [0.0, 2.0, 4.0]}, index=[10, 3, 8])
        y = pd.Series([1, 0, 1], index=X.index)
        model = KNNClassifier(k=1, numeric_columns=["x"]).fit(X, y)
        # Self-match is allowed when explicitly querying training rows in a unit
        # test. Experiment validation/test partitions must be separate.
        np.testing.assert_array_equal(model.predict(X), y.to_numpy())

    def test_mixed_nominal_distance_invariant_to_renumbering(self):
        X = pd.DataFrame({"code": [1, 2, 3]}, index=[4, 2, 8])
        y = pd.Series([0, 1, 1], index=X.index)
        query = pd.DataFrame({"code": [1, 88]})  # second category is unseen
        model = KNNClassifier(k=2, categorical_columns=["code"]).fit(X, y)
        mapping = {1: 900, 2: -1, 3: 72, 88: 5000}
        other = KNNClassifier(k=2, categorical_columns=["code"]).fit(X.replace(mapping), y)
        d1, ids1 = model.kneighbors(query)
        d2, ids2 = other.kneighbors(query.replace(mapping))
        np.testing.assert_allclose(d1, d2)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_allclose(model.predict_risk(query), other.predict_risk(query.replace(mapping)))

    def test_training_input_is_copied(self):
        X = pd.DataFrame({"x": [0.0, 100.0]}, index=[1, 2])
        y = pd.Series([1, 0], index=X.index)
        model = KNNClassifier(k=1, numeric_columns=["x"]).fit(X, y)
        X.loc[1, "x"] = 1000
        y.loc[1] = 0
        np.testing.assert_allclose(model.predict_risk(pd.DataFrame({"x": [0.0]})), [1])

    def test_unfitted_model_rejected(self):
        with self.assertRaises(ValueError):
            KNNClassifier(k=1).kneighbors(pd.DataFrame({"x": [0.0]}))

    def test_invalid_fit_arguments_rejected(self):
        X = pd.DataFrame({"x": [0.0, 1.0]})
        y = pd.Series([0, 1])
        cases = [
            KNNClassifier(k=0, distance="raw"),
            KNNClassifier(k=3, distance="raw"),
            KNNClassifier(k=1.5, distance="raw"),
            KNNClassifier(k=1, distance="invalid"),
            KNNClassifier(k=1),  # mixed mode does not cover x
            KNNClassifier(k=1, numeric_columns=["x"], categorical_columns=["x"]),
        ]
        for model in cases:
            with self.subTest(k=model.k, distance=model.distance):
                with self.assertRaises(ValueError):
                    model.fit(X, y)

    def test_invalid_labels_or_indexes_rejected(self):
        X = pd.DataFrame({"x": [0.0, 1.0]}, index=[4, 2])
        for y in [pd.Series([0, 2], index=X.index), pd.Series([0, 1], index=[2, 4])]:
            with self.subTest(labels=y.tolist()):
                with self.assertRaises(ValueError):
                    KNNClassifier(k=1, distance="raw").fit(X, y)

    def test_invalid_neighbor_count_and_schema_rejected(self):
        X = pd.DataFrame({"x": [0.0, 1.0]})
        model = KNNClassifier(k=1, distance="raw").fit(X, pd.Series([0, 1]))
        for count in [0, 3]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    model.kneighbors(X, n_neighbors=count)
        with self.assertRaises(ValueError):
            model.predict_risk(X.rename(columns={"x": "other"}))


if __name__ == "__main__":
    unittest.main()
