"""For distance='mixed', main.KNNPreprocessor scales numeric columns and leaves
nominal codes unchanged. For distance='raw', pass unscaled original features.
DataFrame indexes are unique integer training row IDs, never distance features.
"""

import numpy as np
import pandas as pd


class KNNClassifier:
    def __init__(
        self,
        k: int = 5,
        distance: str = "mixed",
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ):
        self.k = k
        self.distance = distance
        self.numeric_columns = list(numeric_columns or [])
        self.categorical_columns = list(categorical_columns or [])
        self.X_train_: pd.DataFrame | None = None
        self.y_train_: pd.Series | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KNNClassifier":
        """TODO: validate, copy/store training rows, return self.

        Reject k outside [1, len(X)], noninteger k, unknown distance modes,
        empty/missing data, nonbinary y, or misaligned/nonunique indexes.
        Mixed numeric/categorical lists must be disjoint and cover X exactly.
        Raw mode uses every column numerically and ignores those lists.
        Raise ValueError for invalid inputs. No scaler is fitted here.
        """
        # TODO: perform every validation listed above before storing data.
        raise NotImplementedError("Validate kNN training inputs")
        self.feature_names_ = list(X.columns)
        self.X_train_ = X.copy(deep=True)
        self.y_train_ = y.copy(deep=True)
        return self

    def _distances(self, row: pd.Series) -> np.ndarray:
        """TODO: distance from one query to each stored row in training order.

        Raw: sqrt(sum((x_j-z_j)**2)) across every feature.
        Mixed: sqrt(sum numeric squared differences + sum nominal mismatches).
        Numeric columns are already scaled; do not scale them again here.
        Nominal mismatches contribute 0 or 1 regardless of code magnitude.
        """
        if self.X_train_ is None:
            raise ValueError("Fit kNN before calculating distances")
        squared = np.zeros(len(self.X_train_), dtype=float)
        if self.distance == "raw":
            for column in self.feature_names_:
                # TODO: add squared numeric difference from row[column].
                raise NotImplementedError("Add raw squared differences")
        else:
            for column in self.numeric_columns:
                # TODO: add squared differences; values are already scaled.
                raise NotImplementedError("Add scaled numeric differences")
            for column in self.categorical_columns:
                # TODO: add 1 for a mismatch, 0 for a match.
                raise NotImplementedError("Add categorical mismatches")
        return np.sqrt(squared)

    def kneighbors(
        self, X: pd.DataFrame, n_neighbors: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """TODO: return (distances, row_ids), both shaped (len(X), requested_k).

        Default requested_k is self.k; an explicit value may differ from self.k.
        Sort by ascending distance, then ascending integer training row ID.
        Do not use labels to break ties. Return IDs, not array positions.
        Reject requested_k outside [1, n_train] or a different feature schema.
        Raise ValueError before fitting. Keep query input row order.
        """
        # TODO: check fitted state, query schema, and requested neighbor count.
        raise NotImplementedError("Validate kNN neighbor query")
        requested_k = self.k if n_neighbors is None else n_neighbors
        distances = np.empty((len(X), requested_k), dtype=float)
        row_ids = np.empty((len(X), requested_k), dtype=int)
        train_ids = self.X_train_.index.to_numpy()
        for query_number, (_, row) in enumerate(X.iterrows()):
            all_distances = self._distances(row)
            # TODO: sort by distance, then train_ids (np.lexsort is useful).
            # Store the first requested_k distances and IDs in these arrays.
            raise NotImplementedError("Select nearest training row IDs")
        return distances, row_ids

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: mean of binary labels among self.k nearest training rows."""
        raise NotImplementedError("Implement KNNClassifier.predict_risk")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: majority vote; exactly 50% dropout votes predicts class 0."""
        raise NotImplementedError("Implement KNNClassifier.predict")
