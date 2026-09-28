"""Inputs are pandas DataFrames whose numeric features have already been binned
by main.ID3Preprocessor. Labels are aligned Series: 0=Graduate, 1=Dropout.
Do not replace these TODOs with a library classifier.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TreeNode:
    """Keep training class counts even at internal nodes for unseen values."""

    counts: tuple[int, int]  # (Graduate count, Dropout count)
    prediction: int  # majority class; choose 0 on a tie
    feature: str | None = None  # None means leaf
    children: dict[Any, "TreeNode"] = field(default_factory=dict)


class ID3Classifier:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 10):
        """Root depth is 0. max_depth=0 produces a root leaf.

        min_samples_split limits the parent size, not each child's size.
        fit must reject negative depth or min_samples_split < 2.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: TreeNode | None = None
        self.feature_names_: list[str] = []

    @staticmethod
    def entropy(y: pd.Series | np.ndarray) -> float:
        """TODO: binary Shannon entropy in bits; return 0 for empty/pure y."""
        raise NotImplementedError("Implement ID3Classifier.entropy")

    @staticmethod
    def information_gain(values: pd.Series, y: pd.Series) -> float:
        """TODO: entropy before minus weighted entropy after a multiway split.

        values and y are nonempty and aligned by row/index.
        """
        parent_entropy = ID3Classifier.entropy(y)
        weighted_child_entropy = 0.0
        for value in values.unique():
            mask = values.eq(value)
            # TODO: add (number in this branch / total number) * entropy(y in branch).
            # Use y.loc[mask] so the labels stay aligned with the feature values.
            raise NotImplementedError("Add weighted child entropy")
        return parent_entropy - weighted_child_entropy

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ID3Classifier":
        """TODO: validate input, store schema, recursively build root_, return self.

        Reject empty training data, nonbinary labels, missing values, and
        misaligned/nonunique indexes with ValueError. Retain input column order
        for information-gain ties; use the first tied feature. Do not mutate X/y.
        """
        # TODO: validate hyperparameters, X, y, and their indexes as described above.
        raise NotImplementedError("Validate ID3 training inputs")
        self.feature_names_ = list(X.columns)
        self.root_ = self._build_tree(X, y, self.feature_names_, depth=0)
        return self

    def _build_tree(
        self, X: pd.DataFrame, y: pd.Series, features: list[str], depth: int
    ) -> TreeNode:
        """TODO: compute counts; stop or choose maximum-gain feature and recurse.

        Stop at purity, no features, nonpositive gain (tolerance 1e-12), a depth
        limit, or too few samples. Remove the split feature along each branch.
        """
        n_graduate = int((y == 0).sum())
        n_dropout = int((y == 1).sum())
        node = TreeNode(
            counts=(n_graduate, n_dropout),
            prediction=int(n_dropout > n_graduate),
        )

        # TODO: return node when pure, out of features, at max depth, or too small.
        raise NotImplementedError("Add ID3 stopping conditions")

        best_feature = None
        best_gain = 1e-12
        for feature in features:  # Original column order resolves equal-gain ties.
            gain = self.information_gain(X[feature], y)
            # TODO: update best_feature and best_gain only for strictly larger gain.
            raise NotImplementedError("Select the best split feature")
        if best_feature is None:
            return node

        node.feature = best_feature
        remaining = [feature for feature in features if feature != best_feature]
        for value in X[best_feature].unique():
            mask = X[best_feature].eq(value)
            # TODO: recursively build this child's tree using X.loc[mask],
            # y.loc[mask], remaining, and depth + 1; store it in node.children.
            raise NotImplementedError("Build each ID3 child branch")
        return node

    def _traverse(self, row: pd.Series) -> TreeNode:
        """TODO: return leaf or deepest reached node for an unseen branch value."""
        node = self.root_
        while node.feature is not None:
            value = row[node.feature]
            # TODO: return node when value has no child; otherwise move to child.
            raise NotImplementedError("Follow an ID3 branch or fall back")
        return node

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: one dropout fraction per row, preserving input row order.

        Raise ValueError before fit or for a different column schema. Require
        exactly the fitted columns in the fitted order in all prediction APIs.
        """
        raise NotImplementedError("Implement ID3Classifier.predict_risk")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: majority class at reached node; an exact class tie predicts 0."""
        raise NotImplementedError("Implement ID3Classifier.predict")

    def decision_path(self, row: pd.Series) -> list[dict[str, Any]]:
        """TODO: explain a single transformed row (index equals fitted columns).

        Each visited node contributes a dict with feature, value, counts, and
        fallback. counts is (n_Graduate, n_Dropout). At an internal node value
        is the query value and fallback is True if its branch is unseen. At a
        leaf feature/value are None and fallback is False. Include the terminal
        node, including an internal node where traversal falls back. Translate
        numeric bins to original intervals in the report using the preprocessor.
        """
        # TODO: check fitted state and row schema before starting the walk.
        raise NotImplementedError("Validate an ID3 explanation row")
        path = []
        node = self.root_
        while True:
            # TODO: append a dictionary for node, using its split feature,
            # query value, stored counts, and whether the next branch is unseen.
            raise NotImplementedError("Record each visited ID3 node")
            if node.feature is None or row[node.feature] not in node.children:
                return path
            node = node.children[row[node.feature]]

    def tree_stats(self) -> dict[str, int]:
        """TODO: return depth (root=0), n_nodes, and n_leaves; reject before fit."""
        if self.root_ is None:
            raise ValueError("Fit the tree before requesting statistics")

        def walk(node: TreeNode, depth: int) -> tuple[int, int, int]:
            """Return deepest depth, number of nodes, number of leaves."""
            if not node.children:
                return depth, 1, 1
            # TODO: combine the results of walk(child, depth + 1) for each child.
            raise NotImplementedError("Aggregate ID3 tree statistics")

        depth, n_nodes, n_leaves = walk(self.root_, depth=0)
        return {"depth": depth, "n_nodes": n_nodes, "n_leaves": n_leaves}
