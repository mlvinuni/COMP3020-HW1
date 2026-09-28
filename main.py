"""Inspect data now: python main.py --inspect
After completing TODOs: python main.py --data dataset.csv --output results
Tests: python -m unittest discover -s tests -v
"""

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from decision_tree import ID3Classifier
from knn import KNNClassifier


SEED = 42
CHECKPOINTS = ("enrollment", "semester1", "semester2")
CATEGORICAL_COLUMNS = (
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
)
SEMESTER_FIELDS = ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")
SEMESTER1_COLUMNS = tuple(f"Curricular units 1st sem ({f})" for f in SEMESTER_FIELDS)
SEMESTER2_COLUMNS = tuple(f"Curricular units 2nd sem ({f})" for f in SEMESTER_FIELDS)
NUMERIC_COLUMNS = (
    "Application order", "Age at enrollment", *SEMESTER1_COLUMNS,
    *SEMESTER2_COLUMNS, "Unemployment rate", "Inflation rate", "GDP",
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Provided I/O helper: preserve original row positions as anonymous IDs."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.index = pd.Index(np.arange(len(df)), name="row_id")
    return df


def prepare_cohorts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """TODO: return (labeled_X, binary_y, enrolled_X), retaining original indexes.

    Drop Target from both feature frames; map Graduate->0 and Dropout->1.
    Reject missing/unknown labels with ValueError. Never reset row IDs.
    Do not mutate df or fit anything using enrolled_X.
    """
    raise NotImplementedError("Implement prepare_cohorts")


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    """TODO: return nested checkpoint predictor lists in original column order.

    Remove Target and row_id if passed. Exclude both semester groups at
    enrollment, and semester 2 at semester1. semester2 contains all predictors.
    """
    raise NotImplementedError("Implement feature_sets")


def split_row_ids(y: pd.Series, seed: int = SEED) -> dict[str, np.ndarray]:
    """TODO: stratified 60/20/20 split; return train/validation/test integer IDs.

    Import sklearn.model_selection.train_test_split inside this function.
    Split actual y.index values, not positions. Reserve 20% test, then 25% of
    remaining rows for validation, using stratify and seed at both steps.
    """
    raise NotImplementedError("Implement split_row_ids")


class ID3Preprocessor:
    def __init__(self, numeric_columns: list[str], n_bins: int = 3):
        self.numeric_columns = list(numeric_columns)
        self.n_bins = n_bins
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "ID3Preprocessor":
        """TODO: learn internal quantile boundaries from training only, return self.

        Use quantiles 1/n_bins through (n_bins-1)/n_bins, linear interpolation.
        Keep unique boundaries strictly between the observed min/max; constant
        features have none. bin_edges_ holds internal boundaries, not infinities.
        Reject empty/missing data, n_bins < 2 or noninteger n_bins, and numeric
        columns absent from X. Store schema. Do not mutate X.
        """
        # TODO: validate X, numeric_columns, and n_bins before learning edges.
        raise NotImplementedError("Validate ID3 preprocessor input")
        self.feature_names_ = list(X.columns)
        quantile_levels = np.arange(1, self.n_bins) / self.n_bins
        self.bin_edges_ = {}
        for column in self.numeric_columns:
            values = X[column].to_numpy(dtype=float)
            # TODO: compute linear-interpolated quantiles, retain unique values
            # strictly between values.min() and values.max(), then store them.
            raise NotImplementedError("Learn training-only ID3 bin edges")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """TODO: copy X; bin numeric columns, preserving nominal values and IDs.

        Use bins numbered from 0; a value equal to a boundary enters the upper
        bin (np.searchsorted(..., side='right')). Outer bins extend to infinity.
        Reuse learned boundaries unchanged. Require fitted schema/order and
        raise ValueError before fit. Reject missing values.
        """
        # TODO: check fitted state, exact column schema, and missing values.
        raise NotImplementedError("Validate ID3 preprocessor query")
        transformed = X.copy(deep=True)
        for column, edges in self.bin_edges_.items():
            # TODO: use np.searchsorted(edges, X[column], side="right")
            # to replace the original numeric values with integer bin IDs.
            raise NotImplementedError("Apply stored ID3 bin edges")
        return transformed


class KNNPreprocessor:
    def __init__(self, numeric_columns: list[str]):
        self.numeric_columns = list(numeric_columns)
        self.means_: dict[str, float] = {}
        self.scales_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "KNNPreprocessor":
        """TODO: training-only mean and population std (ddof=0), return self.

        Replace zero std with 1. Store schema. Reject empty/missing input or
        numeric columns absent from X. Do not scale integer nominal codes.
        """
        # TODO: validate nonempty X, missing values, and numeric_columns.
        raise NotImplementedError("Validate kNN preprocessor input")
        self.feature_names_ = list(X.columns)
        self.means_ = {}
        self.scales_ = {}
        for column in self.numeric_columns:
            values = X[column].to_numpy(dtype=float)
            # TODO: store the training mean and population standard deviation.
            # Use 1 instead of a zero standard deviation.
            raise NotImplementedError("Learn kNN training scale")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """TODO: return a copy with numeric (x-mean)/scale; preserve other columns.

        Reuse fitted means/scales; retain row index and column order. Require
        fitted schema/order and raise ValueError before fit. Reject missing data.
        The raw kNN comparison bypasses this preprocessor entirely.
        """
        # TODO: check fitted state, exact column schema, and missing values.
        raise NotImplementedError("Validate kNN preprocessor query")
        transformed = X.copy(deep=True)
        for column in self.numeric_columns:
            # TODO: replace column with (X[column] - mean) / scale.
            raise NotImplementedError("Scale a kNN numeric column")
        return transformed


def select_advising_list(risks: pd.Series, capacity: float = 0.10) -> pd.Series:
    """TODO: Boolean selection mask aligned to risks.index (integer row IDs).

    B=max(1,floor(capacity*n)); rank risk descending then row ID ascending.
    Reject empty/nonfinite/out-of-[0,1] scores, nonunique IDs, or capacity outside
    (0,1] with ValueError. Do not use labels to select or break ties.
    """
    # TODO: validate scores, row IDs, and capacity before ranking.
    raise NotImplementedError("Validate advising-list inputs")
    budget = max(1, int(np.floor(capacity * len(risks))))
    # TODO: order IDs by (-risk, row_id) and mark the first budget IDs True.
    # Return a Boolean Series indexed in the ORIGINAL risks.index order.
    raise NotImplementedError("Rank and select the advising list")


def capacity_metrics(
    y: pd.Series, risks: pd.Series, capacity: float = 0.10
) -> dict[str, Any]:
    """TODO: select list and return n, budget, dropouts, reached, precision,
    recall, missed_rate, lift, cutoff_score, cutoff_tied, cutoff_selected.

    cutoff_tied counts ALL students sharing the last selected score;
    cutoff_selected counts selected students sharing that score.
    Use None for recall/missed_rate/lift if there are no actual dropouts.
    y and risks must have identical indexes/order and y must be binary.
    """
    # TODO: validate aligned binary labels and risk scores.
    raise NotImplementedError("Validate capacity-metric inputs")
    selected = select_advising_list(risks, capacity)
    n = len(y)
    budget = int(selected.sum())
    dropouts = int(y.sum())
    reached = int(y.loc[selected].sum())
    # TODO: compute precision, recall, missed_rate, lift, and cutoff-tie counts.
    # Recall, missed_rate, and lift are None when dropouts == 0.
    raise NotImplementedError("Calculate advising capacity metrics")
    return {
        "n": n, "budget": budget, "dropouts": dropouts, "reached": reached,
        "precision": precision, "recall": recall, "missed_rate": missed_rate,
        "lift": lift, "cutoff_score": cutoff_score,
        "cutoff_tied": cutoff_tied, "cutoff_selected": cutoff_selected,
    }


def group_metrics(
    y: pd.Series, selected: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """TODO: one row per group value (index name 'group'), using the GLOBAL list.

    Columns: n, dropouts, selected, reached, selection_rate, missed_rate.
    Return NaN missed_rate for groups without dropouts. Require aligned indexes;
    selected is Boolean. Do not select a new list within each group.
    """
    # TODO: validate that y, selected, and groups have identical indexes.
    raise NotImplementedError("Validate group-metric inputs")
    rows = []
    for group in groups.unique():
        mask = groups.eq(group)
        group_y = y.loc[mask]
        group_selected = selected.loc[mask]  # The existing GLOBAL list.
        # TODO: append group, n, dropouts, selected, reached,
        # selection_rate, and missed_rate (NaN if no dropouts).
        raise NotImplementedError("Calculate group missed-dropout metrics")
    return pd.DataFrame(rows).set_index("group")


class CourseBaseline:
    def __init__(self):
        self.course_risks_: dict[Any, float] = {}
        self.global_risk_: float | None = None

    def fit(self, courses: pd.Series, y: pd.Series) -> "CourseBaseline":
        """TODO: training dropout mean by course plus global mean; return self."""
        raise NotImplementedError("Implement CourseBaseline.fit")

    def predict_risk(self, courses: pd.Series) -> np.ndarray:
        """TODO: preserve input order, use global mean for unseen courses.

        Raise ValueError before fit. Never estimate rates on validation/test.
        """
        raise NotImplementedError("Implement CourseBaseline.predict_risk")


def tune_id3(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_validation: pd.DataFrame, y_validation: pd.Series,
) -> tuple[ID3Preprocessor, ID3Classifier, dict[str, Any]]:
    """TODO: fit the fixed ID3 configuration and score it on validation rows.

    Use n_bins=3, max_depth=3, and min_samples_split=10. Return fitted
    preprocessor, fitted model, and settings including validation precision.
    Fit bins and model on training rows only; no test data here.
    """
    numeric = [column for column in X_train if column in NUMERIC_COLUMNS]
    preprocessor = ID3Preprocessor(numeric, n_bins=3).fit(X_train)
    train_binned = preprocessor.transform(X_train)
    validation_binned = preprocessor.transform(X_validation)
    model = ID3Classifier(max_depth=3, min_samples_split=10).fit(train_binned, y_train)
    risks = pd.Series(model.predict_risk(validation_binned), index=X_validation.index)
    # TODO: use capacity_metrics to record validation Precision@10%.
    raise NotImplementedError("Score the fixed ID3 model on validation rows")
    settings = {
        "n_bins": 3, "max_depth": 3, "min_samples_split": 10,
        "validation_precision": validation_precision,
    }
    return preprocessor, model, settings


def tune_knn(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_validation: pd.DataFrame, y_validation: pd.Series,
    distance: str = "mixed",
) -> tuple[KNNPreprocessor | None, KNNClassifier, dict[str, Any]]:
    """TODO: choose k by validation precision; ties prefer larger k.

    Return fitted preprocessor (None for raw), model, and settings. Use only
    available checkpoint numeric/nominal columns. Never scale the raw variant.
    """
    numeric = [column for column in X_train if column in NUMERIC_COLUMNS]
    categorical = [column for column in X_train if column in CATEGORICAL_COLUMNS]
    preprocessor = None
    train_input, validation_input = X_train, X_validation
    if distance == "mixed":
        preprocessor = KNNPreprocessor(numeric).fit(X_train)
        train_input = preprocessor.transform(X_train)
        validation_input = preprocessor.transform(X_validation)
    # Raw mode uses the unscaled frames above.

    best_model = None
    best_precision = -1.0
    best_k = None
    for k in (3, 5, 11, 21, 51):
        model = KNNClassifier(
            k=k, distance=distance, numeric_columns=numeric,
            categorical_columns=categorical,
        ).fit(train_input, y_train)
        risks = pd.Series(model.predict_risk(validation_input), index=X_validation.index)
        # TODO: compute validation Precision@10%. Update the best model when
        # precision improves OR ties and k is larger. Save its k and score.
        raise NotImplementedError("Compare kNN validation candidates")
    return preprocessor, best_model, {
        "k": best_k, "distance": distance,
        "validation_precision": best_precision,
    }


def run_experiments(df: pd.DataFrame, output_dir: Path, seed: int = SEED) -> None:
    """TODO: implement the full experiment pipeline described in ASSIGNMENT.md.

    Suggested sequence:
    1. Inspect anomalies, prepare cohorts, split once, save split IDs.
    2. Fit fixed ID3 and tune kNN at each checkpoint and in required ablations.
    3. Freeze all choices, evaluate held-out test risks and Course baseline.
    4. Save settings, six-row results table, and two-panel timeline plot.
    5. Report raw vs mixed kNN, redundancy, and Gender-removal comparisons.
    6. Save paired explanations and global-budget fairness group tables.
    7. Save detective summaries and exploratory Enrolled scores (no metrics).

    Make output_dir as needed. Import plotting tools inside this function or
    student helpers; keep imports usable without a graphical environment.
    Wrap risk arrays in pd.Series with the scored X.index for evaluation.
    Implement additional helpers as needed; the provided tests do not verify
    every report requirement or tuning/plotting choice.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, enrolled_X = prepare_cohorts(df)
    columns_by_checkpoint = feature_sets(list(X.columns))
    split = split_row_ids(y, seed=seed)
    train_ids, validation_ids, test_ids = (
        split[name] for name in ("train", "validation", "test")
    )
    # TODO: save the split row IDs so every run is reproducible.

    models = {}
    for checkpoint in CHECKPOINTS:
        columns = columns_by_checkpoint[checkpoint]
        X_train = X.loc[train_ids, columns]
        X_validation = X.loc[validation_ids, columns]
        y_train = y.loc[train_ids]
        y_validation = y.loc[validation_ids]
        models[("id3", checkpoint)] = tune_id3(
            X_train, y_train, X_validation, y_validation,
        )
        models[("knn", checkpoint)] = tune_knn(
            X_train, y_train, X_validation, y_validation, distance="mixed",
        )

    # TODO: score the frozen models on X.loc[test_ids] at each checkpoint.
    # TODO: fit CourseBaseline on train_ids and score the same test_ids.
    # TODO: add raw/reduced-feature kNN and Gender-removal ablations.
    # TODO: save settings, test predictions, metrics tables, and timeline plot.
    # TODO: explain three test rows, audit group misses, summarize Enrolled
    # scores, and save the three data-detective summaries from the assignment.
    raise NotImplementedError("Complete experiment evaluation and artifacts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--inspect", action="store_true", help="Inspect the CSV without running TODOs")
    args = parser.parse_args(argv)
    df = load_dataset(args.data)
    if args.inspect:
        print(f"Rows: {len(df)}; predictors: {len(df.columns) - 1}")
        print(df["Target"].value_counts().to_string())
        print(f"Missing cells: {int(df.isna().sum().sum())}")
        return 0
    try:
        run_experiments(df, args.output, args.seed)
    except NotImplementedError as exc:
        print(f"Starter TODO: {exc}. See ASSIGNMENT.md.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
