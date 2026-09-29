# 5. kNN and distance design

**Goal:** implement k-nearest neighbors with a distance that treats category codes as labels rather than numbers. Then show why this matters by comparing it against a naive distance.

Code: `KNNPreprocessor` and `tune_knn` in `main.py`; `KNNClassifier` in `knn.py`.

The model exposes `fit(X, y)`, `predict(X)`, `predict_risk(X)`, and `kneighbors(X, n_neighbors=None)`.

## Required settings

| Setting | Value |
| --- | --- |
| Column types | Same nominal/numeric lists as [ID3](04-id3.md#column-types) |
| Distance for the timeline runs | `distance="mixed"` |
| `k` grid | `{3, 5, 11, 21, 51}` |
| Tuning score | Validation Precision@10% |
| `k` tie-break | Equal validation scores → the **larger** `k` |
| Voting | Uniform; risk = Dropout neighbors / `k` |
| Vote tie | An exact tie predicts 0 (Graduate) |
| Distance tie-break | Smaller training row ID first. Never put the row ID into the distance. |

## What to do

### A. Two distance modes

**Raw (`distance="raw"`)**: Euclidean distance on the raw integer/numeric predictors, with no scaling. Skip preprocessing entirely.

**Mixed (`distance="mixed"`)**:

```text
d(x, z) = sqrt(
    sum over numeric j: ((x[j] - z[j]) / training_std[j])**2
    + sum over categorical j: 1[x[j] != z[j]]
)
```

1. In `KNNPreprocessor.fit`, compute training means and **population** standard deviations (`ddof=0`). Use scale 1 for a constant numeric feature.
2. In `KNNPreprocessor.transform`, turn each numeric feature into `(x - training_mean) / training_std`. Leave nominal codes unchanged.
3. In the classifier, add the squared differences of the already-scaled numeric values to the count of nominal mismatches. **Do not scale again inside the classifier.**
4. An unseen category counts as a mismatch against every training category that differs from it. Adjacent codes are **not** more similar than distant ones.

A one-hot distance is an optional extra comparison. If you add it, fit its categories on training rows only and document how unseen categories affect distance. Keep the required `mixed` mode for the supplied tests.

### B. Neighbors, votes, and risk

1. `_distances(row)` returns the distance from one query row to every training row.
2. `kneighbors(X, n_neighbors=None)` returns `(distances, row_ids)`. Both arrays have shape `(number of queries, requested neighbors)`. Sort by ascending distance, then ascending training row ID. Return the **actual training row IDs**, not positions. Neighbors come only from training rows. You may vectorize with NumPy, but compute the distances and select the neighbors yourself.
3. `predict_risk(X)` = number of Dropout neighbors / `k`.
4. `predict(X)` = the majority class; an exact tie predicts 0.

### C. Tune `k`: `tune_knn`

For each `k` in the grid:

1. Fit on the training rows. For `mixed`, first fit `KNNPreprocessor` on the training rows and transform the training and validation rows. `raw` uses the unscaled frames.
2. Compute validation Precision@10%.
3. Keep the best `k`. On an equal score, prefer the larger `k`.

Return `(fitted_preprocessor, fitted_classifier, settings_dict)`. For `raw`, the preprocessor is `None`.

Tune the mixed distance **separately at each checkpoint** for primary runs P4–P6.

## Raw vs mixed distance comparison

Run A3 in the [run matrix](README.md#every-run-you-must-do).

1. At the semester 2 checkpoint, use the **same** training, validation, and test row IDs.
2. Tune `k` separately for `raw` and for `mixed` on the validation rows.
3. Evaluate each once on the test rows.
4. Pick at least one test student and compare their neighbor ordering under the two distances.

Explain in the report:

- why arbitrary codes and large numeric ranges distort the raw distance;
- how each original categorical variable contributes to the mixed distance, how unknown categories are handled, and whether the feature groups have comparable influence.

> **Note: example.** The supplied education-code example shows the problem: an Unknown category can sit numerically between two unrelated education categories. Verify the local mappings before claiming that codes 23, 24, and 25 mean anything in particular.

## Reduced-feature ablation

Run A4 in the [run matrix](README.md#every-run-you-must-do).

In this CSV, the correlation between semester 1 and semester 2 approved units is about 0.904. Between `Nacionality` and `International` it is about 0.912. A correlation between arbitrary category codes does not show a meaningful numeric scale, so also inspect the redundancy with category cross-tabs.

1. At semester 2, remove **both** `International` and `Curricular units 1st sem (approved)` at once.
2. Fit the scaler on the reduced **training** matrix.
3. Retune `k` on the same validation rows.
4. Evaluate once on the same test rows.
5. Report its settings and capacity metrics next to the full mixed-distance model (P6).

Explain how redundant signals affect a distance, and why a tree, which can choose just one of two correlated features, is still not immune to redundancy.

## Save / report

See the [doc 5 items in the evidence checklist](10-submission.md#knn-doc-5).

## Check yourself

- `python -m unittest discover -s tests -p 'test_knn.py' -v`
- Work out raw and mixed distances by hand for a few rows and compare them with your code.
- Renumbering category codes arbitrarily must not change the mixed distances.

## Discuss in report

Feeds [question 2](09-report.md#q2-meaningful-splits-and-distances) (distances) and [question 3](09-report.md#q3-duplicated-signals) (duplicated signals).

Next: [Evaluation: the advising list](06-evaluation.md).
