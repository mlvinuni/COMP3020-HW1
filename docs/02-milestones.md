# 2. Milestones: the order to work in

**Goal:** build from a small passing example up to the full experiment. Every check below uses the supplied files; no notebook is needed.

## How the starter code works

- Complete the supplied skeletons. Methods marked `TODO` raise `NotImplementedError` on purpose.
- Longer methods already show the intended loop, the intermediate variables, and where each calculation belongs.
- As you implement each block, replace the `TODO` and the `raise NotImplementedError` next to it. If a raise is still there, that function is unfinished. Code after a raise cannot run until you remove the raise; this is deliberate.
- **Keep the public method signatures and return formats** so the supplied tests can check your work. You may add private helpers and extra scripts.

## Rules every model and helper must follow

- Feature inputs are pandas DataFrames indexed by stable integer row IDs. Labels are aligned pandas Series. Prediction outputs are NumPy arrays in the same row order as the input.
- Prediction requires exactly the fitted feature columns, in the same column order.
- `fit` returns `self`.
- Do not modify the caller's inputs.
- Raise `ValueError` if prediction is attempted before fitting.
- Training data must be nonempty, have no missing values, and have unique, aligned indexes and binary labels.
- The skeleton docstrings define the exact invalid-input behavior and return formats. Follow them.

## Milestones

### Milestone 1: Inspect and split

| | |
| --- | --- |
| Implement | `prepare_cohorts`, `feature_sets`, `split_row_ids` in `main.py` |
| Run | `python main.py --inspect`, then `python -m unittest discover -s tests -p 'test_main.py' -k TestCohortsAndSplits -v` |
| Done when | The cohort/split tests pass. `--inspect` shows only raw label counts, so also call your functions on the real CSV (a few `print` lines are enough) and confirm 3,630 resolved labels; 794 Enrolled rows; feature counts 22 / 28 / 34; split sizes 2,178 / 726 / 726 |
| Save | The three row-ID arrays (train, validation, test), **before** training any model |

Spec: [Data, checkpoints, and splits](03-data-and-splits.md).

### Milestone 2: Make the tree input

| | |
| --- | --- |
| Implement | `ID3Preprocessor.fit` and `ID3Preprocessor.transform` in `main.py` |
| How | Fit on training rows only. Then call `transform` on the training, validation, and test rows. |
| Done when | Transforming validation data does not change the stored bin boundaries, and `python -m unittest discover -s tests -p 'test_main.py' -k TestPreprocessing -v` passes |

Spec: [ID3, step A](04-id3.md#a-bin-numeric-features-id3preprocessor).

### Milestone 3: Build a tiny tree first

| | |
| --- | --- |
| Implement, part 1 | `entropy`, `information_gain`, `_build_tree` in `decision_tree.py`. `_build_tree` already creates a node and lays out stopping, gain selection, and child recursion; fill those TODO blocks in order. |
| Try | A one-column, six-row example before the full CSV |
| Implement, part 2 | `fit`, `_traverse`, `predict_risk`, `predict`, `decision_path`, `tree_stats` |
| Run | `python -m unittest discover -s tests -p 'test_decision_tree.py' -v` |
| Done when | All decision-tree tests pass |

Spec: [ID3 decision tree](04-id3.md).

### Milestone 4: Implement kNN

| | |
| --- | --- |
| Implement | `KNNPreprocessor` in `main.py`; `KNNClassifier` in `knn.py`. The `_distances` skeleton handles the distance sums; the `kneighbors` skeleton handles neighbor sorting. |
| Run | `python -m unittest discover -s tests -p 'test_knn.py' -v` |
| Done when | All kNN tests pass, **and** you have checked both raw and mixed distances by hand on a few rows |

Spec: [kNN and distance design](05-knn.md).

### Milestone 5: Score a meeting list

| | |
| --- | --- |
| Implement | `select_advising_list`, `capacity_metrics`, `group_metrics`, `CourseBaseline` in `main.py` |
| Done when | On the 726-row validation set the budget is 72 and equal risks are ordered by row ID. The full suite `python -m unittest discover -s tests -v` passes. |

Spec: [Evaluation](06-evaluation.md).

### Milestone 6: Run the timeline

| | |
| --- | --- |
| Implement | `tune_id3`, `tune_knn`, and the remaining TODOs in `run_experiments`. The `run_experiments` skeleton already prepares cohorts, splits IDs, and loops over checkpoints. |
| Do, in order | 1. Save the split. 2. Fit the fixed ID3 model. 3. Tune kNN on validation rows. 4. Evaluate the six primary models **once** on test rows. 5. Add the Course baseline, the plot, the ablations, the explanations, the fairness tables, and the data-detective summaries. |
| Run | `python main.py --data dataset.csv --output results --seed 42` |
| Done when | Every item in the [run matrix](README.md#every-run-you-must-do) has saved outputs |

Specs: docs [3](03-data-and-splits.md) to [8](08-data-detective.md).

### Milestone 7: Write the report from saved outputs

1. Fill in the six-row results table and the plot first.
2. Answer the eight [required discussion questions](09-report.md#required-discussion-questions), citing the relevant counts and examples.
3. Delete the output directory, rerun the tests, and rerun the full command (see [Validate and submit](10-submission.md)).

## From work to evidence

The [evidence checklist](10-submission.md#evidence-checklist) lists, topic by topic, everything your code must save and your report must show. Keep it open while you work through milestones 6 and 7.

Next: [Data, checkpoints, and splits](03-data-and-splits.md).
