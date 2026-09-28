# Predict Early Dropouts with ID3 and k-nearest neighbors

Start with the [HW1 introduction](README.MD). This document gives the full implementation and evaluation requirements. The six primary runs are ID3 and designed-distance kNN at each of the three checkpoints. The Course-only baseline and four ablations are additional runs; they do not replace the six primary runs. Report your actual findings; performance need not increase at every checkpoint.

## Starter scripts and implementation order

Complete the supplied skeletons; methods marked TODO raise `NotImplementedError` intentionally. Longer methods already show the intended loop, intermediate variables, and where each calculation belongs. Replace each TODO and its nearby `raise NotImplementedError` as you implement that block; leaving a raise in place means the function is still unfinished. Code after a raise is deliberately unreachable until you remove it. Keep the public method signatures and return formats so the supplied tests can assess your implementations. You may add private helpers and extra scripts.

### Follow these milestones

Work from a small passing example toward the complete experiment. The suggested checks use the supplied files and do not require a notebook.

1. **Inspect and split:** run `python main.py --inspect`; implement `prepare_cohorts`, `feature_sets`, and `split_row_ids` in `main.py`. Check the 3,630 resolved labels, 794 Enrolled rows, 22/28/34 feature counts, and 2,178/726/726 split sizes. Save the three ID arrays before training a model.
2. **Make the tree input:** implement `ID3Preprocessor.fit` and `transform`. Fit on training rows, then call `transform` on training, validation, and test rows. Check that a validation value does not change the stored bin boundaries. Run the preprocessor tests in `tests/test_main.py`.
3. **Build a tiny tree first:** implement `entropy`, `information_gain`, and `_build_tree` in `decision_tree.py`. The `_build_tree` skeleton already creates a node and lays out stopping, gain selection, and child recursion; fill those TODO blocks in order. Try a one-column, six-row example before the full CSV. Then add `fit`, `_traverse`, `predict_risk`, `predict`, `decision_path`, and `tree_stats`. Run `python -m unittest discover -s tests -p 'test_decision_tree.py' -v`.
4. **Implement kNN:** complete `KNNPreprocessor` and `KNNClassifier`; the `_distances` and `kneighbors` skeletons separate the distance sums from neighbor sorting. Then run `python -m unittest discover -s tests -p 'test_knn.py' -v`. Check both raw and mixed distances on a few hand-calculated rows.
5. **Score a meeting list:** implement `select_advising_list`, `capacity_metrics`, `group_metrics`, and `CourseBaseline`. On the 726-row validation set, confirm the budget is 72 and that equal risks are ordered by row ID. Run the full test suite.
6. **Run the timeline:** the `run_experiments` skeleton prepares cohorts, splits IDs, and loops over checkpoints. Fill its remaining TODOs to save that split, fit the fixed ID3 model, tune kNN on validation rows, and evaluate the six primary models once on test rows. Add the Course baseline, plot, ablations, explanations, fairness tables, and data-detective summaries described below.
7. **Write the report from saved outputs:** fill in the six-row table and plot first, then answer the eight highlighted discussion questions with the relevant counts and examples. Finally rerun the tests and the full command in section 11 from a clean output directory.

Use this checklist to connect code to the report:

| Work to perform | Evidence to save or show |
| --- | --- |
| Inspect the CSV, separate resolved and Enrolled outcomes, and create one split | Dataset summary; row IDs for train, validation, and test; a three-checkpoint timeline |
| Implement ID3 and kNN | Passing unit tests; each model's label prediction, dropout risk, and explanation interface |
| Validate fixed ID3 settings and tune kNN with validation rows | ID3 settings and validation Precision@10%; selected kNN settings and validation Precision@10% for each required run |
| Evaluate frozen models on test rows | Six-row timeline table, plot, Course-only baseline, and raw/reduced-feature kNN comparisons |
| Audit and interpret predictions | Three paired student explanations, group missed-dropout tables, Gender-removal comparison, and answers to the required discussion questions |
| Examine unresolved outcomes and data anomalies | Enrolled risk summaries and evidence for all three data detective questions |

All feature inputs are pandas DataFrames with stable integer row IDs as their index. Labels are aligned pandas Series, and prediction outputs are NumPy arrays in input row order. Predictions require the exact fitted feature columns and column order. The skeleton docstrings define invalid-input behavior and return formats. Models must return `self` from `fit`, preserve caller inputs, and raise `ValueError` when prediction is attempted before fitting. Training data must be nonempty, without missing values, and have unique aligned indexes and binary labels.

After installing dependencies as described in section 1, run these commands from the assignment directory:

```bash
python main.py --inspect
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_decision_tree.py' -v
python -m unittest discover -s tests -p 'test_knn.py' -v
# After completing all TODOs:
python main.py --data dataset.csv --output results --seed 42
```

`--inspect` works immediately. The full experiment command reports the unfinished TODO and exits with status 2 until implemented. Only the provided CSV/schema checks pass initially; the implementation tests deliberately fail on the untouched skeleton. Do not skip those tests, mark them as expected failures, or change their expected answers to obtain a passing suite. Add checks for any new behavior you introduce. Passing the supplied tests establishes correctness on small examples, not completion of every experiment/report requirement.

To check just the two provided scaffold checks before implementing the algorithms:

```bash
python -m unittest discover -s tests -p 'test_main.py' -k TestProvidedScaffold -v
```

## 1. Set up and inspect the data

Read [DATASET.MD](DATASET.MD) for the column dictionary, local file statistics, and interpretation caveats.

Use **Python 3.10 or newer**; the starter code uses modern type annotations. Create a virtual environment and install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

NumPy, pandas, and Matplotlib may be used for calculations, data handling, and plots. Scikit-learn may be used for splitting, preprocessing, and evaluation only. Do not use its tree, neighbor, nearest-neighbor search, or distance implementations, or another library's ready-made classifiers. Write entropy, information gain, tree construction/traversal, distances, neighbor selection, and risk scoring yourself. Standard-library `unittest` is sufficient for algorithm checks; no notebook is required.

Read the local file rather than downloading a replacement:

```python
import pandas as pd

df = pd.read_csv("dataset.csv", encoding="utf-8-sig")
```

The file is comma-separated and has a UTF-8 byte-order mark. Verify these local facts:

| Item | Expected value |
| --- | ---: |
| Student records | 4,424 |
| Predictors, excluding `Target` | 34 |
| Graduate | 2,209 |
| Dropout | 1,421 |
| Enrolled | 794 |
| Empty cells | 0 |

Retain a stable row ID derived from the original row position for splitting, tie-breaking, and explanations. Never include that ID or `Target` as a predictor. Check duplicates, column types, ranges, and category frequencies; document anomalies rather than silently deleting rows. Preserve the spelling `Nacionality` in code, or explicitly document any rename.

**Category codes are identifiers.** The local `Course` values are 1–17, whereas the [UCI documentation](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) uses other identifiers and includes predictors absent from this CSV. Do not copy its code-to-name mappings directly. Treat local category meanings as unverified unless you find a matching codebook. Course-rate tables may use local codes. Education regrouping is optional and requires a verified mapping; keeping codes nominal is sufficient.

## 2. Define the target and the prediction timeline

For the main supervised task, retain only `Dropout` and `Graduate`: 3,630 students. Encode Dropout as 1 and Graduate as 0. Set aside all 794 `Enrolled` rows before splitting; these students have not completed the course at the observation horizon, so their eventual outcomes are unresolved. Explain how excluding them changes the population to which your reported metrics apply.

The [dataset paper](https://www.mdpi.com/2306-5729/7/11/146) describes outcomes at the normal course duration: three years, or four for Nursing. Course or institution transfers also count as dropout. Its cohort spans enrollment years 2008/09–2018/19. Treat this as a prediction of the dataset's outcome definition, not necessarily departure from higher education.

Create three **nested** feature sets:

| Checkpoint | Predictors allowed | Count |
| --- | --- | ---: |
| Enrollment | All predictors except the 12 semester columns | 22 |
| End of semester 1 | Enrollment predictors plus all six `Curricular units 1st sem (...)` columns | 28 |
| End of semester 2 | Previous set plus all six `Curricular units 2nd sem (...)` columns | 34 |

Each semester contributes `credited`, `enrolled`, `evaluations`, `approved`, `grade`, and `without evaluations`. Even credited/enrolled units are reserved for their respective semester checkpoint in this assignment. Do not use semester 2 information in an earlier model.

Draw a timeline with these three prediction checkpoints and the later label horizon. State an assignment assumption: non-semester columns are available at enrollment. In reality, `Debtor`, `Tuition fees up to date`, and scholarship status can change; their exact measurement times must be verified before operational use. Annual economic figures may also be published after enrollment. Do not claim that this file proves real-time availability.

Students who left before a semester ended may already have an outcome reflected in later academic data. The file lacks event dates and checkpoint eligibility, so the main experiment uses the same retrospective cohort at all checkpoints. Discuss why a live evaluation would instead include students still eligible for advising at each checkpoint.

## 3. Split once and prevent leakage

Use a stratified 60% training / 20% validation / 20% test split of the labeled rows with random seed 42. One way is to reserve 20% for test, then reserve 25% of the remaining 80% for validation. Save the row IDs and reuse exactly these partitions for both algorithms, all checkpoints, and all required ablations.

For this file, the partition sizes are 2,178 training, 726 validation, and 726 test students. Return the `train`, `validation`, and `test` row-ID arrays from `split_row_ids`; select the corresponding feature and label rows with `.loc` so they remain aligned.

Fit bins, scalers, category vocabularies, and any feature-selection decisions on training rows only. Use validation results for model choices. Freeze those choices before final test evaluation; do not change them in response to test performance. For simplicity, keep the final models trained on the original training partition rather than refitting on training plus validation. Never use the Enrolled rows for fitting or tuning.

A random split measures performance among similar historical students. It does not establish performance on a future enrollment year. Keep row IDs out of the feature matrix, and do not infer chronological order from CSV row order.

## 4. Implement ID3

Complete `ID3Classifier` in `decision_tree.py` and `ID3Preprocessor` in `main.py`. The model exposes `fit(X, y)`, `predict(X)`, `predict_risk(X)`, `decision_path(row)`, and `tree_stats()`. Pass already binned features to the classifier. `TreeNode` stores `(Graduate count, Dropout count)` at every node.

Use categorical, multiway ID3 splits. Treat the following as nominal features: `Marital status`, `Application mode`, `Course`, `Daytime/evening attendance`, `Previous qualification`, `Nacionality`, both parents' qualification and occupation columns, `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, and `International`.

Treat `Application order`, `Age at enrollment`, the twelve semester measures, `Unemployment rate`, `Inflation rate`, and `GDP` as numeric for the required implementation. Explain the assumption that application order has meaningful rank.

1. In `ID3Preprocessor.fit`, discretize numeric features using training-only quantile boundaries. The required experiments use **3 bins**. The preprocessor must also accept other valid bin counts because the supplied tests check its general behavior. Use linear interpolation for internal quantiles, remove repeated boundaries and boundaries equal to the observed minimum/maximum, and store no boundaries for a constant column. `transform` numbers bins from 0, with equality entering the upper bin. Outer bins accept values beyond the training range. Apply stored boundaries unchanged to validation/test data. Each numeric feature then becomes a categorical feature for ID3.
2. Compute entropy `H(S) = -sum(p_c * log2(p_c))`, omitting zero-probability terms. For a candidate feature, compute `IG(S, A) = H(S) - sum_v (|S_v| / |S|) * H(S_v)`.
3. Select the feature with greatest information gain; feature ties use original input column order. Create one branch per observed value and remove that feature from consideration along the branch.
4. Stop for pure nodes, exhausted features, nonpositive gain (tolerance `1e-12`), or configured depth/sample limits. Root depth is 0. Store class counts at every node and the majority class at leaves; an exact majority tie predicts 0 (Graduate).
5. If a prediction encounters an unseen category, fall back to the deepest reached node's stored class distribution. Do not invent an ordering between category codes.
6. Return risk as the fraction of training students at the reached leaf (or fallback node) who are Dropout. Return the majority class for `predict`. The risk is an empirical score, not a guaranteed calibrated probability.

**Use one fixed tree configuration for all required ID3 experiments:** `n_bins=3`, `max_depth=3`, and `min_samples_split=10`. There is no required ID3 hyperparameter grid search. Fit the binning preprocessor and tree on training rows, score validation rows once for comparison with kNN, and keep this configuration for the test run and Gender-removal experiment. Report the configuration and the fitted tree's depth, node count, and leaf count. You may try other tree settings as an optional extension, choosing them on validation rows only.

Keep the provided `main.tune_id3` function name for compatibility, but make it a small fitting wrapper: fit `ID3Preprocessor` with 3 bins, transform training and validation rows, fit `ID3Classifier(max_depth=3, min_samples_split=10)`, compute validation Precision@10%, and return `(fitted_preprocessor, fitted_classifier, settings_dict)`. The tiny model tests explicitly use `min_samples_split=2` to exercise splits on small fixtures; keep the experiment setting at 10. `decision_path` returns the visited-node dictionaries defined in its docstring, including the terminal node or fallback. `tree_stats` returns `depth`, `n_nodes`, and `n_leaves`.

With repeated values, a requested 3-bin feature can have fewer effective bins after redundant boundaries are removed. The class still accepts `max_depth=None` for unlimited depth, although required runs use 3. Both `tune_id3` and `tune_knn` return `(fitted_preprocessor, fitted_classifier, settings_dict)`; raw kNN returns `None` as its preprocessor.

### A direct recipe for the recursive tree

Only `ID3Preprocessor` handles continuous values. Once transformed, the classifier sees every column as a category. It does **not** search for numeric thresholds, group categories into subsets, prune branches, or calibrate risk scores.

For a toy input with `Course = [1, 1, 2, 2]` and `y = [0, 0, 1, 1]`, the parent entropy is 1 bit. The two Course branches are pure, so their weighted entropy is 0 and information gain is 1. In contrast, a constant column has information gain 0. Use these examples to check your formulas before recursing.

In `_build_tree(X, y, features, depth)`, follow this sequence:

1. Count zeros and ones. Create a `TreeNode` containing `(zeros, ones)` and prediction `int(ones > zeros)`. This makes a tie predict Graduate. Keep these counts even if the node will have children.
2. Return that node as a leaf if `y` is pure, `features` is empty, `depth` has reached `max_depth`, or `len(y) < min_samples_split`.
3. Calculate information gain for each remaining feature in the original column order. Choose the first feature with maximum gain. Return the leaf if its gain is at most `1e-12`.
4. Set `node.feature` to that feature. For each value observed in the current `X`, select the matching rows from **both** `X` and `y`, remove the chosen feature from the available-feature list for that child, and recursively build `node.children[value]` at `depth + 1`.

In `_traverse`, start at `root_` and follow the row's value for each split feature. If a value has no child, stop at that **current node**. `predict_risk` returns `node.counts[1] / sum(node.counts)`; `predict` returns `node.prediction`. `decision_path` records each visited node, including the final leaf or unseen-value fallback. `tree_stats` can use a separate short recursive walk: count this node, then sum child counts and take the maximum child depth. The skeleton docstrings give the exact output keys.

> **Required discussion question — category codes:** Suppose a tree splits on `Course <= 8.5`. Which courses would go left, and what evidence says they belong together? Explain how your multiway nominal split avoids that assumption. Then discuss what happens when a course is rare or unseen, and what information is lost when numeric measures are binned. Include one decision path or small tree example from your results.

Optional extension: implement entropy-based numeric threshold splits or categorical subset splits and compare with the required binned ID3. Clearly distinguish the extension from classic categorical ID3.

## 5. Implement kNN and design its distance

Complete `KNNClassifier` in `knn.py` and `KNNPreprocessor` in `main.py`. The model exposes `fit(X, y)`, `predict(X)`, `predict_risk(X)`, and `kneighbors(X, n_neighbors=None)`. The last method returns `(distances, row_ids)`, two arrays of shape `(number of queries, requested neighbors)`; these are actual training IDs, not positions. Compute distances and select neighbors yourself. Vectorized NumPy calculations are allowed. Prediction neighbors must come only from training rows. Use the row ID as a deterministic tie-break for equal distances; do not include it in the distance.

Implement and compare both distances below at the semester 2 checkpoint:

- **Naive comparison:** Euclidean distance on the raw integer/numeric predictors, without scaling. Explain why arbitrary codes and large numeric ranges distort it.
- **Designed distance (`distance="mixed"`):** implement the mixed distance below. Use the same nominal/numeric classification as for ID3. Explain how each original categorical variable contributes, how unknown categories are handled, and whether feature groups have comparable influence. A one-hot distance is an optional additional comparison; retain the required mixed mode for the supplied tests.

For this comparison, use the **same semester 2 training, validation, and test row IDs**. Tune `k` separately for raw and mixed distance on validation rows. Save each chosen `k` and its test Precision@10% and Recall@10%; compare at least one student's neighbor ordering under the two distances.

A suitable mixed distance is:

```text
d(x, z) = sqrt(
    sum over numeric j: ((x[j] - z[j]) / training_std[j])**2
    + sum over categorical j: 1[x[j] != z[j]]
)
```

In `KNNPreprocessor`, fit training means and population standard deviations (`ddof=0`), using scale 1 for constant numeric features. Transform numeric features as `(x - training_mean) / training_std`, leaving nominal codes unchanged. The classifier then uses squared differences of the already scaled numeric values plus nominal mismatches; **do not scale again inside the classifier**. An unseen category mismatches each different training category. The raw comparison (`distance="raw"`) bypasses preprocessing entirely. If you add one-hot encoding, fit categories on training rows only and document unseen-category distances.

Use uniform voting. For a student's `k` nearest neighbors, the dropout risk is `number of Dropout neighbors / k`. Return the majority class for label prediction; an exact vote tie predicts 0 (Graduate). Tune `k` from `{3, 5, 11, 21, 51}` on validation Precision@10% in `main.tune_knn`. Break equal validation scores in favor of larger `k`. Tune the naive and designed variants separately. Use the designed distance for the required timeline comparison; tune it at each checkpoint.

The local correlation between semester 1 and semester 2 approved units is about 0.904; between `Nacionality` and `International`, about 0.912. Correlation of arbitrary category codes is not evidence of a meaningful numeric scale. Inspect redundancy through category cross-tabs as well. At semester 2, rerun designed kNN after removing `International` and semester 1 approved units, using the same tuning protocol. Explain how redundant signals affect distance, and why a tree can choose one correlated feature yet is not immune to redundancy.

The reduced-feature comparison removes **both** named columns at once. Fit its scaler on the reduced training matrix, retune `k` on the same validation rows, and evaluate once on the same test rows. Report its selected settings and capacity metrics next to the full designed-distance kNN model.

The supplied education-code example illustrates the issue: an Unknown category can sit numerically between unrelated education categories. Verify the local mappings before claiming that codes 23, 24, and 25 have particular meanings. Nominal mismatch distance must not consider adjacent codes more similar.

## 6. Evaluate an advising list, not accuracy

For a validation or test partition containing `n` students, set `B = max(1, floor(0.10 * n))`. Sort students by descending dropout risk, then ascending stable row ID for score ties, and select exactly `B`. Tree leaves and neighbor votes create many ties; report how many students share the cutoff score and how many of them are selected. The tie-break must not use outcomes or sensitive attributes.

Let `TP_B` be actual dropouts on the list and `D` the total actual dropouts in that partition. Implement:

| Measure | Formula / meaning |
| --- | --- |
| Dropouts reached | `TP_B`: direct answer to the advising question |
| Precision@10% | `TP_B / B`: useful meetings per available slot |
| Recall@10% | `TP_B / D`: fraction of all dropouts reached |
| Missed-dropout rate | `(D - TP_B) / D` |
| Lift@10% | `Precision@10% / (D / n)` |

Report `n`, `B`, and `D`; do not use accuracy to choose the winner. State explicitly that identifying a future dropout does not establish that a meeting would prevent dropout.

The default validation and test budgets are each 72 students. At a fixed budget, the highest possible Recall@10% is `min(1, B / D)` when `D > 0`; even a perfect advising list cannot reach more than `B` dropouts. Interpret recall against this ceiling, rather than expecting it to approach 100%.

Complete `main.select_advising_list` and `main.capacity_metrics`. Wrap each risk array in a pandas Series using the scored DataFrame's row index. The selection helper returns an aligned Boolean mask. The metrics helper returns the key names in its docstring; return `None` for recall, missed rate, and lift when a partition has no actual dropouts. Use `main.group_metrics` for the fairness tables and `main.CourseBaseline` for the baseline below.

Implement a **Course-only baseline**: estimate each course's dropout fraction from training rows. Assign that score to students of the same course; use the overall training dropout fraction for unseen courses. Evaluate it using the identical budget and tie-break. Also show the random-selection expectation: expected dropouts reached `B * D / n`, expected precision `D / n`, and expected lift 1. This expectation uses evaluation prevalence for context, not for constructing predictions.

Compute the full three-label outcome distribution by local course code for descriptive analysis only. Keep those full-data distributions out of model fitting. The paper reports roughly 72% and 70% on-time graduation for Nursing and Social Service, versus 8% for Biofuel Production Technologies and Informatics Engineering. These percentages include Enrolled students in their denominators; binary-only rates will differ. Do not attach course names without a verified local mapping. [Source: dataset paper](https://www.mdpi.com/2306-5729/7/11/146).

Train each required model independently at each checkpoint: **three ID3 models and three designed-distance kNN models**. Produce a six-row test-results table and a two-panel timeline plot of Precision@10% and Recall@10%, with both models and baseline reference lines. Report changes between checkpoints in percentage points. Semester results may bring a large improvement; investigate rather than force this pattern. Discuss the predictive value of approved units, grades, Course, and fee status, alongside the cost of waiting to intervene.

Each test-results row must identify the checkpoint and model and show at least: test `n`, budget `B`, actual dropouts `D`, dropouts reached, Precision@10%, Recall@10%, lift, chosen settings, and cutoff tie counts. Put the Course baseline and ablations in adjacent tables so the six primary rows remain easy to compare. Show both baseline lines on the plot: Course-only and the random-selection expectation. The latter is a reference value, not a trained model.

## 7. Explain the same students both ways

Before inspecting their outcomes, select three test students using a reproducible rule: highest ID3 risk, closest ID3 risk to 0.5, and lowest ID3 risk (break ties by row ID and avoid duplicates). Use the semester 2 models to explain each student side by side:

- ID3: traversed feature/value branches, original numeric bin intervals where applicable, training counts at the reached node, and dropout risk.
- kNN: the five nearest training students under the designed distance, distances, true training outcomes, and a few feature similarities/differences. If the tuned `k` is not 5, show the five-neighbor explanation separately from the vote over the actual `k`; report that full vote and risk too.

Use anonymous row IDs. Answer: Which explanation can guide an advisor's next conversation? Which might a student consider fair? Does similarity imply a cause or a useful intervention? Identify at least one actionable question and one factor an advisor should not treat as destiny.

## 8. Check who gets missed

Use the enrollment models for a required fairness audit, where an early meeting could be useful. For each model's global top-10% test list, report results by `Gender`, `Scholarship holder`, and `Tuition fees up to date`. Gender is coded 1 for male and 0 for female. Report group size, actual dropout count, selected count, dropouts reached, selection rate, and:

```text
group missed-dropout rate = actual group dropouts outside the global list
                           / all actual group dropouts
```

Use `N/A` if a group has no actual dropouts; report denominators and avoid firm conclusions for small groups. Compute the difference in missed-dropout rates between the two groups for each attribute. Selection is always from one global budget; do not allocate 10% separately inside each group.

Remove `Gender` from the predictors and rerun both enrollment models with the same partitions and global budget. Keep the fixed ID3 configuration; retune kNN on validation rows. Retain Gender separately for auditing. Compare overall performance and gender gaps before and after removal. Other features may preserve a gap; removal is not proof of fairness. Discuss whether debtor/fee flags prompt financial support or penalties, how advisors would use them, and why observed associations cannot establish causes.

## 9. Investigate three data mysteries

Answer all three with evidence from the CSV and clearly separate evidence from hypotheses:

1. **Hidden year:** count distinct `(Unemployment rate, Inflation rate, GDP)` triples (the supplied CSV has 10), report their sizes and label distributions, and explain how they may proxy enrollment cohorts. There is no explicit year column, and ten triples do not prove a one-to-one mapping to years. Why might a GDP rule fail next year? Propose a future-year validation design with verified dates. Optional: compare a holdout of whole economic triples; call it a group generalization check, not a verified chronological split.
2. **Zero grades:** report zero counts and grade summaries for both semesters. In this CSV, semester 1 has 718 zeros, mean about 10.641, and median about 12.286; semester 2 has 870 zeros, mean about 10.230, and median 12.2. Cross-tab zero grades against enrolled, approved, and without-evaluation units. Does the evidence support failing grades, absence of assessment, or several mechanisms? Do not assert that zeros are missing without evidence. Explain the consequences for distance and tree bins. Propose a sensitivity comparison retaining zeros versus an explicit no-assessment indicator with a justified grade treatment, fitted on training data only.
3. **What is dropout?** Explain how counting course/institution transfers changes the advising goal. Could a successful transfer be incorrectly treated as an undesirable outcome? What additional labels or event dates would distinguish transfer support, delayed completion, and leaving education?

For Enrolled students, use the frozen semester 2 models to produce exploratory risk summaries and compare their score distributions with those of the labeled test students. **Do not compute dropout precision, recall, or accuracy for Enrolled students or relabel them as Graduate.** State the assumption and limitation of applying models learned only from resolved outcomes to this group.

## 10. Required discussion questions

Answer **every** question in a clearly labeled part of your report. For questions about model performance or group differences, give the relevant result, denominator, and checkpoint; distinguish observations from hypotheses.

1. **Earlier help or later accuracy?** How much do Precision@10% and Recall@10% change from enrollment to semester 1 to semester 2 for each model? Which checkpoint would you recommend for a first advising meeting, and what information becomes available too late to support an earlier meeting? Explain why a high score on this retrospective test does not establish that an intervention will prevent dropout.
2. **Meaningful splits and distances?** Answer the highlighted `Course <= 8.5` question in section 4. Explain why a one-unit difference between category codes is not necessarily small; use a parent-education or Course example. Compare raw and designed kNN neighbor lists for one student and state which distance you would use.
3. **Duplicated signals?** Use the approved-unit and `Nacionality`/`International` evidence, plus your reduced-feature kNN result. Did removing the two columns change the advising list or its performance? Why can repeated information affect kNN more directly than a single tree split?
4. **Useful and fair explanations?** For the same three test students, which parts of the tree path and neighbor evidence could guide an advisor's conversation? Which parts might a student dispute? Identify one actionable question and one factor that should not be treated as destiny. Similarity and correlation do not establish causes.
5. **Who is missed?** Compare missed-dropout rates for Gender, scholarship, and fee-status groups under one global 10% budget. After excluding Gender from the predictors, does the gender gap remain? Explain how related features could carry similar information, and discuss whether debt or fee flags should trigger support or a penalty.
6. **Hidden year?** What do the 10 economic-value triples reveal, and what can they not prove without actual enrollment-year labels? Why might a GDP split fail for future students? Describe a feasible future-cohort validation design.
7. **Zero grades and label definition?** What does the cross-tab evidence suggest about grade 0? How would alternative treatments affect ID3 bins and kNN distances? What does the source definition of dropout do to students who transfer courses or institutions, and what additional data would help distinguish their needs?
8. **Unresolved Enrolled students?** How do their semester 2 risk-score distributions compare with those of labeled test students? Why can you not compute eventual-dropout performance for this group from the supplied CSV?

## 11. Validate and submit

Complete all TODOs and run `python -m unittest discover -s tests -v`; the finished implementation must pass the supplied tests. The tests include small, meaningful algorithm checks:

- Entropy is 0 for a pure node and 1 for an equally mixed binary node; a hand-computed information-gain example selects the expected feature.
- A tiny ID3 example predicts the expected leaf fractions and safely handles an unseen category.
- A hand-computed kNN example verifies distances, neighbor order, votes, and ties; categorical mismatch is unchanged by arbitrary renumbering of codes.
- A ranking example checks the exact budget, tied cutoff, and group missed-dropout denominator. Verify that all risks lie in `[0, 1]` and that the feature snapshots are nested and exclude future columns.

The supplied suite has 46 test methods. It does not execute the fitting/tuning helpers or the full `run_experiments` pipeline. After the tests pass, also run the required entry point and check that saved split IDs, predictions, selections, and report tables agree. Keep the stated fixed ID3 settings; make all kNN tuning choices using validation data before reporting final test results.

Submit:

1. Completed `main.py`, `decision_tree.py`, and `knn.py`, plus any additional helpers. The required entry point is `python main.py --data dataset.csv --output results --seed 42`; it must execute the experiments and save the requested artifacts.
2. Saved split row IDs, fixed ID3 settings, and selected kNN hyperparameters. State tie-breaking and unseen-value behavior.
3. PDF file (with previous exercises) with results table/plot, Course baseline, distance and redundancy comparisons, three paired explanations, fairness tables and Gender-removal comparison, detective answers, Enrolled summaries, and **clearly labeled answers to all eight required discussion questions**.
4. Machine-readable results (CSV or JSON), generated figures, the supplied tests, and any additional checks with instructions to run them. Keep `requirements.txt` and record installed package versions in your README; add dependencies only if justified.

Use filenames and experiment labels that distinguish the six timeline models and additional experiments. Your submission must reproduce every reported metric. A reasonable report length is 6–10 pages excluding code and detailed tables; completeness matters more than length.

### Grading rubric

| Component | Points |
| --- | ---: |
| Correct ID3 implementation, risk scores, and categorical handling | 5 |
| Correct kNN implementation, risk scores, and distance design | 5 |
| Timeline, leakage prevention, shared splits, and tuning protocol | 4 |
| Capacity metrics, Course baseline, results and plots | 4 |
| Paired explanations and interpretation | 3 |
| Fairness audit and Gender-removal experiment | 2 |
| Data detective work, Enrolled analysis, and reproducibility | 2 |
| **Total** | **25** |

Extensions are optional and do not replace required work. No points depend on exceeding an arbitrary performance threshold.
