# HW1 instructions: Predict early dropouts with ID3 and kNN

Start with the [HW1 introduction](../README.md) for the story behind the task. These docs give the full requirements. Read them in order. Each topic doc tells you **what to do**, **which settings to use**, **what to save and report**, and **how to check your work**.

## Reading order

| # | Doc | What it covers |
| ---: | --- | --- |
| 1 | [Setup and rules](01-setup.md) | Python version, install steps, allowed libraries, data checks, test commands |
| 2 | [Milestones](02-milestones.md) | The order to implement things in, and how to know each step is done |
| 3 | [Data, checkpoints, and splits](03-data-and-splits.md) | Target labels, the three feature sets, the timeline, the one shared split, leakage rules |
| 4 | [ID3 decision tree](04-id3.md) | Binning, entropy, information gain, tree building, fixed tree settings |
| 5 | [kNN and distance design](05-knn.md) | Raw and mixed distances, voting, tuning `k`, redundancy ablation |
| 6 | [Evaluation: the advising list](06-evaluation.md) | 10% budget, metrics, Course-only baseline, the six-row results table and plot |
| 7 | [Explanations and fairness](07-explanations-and-fairness.md) | Three paired student explanations, missed-dropout audit, Gender removal |
| 8 | [Data detective and Enrolled students](08-data-detective.md) | Hidden year, zero grades, the dropout definition, Enrolled risk summaries |
| 9 | [Report and discussion questions](09-report.md) | What the PDF must contain and the eight required questions |
| 10 | [Validate and submit](10-submission.md) | Final checks, what to submit, how to hand in (zip on Canvas), the evidence checklist, the rubric |

Read [DATASET.md](../DATASET.md) before you interpret any column or category code.

## Every run you must do

The labels P1–P6 and A1–A6 are only for cross-referencing in these docs; you don't have to use them in code or filenames. All runs use the **same** training, validation, and test row IDs (see [doc 3](03-data-and-splits.md#split-once)). All test metrics use the same 10% budget and tie-break (see [doc 6](06-evaluation.md)).

**Primary runs (6).** These make up the timeline results table.

| Run | Model | Checkpoint | Features | Settings chosen by |
| --- | --- | --- | ---: | --- |
| P1 | ID3 | Enrollment | 22 | Fixed: `n_bins=3`, `max_depth=3`, `min_samples_split=10` |
| P2 | ID3 | End of semester 1 | 28 | Fixed (same as P1) |
| P3 | ID3 | End of semester 2 | 34 | Fixed (same as P1) |
| P4 | kNN, mixed distance | Enrollment | 22 | `k` tuned on validation Precision@10% |
| P5 | kNN, mixed distance | End of semester 1 | 28 | `k` tuned on validation Precision@10% |
| P6 | kNN, mixed distance | End of semester 2 | 34 | `k` tuned on validation Precision@10% |

**Additional runs.** These are extra and do not replace the six primary runs.

| Run | What | Checkpoint | Details in |
| --- | --- | --- | --- |
| A1 | Course-only baseline, scored on the test rows | Not tied to a checkpoint (it uses only `Course`); drawn as a reference line | [doc 6](06-evaluation.md#course-only-baseline) |
| A2 | Random-selection expectation (a reference value, not a trained model) | Not tied to a checkpoint; drawn as a reference line | [doc 6](06-evaluation.md#random-selection-expectation) |
| A3 | kNN, raw distance (`k` tuned separately) | Semester 2 | [doc 5](05-knn.md#raw-vs-mixed-distance-comparison) |
| A4 | kNN, mixed distance, with `International` and `Curricular units 1st sem (approved)` removed (`k` retuned) | Semester 2 | [doc 5](05-knn.md#reduced-feature-ablation) |
| A5 | ID3 with `Gender` removed (fixed settings) | Enrollment | [doc 7](07-explanations-and-fairness.md#part-d-gender-removal-rerun) |
| A6 | kNN, mixed distance, with `Gender` removed (`k` retuned) | Enrollment | [doc 7](07-explanations-and-fairness.md#part-d-gender-removal-rerun) |

Also score the Enrolled students with the frozen P3 and P6 models, as an exploratory analysis only ([doc 8](08-data-detective.md#enrolled-students)).

## Fixed settings at a glance

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| Split of labeled rows | Stratified 60% train / 20% validation / 20% test (2,178 / 726 / 726): 20% test first, then 25% of the rest for validation |
| Label encoding | Dropout = 1, Graduate = 0; Enrolled set aside |
| ID3 | `n_bins=3`, `max_depth=3`, `min_samples_split=10`, gain tolerance `1e-12` |
| kNN `k` grid | `{3, 5, 11, 21, 51}`; pick the best validation Precision@10%; equal scores → larger `k` |
| kNN voting | Uniform; risk = Dropout neighbors / `k` |
| Advising budget | `B = max(1, floor(0.10 * n))`, which is 72 for validation and for test |
| Ranking tie-break | Risk descending, then row ID ascending |
| Label tie-break (ID3 leaf, kNN vote) | An exact tie predicts 0 (Graduate) |
| Selection metric for tuning | Validation Precision@10% (never accuracy, never test data) |

Report your actual findings. Performance does not have to increase at every checkpoint, and no points depend on beating a performance threshold.
