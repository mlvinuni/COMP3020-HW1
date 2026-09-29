# HW1: Predict Early Student Dropouts

An advising office can meet only a small share of students. In this assignment, you will build two classifiers from scratch—an ID3 decision tree and k-nearest neighbors (kNN)—and use their **dropout risk scores** to decide which students would appear on a meeting list limited to 10% of the cohort.

You will make that list at three points: enrollment, the end of semester 1, and the end of semester 2. Later academic results may improve the ranking, but waiting for them also delays help. The dataset records outcomes at the course's normal completion horizon, so your analysis must distinguish a useful historical prediction from an intervention that could actually help a student.

The main experiment predicts **Dropout versus Graduate**. Students labeled `Enrolled` have unresolved eventual outcomes and are analyzed separately. You will compare both models with a Course-only baseline, examine what their predictions mean for individual students, and check which groups the advising list misses.

## Where to start

1. Read the [assignment docs](docs/README.md) in order. They cover the required methods, step-by-step milestones, experiments, discussion questions, submission requirements, and the [25-point rubric](docs/10-submission.md#grading-rubric).
2. Read [DATASET.md](DATASET.md) before interpreting columns or category codes. The local [dataset.csv](dataset.csv) differs from some online descriptions of this dataset.
3. Follow the setup and testing commands in [Setup and rules](docs/01-setup.md). The starter files contain intentional `TODO` blocks; most tests will fail until you implement them.

| Starter file | Your main work |
| --- | --- |
| [decision_tree.py](decision_tree.py) | ID3 entropy, information gain, multiway branches, risk scores, and decision paths |
| [knn.py](knn.py) | Raw and designed distances, neighbor selection, votes, and risk scores |
| [main.py](main.py) | Cohorts, shared splits, preprocessing, advising metrics, experiments, and saved outputs |
| [tests/](tests/) | Supplied behavioral checks for the models and pipeline |
| [requirements.txt](requirements.txt) | Packages needed to run the assignment |

Your finished work consists of working code, reproducible experiment outputs, and the report described in the [assignment docs](docs/README.md). Those docs give the exact required settings and evidence; this README is only an introduction.
