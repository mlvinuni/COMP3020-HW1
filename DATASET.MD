# Dataset guide: student dropout and academic success

This guide describes the supplied [dataset.csv](dataset.csv). All counts, ranges, category counts, and descriptive statistics below were computed from this local file. Modeling requirements are in [ASSIGNMENT.md](ASSIGNMENT.md).

## Origin and scope

The source dataset concerns undergraduate students at the Polytechnic Institute of Portalegre, Portugal. It combines institutional records with national economic data, covering enrollment years 2008/09–2018/19 and 17 courses. Outcomes are assessed at the normal course duration: three years, or four for Nursing. Transfers between courses or institutions count as dropout. [Source: Realinho et al., *Predicting Student Dropout and Academic Success* (2022)](https://www.mdpi.com/2306-5729/7/11/146).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) provides source documentation and a CC BY 4.0 license. Credit the dataset creators when using or redistributing the data. The local file differs from the version currently documented by UCI; its exact transformation history is not supplied here.

## File structure and loading

| Property | Local file |
| --- | --- |
| Unit of observation | One student record |
| Rows | 4,424 |
| Columns | 35: 34 predictors and `Target` |
| Separator | Comma |
| Encoding | UTF-8 with a byte-order mark (BOM) |
| Empty/missing cells | 0 |
| Exact duplicate rows, including `Target` | 0 |
| Stored student identifier | None |
| Explicit enrollment year or event dates | None |

All predictors contain numeric values, but **numeric storage does not imply numerical meaning**: 17 predictors are treated as nominal categories and 17 as numeric in this assignment. Binary indicators count as nominal features.

```python
import pandas as pd

df = pd.read_csv("dataset.csv", encoding="utf-8-sig")
df.index.name = "row_id"  # original zero-based row position
```

`main.load_dataset` assigns these stable row IDs automatically. Use them for shared splits, explanations, and deterministic tie-breaking. They are not original institutional IDs and must never enter the feature matrix. Retain the exact header spelling `Nacionality` unless you explicitly rename it throughout your code.

## Target labels

| `Target` | Count | Share of all records | Interpretation at the observation horizon |
| --- | ---: | ---: | --- |
| `Graduate` | 2,209 | 49.93% | Completed the course within its normal duration |
| `Dropout` | 1,421 | 32.12% | Classified as dropout under the source definition, including transfers |
| `Enrolled` | 794 | 17.95% | Still enrolled; eventual graduation or dropout is unresolved |

The assignment uses **Dropout vs Graduate**: 3,630 labeled students, with Dropout encoded as 1 and Graduate as 0. In this binary population, dropout prevalence is 39.15%, rather than the 32.12% observed in the complete file. Keep Enrolled students separate for exploratory scoring; do not treat them as graduates or evaluate eventual dropout metrics against their unknown outcomes.

The label describes a later status, not whether a student needs help today. In particular, transferring successfully can still produce a Dropout label. No intervention outcome or reason for leaving is included.

## Feature availability

| Assignment checkpoint | Included information | Number of predictors |
| --- | --- | ---: |
| Enrollment | Non-semester columns | 22 |
| End of semester 1 | Enrollment features plus six semester 1 measures | 28 |
| End of semester 2 | Previous features plus six semester 2 measures | 34 |

These sets are nested. For this assignment, all six semester measures—including credited and enrolled units—become available only at the respective semester checkpoint.

Treating all non-semester features as enrollment information is an **assignment assumption**. The CSV does not establish the exact recording time of administrative statuses such as debt, fee payment, or scholarships, nor the publication time of annual economic figures. These timings need verification for live advising. Students who already left may also have zeros in later semester records; without event dates, the file cannot identify who remained eligible for advising at each checkpoint.

## Column dictionary

The types below match the starter code. “Categories” means the number of distinct observed codes, not the number of levels on an ordered scale. Ranges are observed minima and maxima, not guaranteed limits for future records.

### Non-semester predictors

| Exact column name | Assignment type | Meaning | Observed values |
| --- | --- | --- | --- |
| `Marital status` | Nominal | Marital-status category | 6 categories |
| `Application mode` | Nominal | Admission/application route | 18 categories |
| `Application order` | Numeric, ordinal | Position of this course in the applicant's preferences; lower indicates an earlier choice | 0–9 |
| `Course` | Nominal | Course/program identifier | 17 categories; codes 1–17 |
| `Daytime/evening attendance` | Nominal, binary | Attendance schedule | 0 = evening; 1 = daytime |
| `Previous qualification` | Nominal | Qualification held before admission | 17 categories |
| `Nacionality` | Nominal | Nationality category | 21 categories |
| `Mother's qualification` | Nominal | Mother's education/qualification category | 29 categories |
| `Father's qualification` | Nominal | Father's education/qualification category | 34 categories |
| `Mother's occupation` | Nominal | Mother's occupation category | 32 categories |
| `Father's occupation` | Nominal | Father's occupation category | 46 categories |
| `Displaced` | Nominal, binary | Displaced-student status indicator | 0 or 1 |
| `Educational special needs` | Nominal, binary | Recorded special-needs status | 0 or 1 |
| `Debtor` | Nominal, binary | Recorded debtor status | 0 or 1 |
| `Tuition fees up to date` | Nominal, binary | Recorded tuition-payment status | 0 or 1 |
| `Gender` | Nominal, binary | Gender category as encoded by this dataset | 0 = female; 1 = male |
| `Scholarship holder` | Nominal, binary | Recorded scholarship status | 0 or 1 |
| `Age at enrollment` | Numeric | Age in years when enrolling | 17–70 |
| `International` | Nominal, binary | International-student status | 0 or 1 |
| `Unemployment rate` | Numeric | National unemployment-rate indicator | 7.6–16.2 |
| `Inflation rate` | Numeric | National inflation-rate indicator | −0.8–3.7 |
| `GDP` | Numeric | National GDP-related indicator; not personal income | −4.06–3.51 |

These descriptions summarize the column roles; they do not supply a complete verified local category codebook. For categories with more than two values, retain nominal codes unless you obtain the matching mapping. Verify indicator conventions and economic units against matching documentation before interpreting individual values operationally.

### Semester 1 predictors

Curricular units are course/module units. Unit counts should not automatically be interpreted as ECTS credits. Evaluation counts can exceed the number of enrolled units.

| Exact column name | Assignment type | Meaning | Observed range |
| --- | --- | --- | --- |
| `Curricular units 1st sem (credited)` | Numeric count | Units credited for semester 1 | 0–20 |
| `Curricular units 1st sem (enrolled)` | Numeric count | Units enrolled in semester 1 | 0–26 |
| `Curricular units 1st sem (evaluations)` | Numeric count | Semester 1 evaluation count | 0–45 |
| `Curricular units 1st sem (approved)` | Numeric count | Units approved/passed in semester 1 | 0–26 |
| `Curricular units 1st sem (grade)` | Numeric | Semester 1 aggregate grade | 0–18.875 |
| `Curricular units 1st sem (without evaluations)` | Numeric count | Semester 1 units without evaluations | 0–12 |

### Semester 2 predictors

| Exact column name | Assignment type | Meaning | Observed range |
| --- | --- | --- | --- |
| `Curricular units 2nd sem (credited)` | Numeric count | Units credited for semester 2 | 0–19 |
| `Curricular units 2nd sem (enrolled)` | Numeric count | Units enrolled in semester 2 | 0–23 |
| `Curricular units 2nd sem (evaluations)` | Numeric count | Semester 2 evaluation count | 0–33 |
| `Curricular units 2nd sem (approved)` | Numeric count | Units approved/passed in semester 2 | 0–20 |
| `Curricular units 2nd sem (grade)` | Numeric | Semester 2 aggregate grade | 0–18.571429, rounded |
| `Curricular units 2nd sem (without evaluations)` | Numeric count | Semester 2 units without evaluations | 0–12 |

## Local codes and source-version differences

UCI documents Course identifiers such as 33 and 9500, while this file contains 1–17. Its documented parent-qualification codes also differ from the compact codes found here. UCI additionally lists `Previous qualification (grade)` and `Admission grade`, which are absent from this CSV. [Source: UCI variable documentation](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

Do not infer that the local codes were assigned in the order of an online dictionary. In particular, do not attach a course name or an education level to a local code without confirming the mapping. The CSV alone cannot establish that local parent-education code 24 means Unknown. Unknown or unspecified categories may be represented by codes even though there are no empty cells.

For kNN, arbitrary category-code differences are not valid measures of similarity. Use categorical mismatches plus scaled numeric differences. For ID3, use multiway nominal branches; a threshold such as `Course <= 8.5` imposes an unsupported order. Parent education may be regrouped into meaningful levels only with a verified mapping.

## Observed patterns and interpretation limits

### Zero grades

| Grade feature | Zero records | Mean | Median |
| --- | ---: | ---: | ---: |
| Semester 1 | 718 | 10.641 | 12.286 |
| Semester 2 | 870 | 10.230 | 12.200 |

A zero is a stored value, not a missing cell. It may have several possible interpretations, including failure or absence of assessment. Cross-tab it with enrolled, approved, and without-evaluation units before choosing a treatment. Do not silently replace all zeros with missing values. Such choices affect both numeric distances and training-derived bins.

### Redundant signals

The Pearson correlation between semester 1 and semester 2 approved units is approximately 0.904. The correlation between the stored `Nacionality` codes and `International` is approximately 0.912. The latter depends on arbitrary numeric coding; inspect a category cross-tab rather than interpreting it as a meaningful numeric relationship. Correlated or overlapping features can give repeated influence to a signal in kNN.

### Possible enrollment-cohort proxies

There are 10 distinct `(Unemployment rate, Inflation rate, GDP)` triples. Their repetition suggests shared cohort context, but does not prove an exact year mapping. There is no year column. A random split can distribute the same economic patterns across training and test, so performance on that split does not establish performance for next year's students. Do not infer chronological order from row positions.

### Associations, support, and fairness

Gender, scholarships, debt, and fee status may be associated with outcomes. These associations do not establish causes or justify penalties. Evaluate who is missed by the same global advising list, retain sensitive attributes separately for auditing, and check whether gaps remain when Gender is removed from model inputs. Risk scores estimate patterns in historical records; they do not measure the effect of providing support.

## Handling conventions for this assignment

Keep the CSV unchanged. Split the 3,630 resolved outcomes once, using stratified training/validation/test proportions of 60%/20%/20% and seed 42. Fit all bin boundaries and scaling statistics on training rows only. Reuse the same partitions across checkpoints and models, and reserve Enrolled rows for exploratory analysis. Evaluate the top 10% advising list against the Course-only baseline; never include `Target` or row IDs among predictors. See [ASSIGNMENT.md](ASSIGNMENT.md) for the complete protocol.
