# Student Performance Prediction

Predicts whether a higher education student will drop out, stay enrolled, or graduate, using data from their first academic year. Built for GSB-S545 Advanced Machine Learning at Cal Poly.

**Team:** Lucas Calaff, Shane Waldron, Tyler Wolf Williams

---

## What this project does

Three outcomes, one model:

- **Dropout** — student leaves before finishing
- **Enrolled** — still attending
- **Graduate** — finishes within the normal timeframe

The point is catching at-risk students early, when there's still time to act.

---

## Repository structure

```
Student_Performance_Prediction/
├── Data/
│   └── Student_Data.csv            # Raw dataset (4,424 students, 36 features)
├── Student_Dropout_&_Success.ipynb # Full pipeline: EDA through model evaluation
└── Predicting Student Academic Outcomes Using Machine Learning.pdf
```

---

## Dataset

4,424 students, 36 features across five categories:

| Category | Examples |
|---|---|
| Demographic | Age at enrollment, gender, nationality, marital status |
| Socioeconomic | Tuition status, scholarship holder, debtor, parents' occupations |
| Application | Application mode, course, admission grade, previous qualification |
| Academic performance | Curricular units enrolled/approved/graded per semester |
| Macroeconomic | GDP, unemployment rate, inflation rate |

**Class breakdown:**

| Class | Count | Share |
|---|---|---|
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 18.0% |

The file is semicolon-delimited: `pd.read_csv('Data/Student_Data.csv', sep=';')`.

---

## Notebook walkthrough

Ten steps, meant to run in order:

| Step | What it does |
|---|---|
| 1 | Imports and setup — seeds fixed for reproducibility |
| 2 | Data loading and initial inspection |
| 3 | EDA — class balance, academic signals, financial signals, correlation heatmap |
| 4 | Feature engineering — 10 new features added |
| 5 | Preprocessing and 70/15/15 stratified split |
| 6 | Stage 1: Logistic Regression baseline |
| 7 | Stage 2: Random Forest and Gradient Boosting |
| 8 | Stage 3: Feedforward Neural Network (128 to 64) |
| 9 | All four models evaluated on the held-out test set |
| 10 | Conclusions and what's left to do |

---

## Feature engineering

Ten features added on top of the original 36:

- **Ratio features** — semester-over-semester change in approved units and grades
- **Interaction terms** — scholarship status with first-semester grade; tuition status with first-semester approved units
- **Early-warning flags** — zero approved units in semester 1, low grades, unpaid tuition
- **Age buckets** — enrollment age split into three groups: under 22, 22-27, over 27

---

## Results (test set)

All models use the same 70/15/15 stratified split. Primary metric is macro-F1.

| Model | Accuracy | Macro-F1 | Dropout recall |
|---|---|---|---|
| Random Forest | 77.9% | 0.721 | 76.5% |
| Neural Network (128-64) | 77.3% | 0.716 | 78.9% |
| Logistic Regression | 74.0% | 0.706 | 72.3% |
| Gradient Boosting | 72.9% | 0.686 | 69.5% |

Random Forest edges out the others. Logistic Regression trails on macro-F1 but has one thing going for it: you can actually read the coefficients and explain a prediction. That matters depending on your audience.

The Enrolled class is where every model breaks down. Enrolled students haven't resolved their outcome yet, so their first-year data looks like a mix of both extremes. Adding more model complexity doesn't fix that.

---

## Key findings

Semester-2 approved units and grades are the strongest signal, along with the engineered approval-rate features. Two independent model families landed on the same top features, and those matched the logistic regression coefficients too. That kind of agreement across methods is usually a good sign.

Tuition payment status is close to a clean separator. Students not current on fees graduate at a much lower rate; scholarship holders skew strongly toward graduation. Debt status shows up as informative but weaker.

Enrolled stayed hard to classify no matter what was tried. The models aren't confused because of a bad hyperparameter — they're confused because the data for those students is genuinely ambiguous.

---

## How to run

Python 3.8+:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
```

Open `Student_Dropout_&_Success.ipynb` and run all cells top to bottom. Seed is fixed at 42. Results should reproduce across runs, with minor floating-point variation possible across hardware.

---

## Report

The PDF has the full written analysis — methodology, EDA discussion, model interpretations, and next steps. Planned additions include a stacking ensemble, Optuna-based neural network tuning, per-class threshold optimization, and a fairness check across gender and international student subgroups.
