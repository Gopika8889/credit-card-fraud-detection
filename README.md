# Credit Card Fraud Detection

# Credit Card Fraud Detection

End-to-end machine learning project to detect fraudulent credit card
transactions on a highly imbalanced dataset (0.17% fraud).

## Status

- [x] Stage 1: Setup, data loading, EDA
- [x] Stage 2: Preprocessing, imbalance handling and baselines
- [ ] Stage 3: Modeling and evaluation
- [ ] Stage 4: API
- [ ] Stage 5: Dashboard and deployment

## Dataset

[Kaggle: Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 transactions, 492 frauds. Features: `Time`, `V1-V28` (PCA-anonymized),
`Amount`, `Class`.

## Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

## Project structure

    data/
    notebooks/
    src/
    models/
    app/
    tests/

## Key EDA findings

1. **Severe imbalance:** 492 frauds in 284,807 transactions (0.1727%), about 1 fraud per 578 legit. A classifier that always predicts legit achieves 99.8273% accuracy, showing why accuracy is misleading. Use precision, recall, F1 and PR-AUC.

2. **Clean data:** no missing values were found. There are 1,081 duplicate rows (0.380% of the data), including 1,062 legitimate and 19 fraudulent transactions. These duplicates must be handled before splitting in Stage 2.

3. **Amount is highly skewed:** most transactions are small. The median fraud amount is 9.25 compared with 22.00 for legitimate transactions. There are 1,825 zero-amount transactions. Large legitimate transactions exist, so outliers should not be blindly removed.

4. **Time is a weak, noisy signal:** the hourly fraud rate varies, but low-volume hours can produce unstable rates. `Time` is a relative offset from the first transaction, not a clock time.

5. **A handful of PCA features carry most of the signal:** the strongest correlations with `Class` are observed for features such as V17, V14, V12, V10, V16, V3, V7, V11, V4 and V18. Their fraud and legitimate distributions are visibly different.

6. **V1-V28 are mostly uncorrelated with each other:** this is expected because they are PCA-derived features, so multicollinearity is not a major concern among these features.

7. **Implication for modeling:** use a stratified train/test split, apply scaling and resampling only on training data, and evaluate models using PR-AUC, recall, precision and F1.

## Stage 2: Baseline Results

### Key Findings

1. **1,081 duplicates** were removed; after cleaning there are **473 frauds in 283,726 rows (0.167%)**.

2. `log_amount` reduced the skew of `Amount`, while `RobustScaler` alone did not change the underlying skewness.

3. The **best baseline by PR-AUC** was **Random Forest + none**, with **PR-AUC = 0.839 ± 0.036**.

4. Resampling and class weights generally increased recall but reduced precision. `rf + none` achieved **93.7% precision and 77.5% recall**, while `rf + smote_under` achieved **66.6% precision and 84.4% recall**.

5. ROC-AUC was high across almost all models (**0.965–0.984**), while PR-AUC provided a clearer comparison for the highly imbalanced dataset.

6. Fold-to-fold PR-AUC standard deviation ranged from **0.029 to 0.130**, so small PR-AUC differences should be interpreted cautiously.

### Baseline Results Table

| Model + Strategy | Precision | Recall | F1 | PR-AUC | ROC-AUC | PR-AUC Std |
|---|---:|---:|---:|---:|---:|---:|
| rf + none | 0.937 | 0.775 | 0.846 | 0.839 | 0.976 | 0.036 |
| rf + class_weight | 0.882 | 0.794 | 0.833 | 0.827 | 0.965 | 0.038 |
| rf + smote_under | 0.666 | 0.844 | 0.744 | 0.827 | 0.984 | 0.035 |
| rf + smote | 0.674 | 0.825 | 0.740 | 0.818 | 0.982 | 0.036 |
| rf + undersample | 0.068 | 0.905 | 0.126 | 0.753 | 0.979 | 0.060 |
| logreg + none | 0.857 | 0.611 | 0.711 | 0.751 | 0.976 | 0.039 |
| logreg + smote_under | 0.106 | 0.892 | 0.189 | 0.743 | 0.981 | 0.030 |
| logreg + smote | 0.056 | 0.907 | 0.105 | 0.742 | 0.981 | 0.029 |
| logreg + class_weight | 0.059 | 0.915 | 0.110 | 0.740 | 0.982 | 0.030 |
| logreg + undersample | 0.043 | 0.910 | 0.083 | 0.579 | 0.982 | 0.130 |


## Stage 3: Advanced Models and Evaluation

### Key Findings

1. **XGBoost was the best boosted model by cross-validated PR-AUC**, achieving **0.8497 ± 0.0356**, compared with **0.8458 ± 0.0303** for LightGBM and the Stage 2 Random Forest baseline of **0.839**.

2. For XGBoost, `class_weight` and `none` achieved the same CV PR-AUC of **0.847**, while SMOTE achieved **0.844**. SMOTE took about **1.96× longer** than class weighting for XGBoost in the imbalance comparison.

3. On the held-out test set, the tuned XGBoost achieved **PR-AUC = 0.826**, compared with **0.791** for the Stage 2 Random Forest baseline. XGBoost achieved **96.0% precision, 75.8% recall and 0.847 F1**.

4. The **cost-based threshold was 0.0158**, selected using out-of-fold training predictions. At this threshold, the test set achieved **76.2% precision and 81.1% recall**. The alternative threshold targeting recall ≥90% was **0.0002**, giving **10.8% precision and 88.4% recall**.

5. The cost-based threshold resulted in a test cost of approximately **2,286.9**, compared with **11,614.2** when no transactions are flagged and **283,255.0** when every transaction is flagged.

6. **Isolation Forest performed substantially worse than the supervised models**, with **PR-AUC = 0.104**, showing the importance of labeled fraud examples for this task.

7. SHAP analysis identified **V14, V4, V12, V10, V3, V11, V8, V15, V17 and V19** among the most influential components. These are anonymized PCA features, so no real-world business interpretation is claimed.

### Advanced Model Results

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | CV PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest baseline | 0.972 | 0.737 | 0.838 | 0.791 | 0.971 | 0.839 |
| XGBoost tuned | 0.960 | 0.758 | 0.847 | 0.826 | 0.983 | 0.850 |
| LightGBM tuned | 0.973 | 0.768 | 0.859 | 0.810 | 0.979 | 0.846 |
| Isolation Forest | 0.187 | 0.211 | 0.198 | 0.104 | 0.928 | — |

### Threshold Results

| Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Default 0.5 | 0.960 | 0.758 | 0.847 | 0.826 | 0.983 |
| Cost-based 0.0158 | 0.762 | 0.811 | 0.786 | 0.826 | 0.983 |
| Recall ≥ 90%: 0.0002 | 0.108 | 0.884 | 0.193 | 0.826 | 0.983 |