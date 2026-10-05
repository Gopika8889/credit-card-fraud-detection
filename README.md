# Credit Card Fraud Detection

End-to-end machine learning project to detect fraudulent credit card
transactions on a highly imbalanced dataset (0.17% fraud).

## Status
- [x] Stage 1: Setup, data loading, EDA
- [ ] Stage 2: Preprocessing and imbalance handling
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
    data/ notebooks/ src/ models/ app/ tests/

## Key EDA findings
1. **Severe imbalance:** 492 frauds in 284,807 transactions (0.1727%), about 1 fraud per 578 legit. A classifier that always predicts legit achieves 99.8273% accuracy, showing why accuracy is misleading. Use precision, recall, F1 and PR-AUC.
2. **Clean data:** no missing values were found. There are 1,081 duplicate rows (0.380% of the data), including 1,062 legitimate and 19 fraudulent transactions. These duplicates must be handled before splitting in Stage 2.
3. **Amount is highly skewed:** most transactions are small. The median fraud amount is 9.25 compared with 22.00 for legitimate transactions. There are 1,825 zero-amount transactions. Large legitimate transactions exist, so outliers should not be blindly removed.
4. **Time is a weak, noisy signal:** the hourly fraud rate varies, but low-volume hours can produce unstable rates. `Time` is a relative offset from the first transaction, not a clock time.
5. **A handful of PCA features carry most of the signal:** the strongest correlations with `Class` are observed for features such as V17, V14, V12, V10, V16, V3, V7, V11, V4 and V18. Their fraud and legitimate distributions are visibly different.
6. **V1-V28 are mostly uncorrelated with each other:** this is expected because they are PCA-derived features, so multicollinearity is not a major concern among these features.
7. **Implication for modeling:** use a stratified train/test split, apply scaling and resampling only on training data, and evaluate models using PR-AUC, recall, precision and F1.