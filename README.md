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
(To be filled in after running `notebooks/01_eda.ipynb`.)