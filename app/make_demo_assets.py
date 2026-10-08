"""Create demo samples and the PR-curve image from the final model.

Usage (project root): python -m app.make_demo_assets
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.model_service import ModelService  # noqa: E402
from src.config import RANDOM_STATE  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.evaluate import plot_pr_curves  # noqa: E402
from src.preprocessing import RAW_FEATURE_COLUMNS, remove_duplicates  # noqa: E402
from src.split import stratified_split  # noqa: E402

N_FRAUD, N_LEGIT = 50, 50


def main() -> None:
    """Write models/demo_samples.csv and models/pr_curve.png."""
    models_dir = get_settings().models_dir
    service = ModelService.load(models_dir)
    df, _ = remove_duplicates(load_data(verbose=False))
    _, X_test, _, y_test = stratified_split(df)

    scores = service.predict_batch(X_test, explain=False)["fraud_probability"]
    ax = plot_pr_curves(
        y_test, {service.model_version: scores.to_numpy()})
    ax.set_title("Precision-Recall curve (held-out test set)")
    ax.figure.savefig(models_dir / "pr_curve.png", dpi=150,
                      bbox_inches="tight")

    frauds = X_test[y_test == 1].sample(
        min(N_FRAUD, int(y_test.sum())), random_state=RANDOM_STATE)
    legit = X_test[y_test == 0].sample(N_LEGIT, random_state=RANDOM_STATE)
    samples = pd.concat([
        frauds[RAW_FEATURE_COLUMNS].assign(Class=1),
        legit[RAW_FEATURE_COLUMNS].assign(Class=0)])
    samples.to_csv(models_dir / "demo_samples.csv", index=False)
    print(f"Saved {len(samples)} demo rows and pr_curve.png to {models_dir}")


if __name__ == "__main__":
    main()