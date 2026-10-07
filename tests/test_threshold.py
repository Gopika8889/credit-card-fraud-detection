"""Unit tests for src.threshold."""

from src.threshold import (
    select_cost_threshold, select_recall_threshold, threshold_table)


def test_threshold_table_counts():
    table = threshold_table([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    row = table[table["threshold"] == 0.4].iloc[0]
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 1)


def test_cost_threshold_on_separable_data():
    table = threshold_table([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    best = select_cost_threshold(table, missed_fraud_cost=100,
                                 false_alarm_cost=5)
    assert best["threshold"] == 0.8
    assert best["cost"] == 0


def test_recall_threshold_meets_target():
    table = threshold_table([0, 1, 0, 1], [0.9, 0.8, 0.3, 0.2])
    best = select_recall_threshold(table, min_recall=0.5)
    assert best["recall"] >= 0.5