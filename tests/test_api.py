"""API tests: health, auth, predict, batch, alerts, transactions."""

import io

import pandas as pd
import pytest

from tests.conftest import TEST_THRESHOLD


def _csv(rows) -> str:
    return pd.DataFrame(rows).to_csv(index=False)


def _upload(client, text, headers, name="tx.csv"):
    return client.post("/predict/batch", headers=headers,
                       files={"file": (name, text, "text/csv")})


def test_health_is_public(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["threshold"] == pytest.approx(TEST_THRESHOLD)
    assert body["model_version"].startswith("xgb-")


def test_login_wrong_password_401(client):
    resp = client.post("/auth/login",
                       data={"username": "alice", "password": "nope"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_credentials"


def test_predict_without_token_401(client, sample_rows):
    resp = client.post("/predict", json=sample_rows["fraud"])
    assert resp.status_code == 401


def test_alerts_without_token_401(client):
    resp = client.get("/alerts")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_alerts_forbidden_for_user_role_403(client, user_headers):
    resp = client.get("/alerts", headers=user_headers)
    assert resp.status_code == 403


def test_predict_fraud_creates_alert(client, user_headers, sample_rows):
    resp = client.post("/predict", json=sample_rows["fraud"],
                       headers=user_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_fraud"] is True
    assert body["risk_level"] in ("Medium", "High")
    assert len(body["top_factors"]) == 5
    assert body["alert_id"] is not None


def test_predict_legit_has_no_alert(client, user_headers, sample_rows):
    body = client.post("/predict", json=sample_rows["legit"],
                       headers=user_headers).json()
    assert body["is_fraud"] is False
    assert body["risk_level"] == "Low"
    assert body["alert_id"] is None


def test_predict_negative_amount_422(client, user_headers, sample_rows):
    payload = dict(sample_rows["legit"], Amount=-5.0)
    resp = client.post("/predict", json=payload, headers=user_headers)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_predict_missing_field_422(client, user_headers, sample_rows):
    payload = {k: v for k, v in sample_rows["legit"].items() if k != "V1"}
    resp = client.post("/predict", json=payload, headers=user_headers)
    assert resp.status_code == 422
    fields = [d["field"] for d in resp.json()["error"]["details"]]
    assert "V1" in fields


def test_batch_missing_column_422(client, user_headers, sample_rows):
    row = {k: v for k, v in sample_rows["legit"].items() if k != "V5"}
    resp = _upload(client, _csv([row]), user_headers)
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "missing_columns"
    assert error["details"]["missing"] == ["V5"]


def test_batch_reports_bad_row_and_downloads_flagged(
        client, user_headers, sample_rows):
    bad = dict(sample_rows["legit"], Amount=-5.0)
    rows = [sample_rows["fraud"], sample_rows["legit"], bad,
            sample_rows["fraud"]]
    resp = _upload(client, _csv(rows), user_headers)
    assert resp.status_code == 200
    summary = resp.json()
    assert (summary["n_rows"], summary["n_scored"],
            summary["n_rejected"], summary["n_flagged"]) == (4, 3, 1, 2)
    assert summary["rejected_rows"][0]["row"] == 4  # header + 3rd data row
    download = client.get(summary["download_url"], headers=user_headers)
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/csv")
    assert len(pd.read_csv(io.StringIO(download.text))) == 2


def test_batch_file_too_large_413(client, user_headers, sample_rows,
                                  monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    resp = _upload(client, _csv([sample_rows["legit"]]), user_headers)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "file_too_large"


def test_alert_review_updates_stats(
        client, user_headers, analyst_headers, sample_rows):
    client.post("/predict", json=sample_rows["fraud"], headers=user_headers)
    alerts = client.get("/alerts", headers=analyst_headers).json()
    assert alerts["total"] == 1
    alert_id = alerts["items"][0]["id"]

    resp = client.patch(
        f"/alerts/{alert_id}", headers=analyst_headers,
        json={"status": "confirmed_fraud", "notes": "card stolen"})
    assert resp.status_code == 200
    assert resp.json()["reviewed_by"] == "bob"
    assert client.get("/alerts", headers=analyst_headers).json()["total"] == 0

    stats = client.get("/stats", headers=user_headers).json()
    assert stats["analyst_precision"] == 1.0
    forbidden = client.patch(f"/alerts/{alert_id}", headers=user_headers,
                             json={"status": "false_positive"})
    assert forbidden.status_code == 403


def test_transactions_filter_by_label(client, user_headers, sample_rows):
    client.post("/predict", json=sample_rows["fraud"], headers=user_headers)
    client.post("/predict", json=sample_rows["legit"], headers=user_headers)
    everything = client.get("/transactions", headers=user_headers).json()
    flagged = client.get("/transactions?is_fraud=true",
                         headers=user_headers).json()
    low = client.get("/transactions?risk_level=Low",
                     headers=user_headers).json()
    assert (everything["total"], flagged["total"], low["total"]) == (2, 1, 1)