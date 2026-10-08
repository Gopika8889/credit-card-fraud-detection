"""Streamlit dashboard. Talks to the FastAPI backend over HTTP only."""

import io
import os
from typing import Any, Dict, Optional

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
RAW_FIELDS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
GREEN, AMBER, RED = "#2e7d32", "#f9a825", "#c62828"
RISK_COLORS = {"Low": GREEN, "Medium": AMBER, "High": RED}

st.set_page_config(page_title="Fraud Detection", layout="wide")


class ApiError(Exception):
    """Raised when the backend returns an error or is unreachable."""


def api(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Call the backend with the stored bearer token; raise ApiError."""
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.request(method, f"{API_URL}{path}", headers=headers,
                             timeout=60.0, **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}: {exc}") from exc
    if resp.status_code >= 400:
        if resp.status_code == 401 and token:
            st.session_state.pop("token", None)
        try:
            message = resp.json()["error"]["message"]
        except Exception:  # non-JSON error
            message = resp.text[:200]
        raise ApiError(f"{resp.status_code}: {message}")
    return resp


@st.cache_data(ttl=300)
def get_health() -> Dict[str, Any]:
    """Public /health (threshold and risk cut-offs for the gauge)."""
    return httpx.get(f"{API_URL}/health", timeout=10).json()


# ---------------------------------------------------------------- login
def login_sidebar() -> bool:
    """Render login/logout; return True when signed in."""
    with st.sidebar:
        st.title("Fraud Detection")
        if st.session_state.get("token"):
            st.caption(f"Signed in as **{st.session_state['username']}** "
                       f"({st.session_state['role']})")
            if st.button("Log out"):
                for key in ("token", "username", "role", "last_result"):
                    st.session_state.pop(key, None)
                st.rerun()
            return True
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                try:
                    data = api("POST", "/auth/login",
                               data={"username": username,
                                     "password": password}).json()
                    st.session_state.update(
                        token=data["access_token"], username=username,
                        role=data["role"])
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
        return False


# ------------------------------------------------------- page: check one
def load_sample(kind: str) -> None:
    """Button callback: fill the form with a random demo transaction."""
    try:
        sample = api("GET", "/demo/sample", params={"kind": kind}).json()
        for name, value in sample.items():
            st.session_state[f"f_{name}"] = float(value)
        st.session_state.pop("last_result", None)
    except ApiError as exc:
        st.session_state["sample_error"] = str(exc)


def gauge(probability: float, threshold: float, high: float) -> go.Figure:
    """Probability gauge with Low / Medium / High bands."""
    return go.Figure(go.Indicator(
        mode="gauge+number", value=probability,
        number={"valueformat": ".3f"},
        title={"text": "Fraud probability"},
        gauge={
            "axis": {"range": [0, 1]}, "bar": {"color": "#263238"},
            "steps": [{"range": [0, threshold], "color": GREEN},
                      {"range": [threshold, high], "color": AMBER},
                      {"range": [high, 1], "color": RED}],
            "threshold": {"line": {"color": "black", "width": 4},
                          "value": threshold}}))


def shap_chart(factors: list) -> go.Figure:
    """Horizontal bar chart of SHAP contributions."""
    df = pd.DataFrame(factors).iloc[::-1]
    df["label"] = df["feature"] + " = " + df["value"].round(2).astype(str)
    df["direction"] = (df["shap_value"] > 0).map(
        {True: "Raises risk", False: "Lowers risk"})
    return px.bar(df, x="shap_value", y="label", color="direction",
                  orientation="h",
                  color_discrete_map={"Raises risk": RED,
                                      "Lowers risk": GREEN},
                  labels={"shap_value": "SHAP value (log-odds)",
                          "label": ""})


def render_result(result: Dict[str, Any]) -> None:
    """Show gauge, risk level and SHAP factors."""
    health = get_health()
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(result["fraud_probability"],
                              health["threshold"],
                              health["high_risk_cutoff"]),
                        use_container_width=True)
        color = RISK_COLORS[result["risk_level"]]
        verdict = "FLAGGED as fraud" if result["is_fraud"] else "Not flagged"
        st.markdown(
            f"<h3 style='color:{color}'>{result['risk_level']} risk - "
            f"{verdict}</h3>", unsafe_allow_html=True)
        st.caption(f"Transaction `{result['transaction_id']}` - model "
                   f"{result['model_version']} - threshold "
                   f"{result['threshold']:.3f}")
    with right:
        st.subheader("Top contributing features")
        st.plotly_chart(shap_chart(result["top_factors"]),
                        use_container_width=True)
        st.caption(
            "SHAP values show how each feature pushed this model's score. "
            "V1-V28 are anonymized PCA components, so this shows which "
            "components the model used, not a business reason. "
            "log_amount and Time are shown in scaled units.")


def page_check() -> None:
    """Page 1: score a single transaction."""
    st.header("Check a transaction")
    for name in RAW_FIELDS:
        st.session_state.setdefault(f"f_{name}", 0.0)
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button("Load random fraud sample", on_click=load_sample,
              args=("fraud",))
    c2.button("Load random legit sample", on_click=load_sample,
              args=("legit",))
    error = st.session_state.pop("sample_error", None)
    if error:
        st.error(error)

    a, b = st.columns(2)
    a.number_input("Time (seconds)", key="f_Time", min_value=0.0,
                   format="%.2f")
    b.number_input("Amount", key="f_Amount", min_value=0.0, format="%.2f")
    with st.expander("PCA features V1-V28"):
        cols = st.columns(4)
        for i in range(1, 29):
            cols[(i - 1) % 4].number_input(
                f"V{i}", key=f"f_V{i}", format="%.6f")

    if st.button("Score transaction", type="primary"):
        payload = {n: float(st.session_state[f"f_{n}"]) for n in RAW_FIELDS}
        try:
            st.session_state["last_result"] = api(
                "POST", "/predict", json=payload).json()
        except ApiError as exc:
            st.error(str(exc))
    if st.session_state.get("last_result"):
        render_result(st.session_state["last_result"])


# ------------------------------------------------------ page: batch upload
def page_batch() -> None:
    """Page 2: score a CSV and download flagged rows."""
    st.header("Batch upload")
    st.caption("CSV with columns: Time, V1-V28, Amount (extra columns such "
               "as Class are ignored).")
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload and st.button("Score file", type="primary"):
        try:
            summary = api("POST", "/predict/batch",
                          files={"file": (upload.name, upload.getvalue(),
                                          "text/csv")}).json()
            flagged_csv = api("GET", summary["download_url"]).content
            st.session_state["batch"] = (summary, flagged_csv)
        except ApiError as exc:
            st.session_state.pop("batch", None)
            st.error(str(exc))

    if "batch" not in st.session_state:
        return
    summary, flagged_csv = st.session_state["batch"]
    m = st.columns(4)
    m[0].metric("Rows in file", summary["n_rows"])
    m[1].metric("Scored", summary["n_scored"])
    m[2].metric("Rejected", summary["n_rejected"])
    m[3].metric("Flagged", summary["n_flagged"],
                f"{summary['flagged_rate']:.2%}")
    risk = pd.Series(summary["risk_counts"]).rename_axis("risk").reset_index(
        name="count")
    st.plotly_chart(px.bar(risk, x="risk", y="count", color="risk",
                           color_discrete_map=RISK_COLORS),
                    use_container_width=True)
    if summary["rejected_rows"]:
        st.warning(f"{summary['n_rejected']} row(s) rejected (showing up to "
                   f"{len(summary['rejected_rows'])}).")
        st.dataframe(pd.DataFrame(summary["rejected_rows"]),
                     hide_index=True)
    st.subheader("Flagged transactions")
    flagged = pd.read_csv(io.BytesIO(flagged_csv))
    st.dataframe(flagged, hide_index=True, use_container_width=True)
    st.download_button("Download flagged CSV", flagged_csv,
                       file_name=f"flagged_{summary['batch_id']}.csv",
                       mime="text/csv")


# ----------------------------------------------------- page: analyst review
def review(alert_id: int, status: str, notes: str) -> None:
    """Send a review decision, then refresh the page."""
    try:
        api("PATCH", f"/alerts/{alert_id}",
            json={"status": status, "notes": notes or None})
        st.rerun()
    except ApiError as exc:
        st.error(str(exc))


def page_review() -> None:
    """Page 3: confirm or reject open alerts (analysts only)."""
    st.header("Analyst review")
    if st.session_state.get("role") != "analyst":
        st.warning("This page needs the analyst role.")
        return
    try:
        data = api("GET", "/alerts",
                   params={"status": "open", "page_size": 20}).json()
    except ApiError as exc:
        st.error(str(exc))
        return
    st.caption(f"{data['total']} open alert(s), highest risk first "
               f"(showing {len(data['items'])}).")
    for alert in data["items"]:
        with st.container(border=True):
            info, actions = st.columns([3, 1])
            info.markdown(
                f"**Alert #{alert['id']}** - {alert['transaction_time']} - "
                f"probability **{alert['fraud_probability']:.3f}** - "
                f"risk **{alert['risk_level']}** - amount "
                f"{alert['amount']}")
            if alert["top_factors"]:
                info.dataframe(
                    pd.DataFrame(alert["top_factors"])[
                        ["feature", "value", "shap_value"]],
                    hide_index=True)
            notes = actions.text_input("Notes", key=f"note_{alert['id']}")
            if actions.button("Confirm fraud", key=f"c_{alert['id']}"):
                review(alert["id"], "confirmed_fraud", notes)
            if actions.button("False positive", key=f"f_{alert['id']}"):
                review(alert["id"], "false_positive", notes)


# --------------------------------------------------------- page: analytics
def page_analytics() -> None:
    """Page 4: operational stats, model metrics and the PR curve."""
    st.header("Analytics")
    days = st.slider("Window (days)", 1, 90, 30)
    try:
        stats = api("GET", "/stats", params={"days": days}).json()
        info = api("GET", "/model/info").json()
    except ApiError as exc:
        st.error(str(exc))
        return

    m = st.columns(4)
    m[0].metric("Transactions", stats["total_transactions"])
    m[1].metric("Flagged", stats["flagged_transactions"])
    m[2].metric("Flag rate", f"{stats['fraud_rate']:.2%}")
    precision = stats["analyst_precision"]
    m[3].metric("Analyst-confirmed precision",
                "n/a" if precision is None else f"{precision:.0%}",
                f"{stats['reviewed_alerts']} reviewed")

    daily = pd.DataFrame(stats["daily"])
    left, right = st.columns(2)
    if not daily.empty:
        daily["flag_rate"] = daily["flagged"] / daily["total"]
        left.plotly_chart(px.line(daily, x="date", y="flag_rate",
                                  markers=True, title="Flag rate over time"),
                          use_container_width=True)
        right.plotly_chart(px.bar(daily, x="date", y="flagged",
                                  title="Flagged per day"),
                           use_container_width=True)
    risk = pd.Series(stats["risk_counts"]).rename_axis("risk").reset_index(
        name="count")
    st.plotly_chart(px.bar(risk, x="risk", y="count", color="risk",
                           color_discrete_map=RISK_COLORS,
                           title="Risk level distribution"),
                    use_container_width=True)

    st.subheader("Model")
    metrics = info.get("metrics_test_at_chosen_threshold", {})
    cols = st.columns(max(len(metrics), 1))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")
    st.caption(f"Model {info.get('model_name')} - trained "
               f"{info.get('training_date_utc')} - threshold "
               f"{info.get('threshold', {}).get('chosen')}")
    with st.expander("Library versions and hyperparameters"):
        st.json({"library_versions": info.get("library_versions"),
                 "hyperparameters": info.get("hyperparameters")})
    try:
        st.image(api("GET", "/model/pr-curve").content,
                 caption="Precision-Recall curve (held-out test set)")
    except ApiError as exc:
        st.info(f"PR curve not available ({exc}).")


# -------------------------------------------------------------------- main
def main() -> None:
    """Route between the dashboard pages."""
    if not login_sidebar():
        st.info("Log in from the sidebar to continue.")
        return
    pages = {"Check a transaction": page_check, "Batch upload": page_batch,
             "Analyst review": page_review, "Analytics": page_analytics}
    choice = st.sidebar.radio("Page", list(pages))
    try:
        pages[choice]()
    except ApiError as exc:
        st.error(str(exc))


main()