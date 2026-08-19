import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival — Multi-Model Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-title {
        font-size: 2.3rem; font-weight: 800; color: #f5f5f5;
        margin-bottom: 0px;
    }
    .subtitle { color: #9aa0a6; font-size: 1.02rem; margin-top: 0px; }
    .model-card {
        border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;
        border: 1px solid #2a2e37;
    }
    .survive { background: linear-gradient(135deg, #123c2b, #0e1117); border-color:#1f7a4d; }
    .perish  { background: linear-gradient(135deg, #3c1414, #0e1117); border-color:#a13939; }
    .model-name { font-size: 0.95rem; color: #c7cbd1; font-weight: 600; letter-spacing: .3px;}
    .pred-label { font-size: 1.4rem; font-weight: 800; margin-top: 4px;}
    .pred-conf { color: #9aa0a6; font-size: 0.85rem; }
    .consensus-box {
        border-radius: 14px; padding: 22px; text-align: center;
        border: 1px solid #2a2e37;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚢 Titanic Survival Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compare predictions from 5 machine learning models side-by-side</div>', unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------------------------
def _patch_version_gaps(model):
    """
    Guards against AttributeErrors caused by loading a pickle that was saved
    with a different scikit-learn version than the one currently installed
    (e.g. 'multi_class' missing on LogisticRegression after a version change).
    Fills in safe defaults for attributes newer/older sklearn code expects.
    """
    defaults = {
        "multi_class": "auto",
        "n_jobs": None,
        "l1_ratio": None,
        "warm_start": False,
    }
    for attr, val in defaults.items():
        if not hasattr(model, attr):
            try:
                setattr(model, attr, val)
            except Exception:
                pass
    return model


@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
        "K-Nearest Neighbors": joblib.load("knn_model.pkl"),
        "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
        "SVM": joblib.load("titanic_model.pkl"),
    }
    for name in models:
        models[name] = _patch_version_gaps(models[name])
    models["scaler"] = joblib.load("scaler.pkl")
    models["label_encoder"] = joblib.load("label_encoder.pkl")
    return models

artifacts = load_artifacts()
scaler = artifacts["scaler"]
le = artifacts["label_encoder"]  # classes_: ['C','Q','S']  ->  embarked: C=0, Q=1, S=2

# Models trained on the *scaled* feature set in the notebook
SCALED_MODELS = {"K-Nearest Neighbors", "Decision Tree", "SVM"}
# Models trained on the *raw* (unscaled) feature set in the notebook
RAW_MODELS = {"Logistic Regression", "Naive Bayes"}

FEATURE_ORDER = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "alone"]

MODEL_COLORS = {
    "Logistic Regression": "#4C9AFF",
    "K-Nearest Neighbors": "#F5A623",
    "Naive Bayes": "#B084F5",
    "Decision Tree": "#50D890",
    "SVM": "#FF6B81",
}

# ----------------------------------------------------------------------------
# SIDEBAR — PASSENGER INPUT
# ----------------------------------------------------------------------------
st.sidebar.header("🧍 Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], index=2,
                               format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
sex = st.sidebar.radio("Sex", ["male", "female"], horizontal=True)
age = st.sidebar.slider("Age", 0, 80, 29)
sibsp = st.sidebar.number_input("Siblings / Spouses aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents / Children aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare ($)", 0, 512, 32)
embarked_label = st.sidebar.selectbox(
    "Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"]
)

alone_auto = (sibsp + parch) == 0
st.sidebar.caption(f"Traveling alone: **{'Yes' if alone_auto else 'No'}** (derived from family size)")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔮 Predict with all models", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# ENCODE INPUT
# ----------------------------------------------------------------------------
sex_encoded = 1 if sex == "male" else 0  # LabelEncoder: female=0, male=1

port_map = {"Southampton (S)": "S", "Cherbourg (C)": "C", "Queenstown (Q)": "Q"}
embarked_letter = port_map[embarked_label]
embarked_encoded = int(le.transform([embarked_letter])[0])  # C=0, Q=1, S=2

alone_encoded = int(alone_auto)

raw_input = pd.DataFrame([{
    "pclass": pclass,
    "sex": sex_encoded,
    "age": age,
    "sibsp": sibsp,
    "parch": parch,
    "fare": fare,
    "embarked": embarked_encoded,
    "alone": alone_encoded,
}])[FEATURE_ORDER]

scaled_input = pd.DataFrame(scaler.transform(raw_input), columns=FEATURE_ORDER)

# ----------------------------------------------------------------------------
# RUN PREDICTIONS
# ----------------------------------------------------------------------------
def predict_all():
    results = {}
    for name in ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes", "Decision Tree", "SVM"]:
        model = artifacts[name]
        X = scaled_input if name in SCALED_MODELS else raw_input
        pred = int(model.predict(X)[0])
        confidence, survive_proba = None, None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                confidence = float(proba[pred])
                survive_proba = float(proba[1])
            except AttributeError:
                # Version-mismatch gap slipped through _patch_version_gaps;
                # fall back gracefully instead of crashing the app.
                pass
        results[name] = {"pred": pred, "confidence": confidence, "survive_proba": survive_proba}
    return results

if "results" not in st.session_state:
    st.session_state.results = predict_all()

if run_btn:
    st.session_state.results = predict_all()

results = st.session_state.results

# ----------------------------------------------------------------------------
# SUMMARY / CONSENSUS
# ----------------------------------------------------------------------------
votes = [r["pred"] for r in results.values()]
survive_votes = sum(votes)
total = len(votes)
consensus_survive = survive_votes > total / 2

c1, c2 = st.columns([1, 2])
with c1:
    box_class = "survive" if consensus_survive else "perish"
    icon = "✅" if consensus_survive else "☠️"
    verdict = "LIKELY SURVIVES" if consensus_survive else "LIKELY DOES NOT SURVIVE"
    st.markdown(f"""
    <div class="consensus-box {box_class}">
        <div style="font-size:2.2rem;">{icon}</div>
        <div style="font-size:1.3rem; font-weight:800; color:#f5f5f5; margin-top:4px;">{verdict}</div>
        <div style="color:#9aa0a6; margin-top:6px;">{survive_votes} of {total} models predict survival</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    probs = [r["survive_proba"] for r in results.values() if r["survive_proba"] is not None]
    names_with_proba = [n for n, r in results.items() if r["survive_proba"] is not None]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names_with_proba,
        y=probs,
        marker_color=[MODEL_COLORS[n] for n in names_with_proba],
        text=[f"{p:.0%}" for p in probs],
        textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% threshold")
    fig.update_layout(
        title="Predicted Probability of Survival by Model",
        yaxis=dict(title="P(Survive)", range=[0, 1.15], tickformat=".0%"),
        template="plotly_dark",
        height=320,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# PER-MODEL CARDS
# ----------------------------------------------------------------------------
st.subheader("📊 Individual Model Predictions")
cols = st.columns(5)
for col, (name, r) in zip(cols, results.items()):
    survived = r["pred"] == 1
    card_class = "survive" if survived else "perish"
    label = "Survived ✅" if survived else "Did Not Survive ☠️"
    conf_text = f"Confidence: {r['confidence']:.1%}" if r["confidence"] is not None else "Confidence: N/A"
    with col:
        st.markdown(f"""
        <div class="model-card {card_class}">
            <div class="model-name">{name}</div>
            <div class="pred-label">{label}</div>
            <div class="pred-conf">{conf_text}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# DETAILED TABLE
# ----------------------------------------------------------------------------
st.subheader("📋 Detailed Comparison")
table_rows = []
for name, r in results.items():
    table_rows.append({
        "Model": name,
        "Prediction": "Survived" if r["pred"] == 1 else "Did Not Survive",
        "P(Survive)": f"{r['survive_proba']:.1%}" if r["survive_proba"] is not None else "N/A",
        "P(Did Not Survive)": f"{(1 - r['survive_proba']):.1%}" if r["survive_proba"] is not None else "N/A",
        "Confidence": f"{r['confidence']:.1%}" if r["confidence"] is not None else "N/A",
        "Trained on": "Scaled features" if name in SCALED_MODELS else "Raw features",
    })
df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# INPUT SUMMARY (collapsible)
# ----------------------------------------------------------------------------
with st.expander("🔧 View encoded model input"):
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Raw features** (used by Logistic Regression & Naive Bayes)")
        st.dataframe(raw_input, hide_index=True, use_container_width=True)
    with ic2:
        st.markdown("**Scaled features** (used by KNN, Decision Tree & SVM)")
        st.dataframe(scaled_input.round(3), hide_index=True, use_container_width=True)
    st.caption(
        "Encoding: sex → female=0, male=1 · embarked → C=0, Q=1, S=2 · "
        "alone → derived as 1 if siblings/spouses + parents/children aboard = 0, else 0."
    )

st.markdown("---")
st.caption("Built with Streamlit · Models: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM — trained on the Titanic dataset.")