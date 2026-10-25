import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml_backend import FEATURES, TARGET, load_dataset, prepare_data, train_evaluate, predict_one

st.set_page_config(page_title="Grid Stability KNN", page_icon="⚡", layout="wide")
st.title("⚡ KNN-Based Electrical Grid Stability Classification")
st.caption("Interactive engineering ML application — Jupyter Notebook backend + Streamlit GUI")
st.warning("Educational use only. This classifier is not certified for real-time grid protection or operational control.")

with st.sidebar:
    st.header("Dataset and model settings")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    use_selection = st.checkbox("Enable feature selection", value=True)
    n_features = st.slider("Number of selected features", min_value=2, max_value=11, value=6)
    test_pct = st.slider("Test set (%)", min_value=10, max_value=40, value=20, step=5)
    k_candidates = st.multiselect("Candidate K values", [1,3,5,7,9,11,15,21,25,31], default=[1,3,5,7,9,11,15,21])
    train_btn = st.button("Train and evaluate", type="primary")

try:
    df = pd.read_csv(uploaded) if uploaded is not None else load_dataset()
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.info("Check your internet connection for the first UCI download, or upload a CSV file.")
    st.stop()

st.subheader("1. Dataset preview")
st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns)}**")
st.dataframe(df.head(10), use_container_width=True)
with st.expander("Data quality and summary"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing cells", int(df.isna().sum().sum()))
    c2.metric("Duplicate rows", int(df.duplicated().sum()))
    c3.metric("Numeric columns", int(len(df.select_dtypes(include="number").columns)))
    st.write("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing values"))
    st.write("Summary statistics")
    st.dataframe(df.describe(include="all").transpose())

default_target = TARGET if TARGET in df.columns else str(df.columns[-1])
target = st.selectbox("2. Target column", list(df.columns), index=list(df.columns).index(default_target))
defaults = [c for c in FEATURES if c in df.columns and c != target]
features = st.multiselect("Predictor features", [c for c in df.columns if c != target], default=defaults)

if target == "stabf" and any(c in features for c in ["stab", "p1"]):
    st.warning("For the UCI dataset, exclude `stab` (target leakage risk) and `p1` (documented as non-predictive).")

if train_btn:
    if not features:
        st.error("Select at least one predictor feature.")
    elif not k_candidates:
        st.error("Select at least one candidate K.")
    else:
        try:
            X, y, cleaned = prepare_data(df, target=target, features=features)
            if y.nunique() < 2:
                raise ValueError("The target must contain at least two classes.")
            if y.value_counts().min() < 5:
                raise ValueError("Each class needs at least five records for 5-fold cross-validation.")
            result = train_evaluate(
                X, y, use_feature_selection=use_selection,
                n_features=min(n_features, len(features)),
                test_size=test_pct/100, random_state=42, k_values=k_candidates
            )
            st.session_state["knn_result"] = result
            st.session_state["knn_features"] = features
            st.session_state["knn_target"] = target
        except Exception as exc:
            st.error(f"Training failed: {exc}")

result = st.session_state.get("knn_result")
if result is None:
    st.info("Choose the dataset and settings, then click **Train and evaluate**.")
    st.stop()

st.subheader("3. Evaluation results")
st.write(f"Selected K: **{result['best_k']}** · 5-fold CV accuracy: **{result['cv_accuracy']:.2%}**")
a,b,c,d = st.columns(4)
a.metric("Test accuracy", f"{result['accuracy']:.2%}")
b.metric("Weighted precision", f"{result['precision_weighted']:.2%}")
c.metric("Weighted recall", f"{result['recall_weighted']:.2%}")
d.metric("Weighted F1-score", f"{result['f1_weighted']:.2%}")
if "roc_auc" in result:
    st.metric("ROC-AUC", f"{result['roc_auc']:.4f}")
if result["accuracy"] >= 0.90:
    st.success("The held-out test accuracy meets the 90% project benchmark.")
else:
    st.warning("The held-out test accuracy is below 90%. Review features and methodology; do not tune on the test set.")

tab1, tab2, tab3 = st.tabs(["Confusion matrix", "Classification report", "K selection"])
with tab1:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=result["classes"], yticklabels=result["classes"], ax=ax)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    ax.set_title("Held-out test confusion matrix")
    st.pyplot(fig); plt.close(fig)
with tab2:
    from sklearn.metrics import classification_report
    report = classification_report(result["y_test"], result["predictions"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    st.code(result["classification_report"])
with tab3:
    cvres = pd.DataFrame(result["search"].cv_results_)
    cvres["K"] = cvres["param_knn__n_neighbors"].astype(int)
    cvres = cvres.sort_values("K")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.errorbar(cvres["K"], cvres["mean_test_score"], yerr=cvres["std_test_score"], marker="o", capsize=4)
    ax.axvline(result["best_k"], linestyle="--", label=f"Best K = {result['best_k']}")
    ax.set_xlabel("K"); ax.set_ylabel("Mean 5-fold CV accuracy")
    ax.set_title("K selection using training data only"); ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.dataframe(cvres[["K","mean_test_score","std_test_score"]].rename(columns={
        "mean_test_score":"Mean CV accuracy","std_test_score":"CV standard deviation"}))

st.subheader("4. Predict a new operating condition")
st.write("Enter values for the same features used to train the model.")
example = {"tau1":4.0,"tau2":4.0,"tau3":4.0,"tau4":4.0,"p2":-0.5,"p3":-0.5,"p4":-0.5,
           "g1":0.5,"g2":0.5,"g3":0.5,"g4":0.5}
chosen_features = st.session_state["knn_features"]
with st.form("new_observation"):
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(chosen_features):
        with cols[i % 3]:
            values[f] = st.number_input(f, value=float(example.get(f, float(df[f].median()) if pd.api.types.is_numeric_dtype(df[f]) else 0.0)), format="%.5f")
    submit = st.form_submit_button("Predict stability")
if submit:
    prediction, probs = predict_one(result["model"], values, feature_names=chosen_features)
    if str(prediction).lower() == "stable":
        st.success(f"Predicted grid state: {str(prediction).upper()}")
    else:
        st.error(f"Predicted grid state: {str(prediction).upper()}")
    st.write("Model probability estimates (not calibrated guarantees)")
    st.bar_chart(pd.Series(probs, name="Estimated probability"))
    st.dataframe(pd.DataFrame({"Class":list(probs.keys()), "Estimated probability":list(probs.values())}))
st.download_button("Download held-out test predictions",
    data=pd.DataFrame({"actual":result["y_test"].to_numpy(),"predicted":result["predictions"]}).to_csv(index=False),
    file_name="knn_grid_test_predictions.csv", mime="text/csv")
