from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p2", "p3", "p4", "g1", "g2", "g3", "g4"
]
TARGET = "stabf"
DATA_PATH = Path(__file__).parent / "data" / "Data_for_UCI_named.csv"

def load_official_dataset(save=True):
    """Fetch the official UCI dataset and optionally save a local copy."""
    dataset = fetch_ucirepo(id=471)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y["stabf"] if "stabf" in y.columns else y.iloc[:, -1]
    else:
        y = pd.Series(y, name=TARGET)
    df = X.copy()
    df[TARGET] = y.to_numpy()
    if save:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df

def load_dataset(path=None):
    """Load an uploaded/local CSV, or fetch UCI data if no local file exists."""
    if path is not None:
        return pd.read_csv(path)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return load_official_dataset(save=True)

def prepare_data(df, target=TARGET, features=None):
    """Check columns, drop duplicates and invalid targets, return numeric X and y."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")
    if features is None:
        features = [c for c in FEATURES if c in df.columns]
    if not features:
        raise ValueError("No predictor features were selected.")
    if target in features:
        raise ValueError("The target column cannot be used as a predictor.")
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Selected columns not found: {missing_cols}")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=[target])
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[target].astype(str).str.strip()
    valid = y.ne("") & y.ne("nan")
    return X.loc[valid], y.loc[valid], df.loc[valid]

def make_pipeline(use_feature_selection=True, n_features=6):
    selector = (
        SelectKBest(score_func=mutual_info_classif, k="all")
        if not use_feature_selection else
        SelectKBest(score_func=mutual_info_classif, k=max(1, int(n_features)))
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", selector),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(metric="minkowski", p=2))
    ])

def train_evaluate(X, y, use_feature_selection=True, n_features=6,
                   test_size=0.20, random_state=42, k_values=None):
    """Split first; tune K only on training data; evaluate once on held-out test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    # Ensure candidate K is valid for every CV training fold.
    smallest_cv_train = len(X_train) - int(__import__("math").ceil(len(X_train) / 5))
    k_values = sorted({int(k) for k in k_values if int(k) >= 1 and int(k) <= smallest_cv_train})
    if not k_values:
        raise ValueError("No valid K values remain for the selected dataset.")
    n_features = min(int(n_features), X.shape[1])
    pipe = make_pipeline(use_feature_selection, n_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipe, {"knn__n_neighbors": k_values}, scoring="accuracy",
        cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "best_k": search.best_params_["knn__n_neighbors"],
        "cv_accuracy": search.best_score_,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=model.classes_),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "classes": model.classes_, "y_test": y_test, "predictions": pred,
        "X_train": X_train, "X_test": X_test, "y_train": y_train,
        "search": search, "model": model
    }
    if len(model.classes_) == 2:
        positive = model.classes_[1]
        pos_index = list(model.classes_).index(positive)
        metrics["roc_auc"] = roc_auc_score(
            (y_test == positive).astype(int), model.predict_proba(X_test)[:, pos_index]
        )
    return metrics

def predict_one(model, values, feature_names=None):
    """Predict one row; values may be a dict keyed by feature name or a sequence."""
    if isinstance(values, dict):
        row = pd.DataFrame([values])
        if feature_names is not None:
            row = row.reindex(columns=feature_names)
    else:
        if feature_names is None:
            raise ValueError("feature_names are required when values is a sequence.")
        row = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    return prediction, dict(zip(model.classes_, probabilities))
