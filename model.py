"""
Model training and validation.

IMPORTANT: we use TimeSeriesSplit, never random shuffling. Randomly shuffling
financial time series lets the model "see the future" during training (data
leakage), which inflates accuracy numbers to the point of being meaningless.
Every split here trains only on the past and tests only on data that comes
strictly after it in time.
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score


def build_model(model_type="rf"):
    if model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def walk_forward_validate(X, y, n_splits=5, model_type="rf"):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = build_model(model_type)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        fold_results.append({"fold": fold, "accuracy": acc, "report": report})

        print(f"  Fold {fold}: accuracy={acc:.3f}  (test size={len(test_idx)})")
        for cls in ["BUY", "SELL", "HOLD"]:
            if cls in report:
                p, r, f1 = report[cls]["precision"], report[cls]["recall"], report[cls]["f1-score"]
                print(f"      {cls:5s}  precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}")

    return fold_results


def train_final_model(X, y, model_type="rf"):
    model = build_model(model_type)
    model.fit(X, y)
    return model
