import streamlit as st

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    st.set_page_config(page_title="Iris Decision Tree Classifier", layout="centered")
    st.title("Iris Decision Tree Classifier")

    iris = load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names

    # UI: hyperparameters
    st.sidebar.header("Decision Tree Parameters")

    criterion = st.sidebar.selectbox(
        "Criterion",
        options=["gini", "entropy", "log_loss"],
        index=0,
    )

    splitter = st.sidebar.selectbox("Splitter", options=["best", "random"], index=0)

    max_depth = st.sidebar.selectbox(
        "Max depth",
        options=[None, 2, 3, 4, 5, 6, 7, 8, 10, 20],
        index=0,
    )

    min_samples_split = st.sidebar.selectbox(
        "Min samples split",
        options=[2, 3, 4, 5, 10, 20],
        index=0,
    )

    min_samples_leaf = st.sidebar.selectbox(
        "Min samples leaf",
        options=[1, 2, 3, 4, 5, 10],
        index=0,
    )

    max_features = st.sidebar.selectbox(
        "Max features",
        options=[None, "sqrt", "log2"],
        index=0,
    )

    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)

    # Preprocess + train
    # Note: fit scaler only on training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=int(random_state),
    )

    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance (Test Set)")
    st.write(f"**Accuracy:** {acc:.4f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=target_names))

    # Input UI for prediction
    st.subheader("Predict Iris Class")

    with st.form("prediction_form"):
        inputs = []
        for name in feature_names:
            default = float(np.mean(X[:, feature_names.index(name)]))
            val = st.number_input(f"{name}", value=default, format="%.4f")
            inputs.append(val)

        submitted = st.form_submit_button("Predict")

    if submitted:
        sample = np.array(inputs, dtype=float).reshape(1, -1)
        sample_scaled = scaler.transform(sample)
        pred_class = int(clf.predict(sample_scaled)[0])

        st.success(f"Predicted Class: {target_names[pred_class]} (label={pred_class})")

        proba = getattr(clf, "predict_proba", None)
        if proba is not None:
            probs = clf.predict_proba(sample_scaled)[0]
            st.write("Prediction Probabilities:")
            st.json({target_names[i]: float(probs[i]) for i in range(len(target_names))})


if __name__ == "__main__":
    main()

