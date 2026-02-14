import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.datasets import load_iris, fetch_california_housing

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(page_title="Supervised Learning Dashboard", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🚀 Supervised Learning Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

task = st.sidebar.selectbox("Select Task", ["Classification", "Regression"])

if task == "Classification":
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )
else:
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Linear Regression", "Decision Tree", "Random Forest"]
    )

n_trees = 100
if "Random Forest" in algorithm:
    n_trees = st.sidebar.slider("Number of Trees", 10, 300, 100, 10)

# Sidebar Summary
with st.sidebar.expander("📌 Current Configuration"):
    st.write(f"Task: **{task}**")
    st.write(f"Algorithm: **{algorithm}**")
    if "Random Forest" in algorithm:
        st.write(f"Trees: **{n_trees}**")

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tabs = st.tabs(["📂 Load Data", "🔄 Preprocess", "🤖 Train & Evaluate", "🔮 Predict"])

# ---------------------------------------------------
# 1️⃣ Load Data
# ---------------------------------------------------
with tabs[0]:
    st.header("Load Dataset")

    dataset_option = st.radio(
        "Choose Dataset Source",
        ["Default Dataset", "Upload CSV"],
        horizontal=True
    )

    if dataset_option == "Default Dataset":
        if task == "Classification":
            data = load_iris(as_frame=True)
            df = data.frame
            target_col = "target"
            st.info("Using Iris Dataset (Classification)")
        else:
            try:
                data = fetch_california_housing(as_frame=True)
                df = data.frame
                target_col = "MedHouseVal"
                st.info("Using California Housing Dataset (Regression)")
            except:
                st.error("California dataset failed to load.")
                st.stop()

    else:
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            target_col = st.selectbox("Select Target Column", df.columns)
        else:
            st.warning("Please upload a CSV file.")
            st.stop()

    st.session_state["df"] = df
    st.session_state["target_col"] = target_col

    with st.expander("Preview Dataset"):
        st.dataframe(df.head(), use_container_width=True)
        st.write("Shape:", df.shape)

# ---------------------------------------------------
# 2️⃣ Preprocessing
# ---------------------------------------------------
with tabs[1]:
    st.header("Data Preprocessing")

    if "df" not in st.session_state:
        st.warning("Please load dataset first.")
        st.stop()

    df = st.session_state["df"]
    target_col = st.session_state["target_col"]

    df_encoded = df.copy()
    label_encoders = {}

    # Encode categorical features
    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Encode target if categorical
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Scaling only for linear models
    if algorithm in ["Logistic Regression", "Linear Regression"]:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        scaler = None
        X_processed = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    st.session_state["X"] = X
    st.session_state["scaler"] = scaler
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    st.success("Preprocessing Completed ✅")

# ---------------------------------------------------
# 3️⃣ Train & Evaluate
# ---------------------------------------------------
with tabs[2]:
    st.header("Train & Evaluate Model")

    if "X_train" not in st.session_state:
        st.warning("Please preprocess data first.")
        st.stop()

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    if task == "Classification":
        if algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        elif algorithm == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    else:
        if algorithm == "Linear Regression":
            model = LinearRegression()
        elif algorithm == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_trees, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.session_state["model"] = model

    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc*100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.4f}")
        col2.metric("R² Score", f"{r2:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--')
        st.pyplot(fig)

    st.success("Model Trained Successfully 🚀")

# ---------------------------------------------------
# 4️⃣ Prediction
# ---------------------------------------------------
with tabs[3]:
    st.header("Make a Prediction")

    if "model" not in st.session_state:
        st.warning("Please train the model first.")
        st.stop()

    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    X = st.session_state["X"]

    inputs = []
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            val = st.number_input(
                f"{col}",
                value=float(X[col].mean())
            )
            inputs.append(val)

    if st.button("Predict"):
        input_array = np.array(inputs).reshape(1, -1)

        if scaler is not None:
            input_array = scaler.transform(input_array)

        prediction = model.predict(input_array)

        if task == "Classification":
            st.success(f"Predicted Class: {int(prediction[0])}")
        else:
            st.success(f"Predicted Value: {prediction[0]:.4f}")

st.markdown("---")
st.caption("Interactive Supervised Learning ML Dashboard | Built with Streamlit")
