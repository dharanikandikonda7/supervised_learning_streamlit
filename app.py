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

# Page Config
st.set_page_config(page_title="Supervised Learning Dashboard", layout="wide")

# Title with subtle styling
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Supervised Learning Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.header("Model Settings")

task = st.sidebar.selectbox("Select Task", ["Classification", "Regression"], help="Choose supervised learning task")

if task == "Classification":
    algorithm = st.sidebar.selectbox("Algorithm", ["Logistic Regression", "Decision Tree", "Random Forest"], help="Choose classification algorithm")
else:
    algorithm = st.sidebar.selectbox("Algorithm", ["Linear Regression", "Decision Tree", "Random Forest"], help="Choose regression algorithm")

n_trees = 100
if "Random Forest" in algorithm:
    n_trees = st.sidebar.slider("Number of Trees", 10, 300, 100, 10, help="Set number of trees for Random Forest")

# Sidebar summary
with st.sidebar.expander("Current Configuration"):
    st.write(f"Task: **{task}**")
    st.write(f"Algorithm: **{algorithm}**")
    if "Random Forest" in algorithm:
        st.write(f"Number of Trees: **{n_trees}**")

# Tabs for better navigation
tabs = st.tabs(["Load Data", "Preprocess", "Train & Evaluate", "Predict"])

# Step 1: Load Dataset
with tabs[0]:
    st.header("Load Dataset")
    dataset_option = st.radio("Select Dataset Source", ["Default Dataset", "Upload CSV"], horizontal=True)

    if dataset_option == "Default Dataset":
        if task == "Classification":
            data = load_iris(as_frame=True)
            df = data.frame
            target_col = "target"
            st.info("Using Iris dataset for classification")
        else:
            data = fetch_california_housing(as_frame=True)
            df = data.frame
            target_col = "MedHouseVal"
            st.info("Using California Housing dataset for regression")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            target_col = st.selectbox("Select Target Column", df.columns)
        else:
            st.warning("Please upload a CSV file to continue.")
            st.stop()

    with st.expander("View Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)
        st.write("Dataset shape:", df.shape)

# Step 2: Data Preprocessing
with tabs[1]:
    st.header("Data Preprocessing")

    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    st.success("Data preprocessing complete")
    with st.expander("Show feature statistics"):
        st.write(X.describe())

# Step 3: Train and Evaluate
with tabs[2]:
    st.header("Train Model and Evaluate")

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

    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc * 100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric("Mean Squared Error", f"{mse:.4f}")
        st.metric("R Squared", f"{r2:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

# Step 4: Prediction
with tabs[3]:
    st.header("Make a Prediction")

    inputs = []
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            val = st.number_input(
                label=col,
                min_value=float(X[col].min()),
                max_value=float(X[col].max()),
                value=float(X[col].mean()),
                help=f"Input value for {col}"
            )
            inputs.append(val)

    if st.button("Predict"):
        input_arr = np.array(inputs).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)
        result = model.predict(input_scaled)

        if task == "Classification":
            st.success(f"Predicted Class: {int(result[0])}")
        else:
            st.success(f"Predicted Value: {result[0]:.4f}")

st.markdown("---")
st.caption("Interactive Supervised Learning Dashboard")
