import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("Student Performance Prediction")
st.write("Predict student performance using supervised ensemble learning")
st.divider()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_performance.csv")

st.sidebar.header("Dataset Information")
st.sidebar.write("Student Performance Dataset")

try:
    df = load_data()
except FileNotFoundError:
    st.error("student_performance.csv not found in the project directory")
    st.stop()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.write("Dataset shape:", df.shape)

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
st.subheader("Data Preprocessing")

df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded.drop("Performance", axis=1)
y = df_encoded["Performance"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

st.success("Data preprocessing completed")

# --------------------------------------------------
# Models
# --------------------------------------------------
lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
ada = AdaBoostClassifier(n_estimators=100, random_state=42)

voting = VotingClassifier(
    estimators=[
        ("Logistic Regression", lr),
        ("Decision Tree", dt),
        ("Random Forest", rf)
    ],
    voting="hard"
)

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "AdaBoost": ada,
    "Voting Classifier": voting
}

# --------------------------------------------------
# Model Training and Evaluation
# --------------------------------------------------
st.subheader("Model Accuracy Comparison")

accuracy_results = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_results[name] = accuracy_score(y_test, y_pred)
    except Exception as e:
        accuracy_results[name] = None
        st.warning(f"{name} failed to train: {e}")

accuracy_df = (
    pd.DataFrame.from_dict(accuracy_results, orient="index", columns=["Accuracy"])
    .reset_index()
    .rename(columns={"index": "Model"})
)

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(accuracy_df, use_container_width=True)

with col2:
    valid_models = accuracy_df.dropna()
    if not valid_models.empty:
        best_model = valid_models.loc[valid_models["Accuracy"].idxmax()]
        acc_percent = round(best_model["Accuracy"] * 100, 2)
        st.metric("Best Model", best_model["Model"], f"{acc_percent}%")

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.divider()
st.subheader("Student Performance Prediction")

inputs = []
cols = st.columns(3)

# Use median for numeric defaults
numeric_medians = X.median()

for idx, col in enumerate(X.columns):
    with cols[idx % 3]:
        if col in label_encoders:  # Categorical column
            options = df[col].unique().tolist()
            selected = st.selectbox(col, options)
            value = label_encoders[col].transform([selected])[0]
        else:  # Numeric column
            default = float(numeric_medians[col])
            value = st.number_input(col, min_value=0.0, step=1.0, value=default)
        inputs.append(value)

if st.button("Predict Performance", use_container_width=True):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = voting.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Prediction result: PASS")
    else:
        st.error("Prediction result: FAIL")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Student Performance Prediction Application")
