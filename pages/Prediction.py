import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

st.set_page_config(page_title='DHV Decision Tree', layout='wide')
#st.title("DHV AI Startup: Preoperative Risk Prediction (trained with 6,388 patient data)")
st.markdown("<h1 style='text-align: center;'>DHV AI Startup: Preoperative Risk Prediction (trained with 6,388 patient data)</h1>", unsafe_allow_html=True)
st.write("")

# Read the training_data.csv file into a DataFrame
data = pd.read_csv("data.csv")

# Drop rows with NaN values in the "risk" column
#data = data.dropna(subset=["risk"])

# Separate features and target
columns_to_drop = ["risk","caseid","death_inhosp","department","optype","dx","ane_type","preop_ecg","preop_pft","icu_days",
"intraop_epi", "preop_ph", "preop_hco3", "preop_be", "preop_pao2", "preop_paco2", "preop_sao2" ]
features = data.drop(columns_to_drop, axis=1)
target = data["risk"]

# Perform any necessary data preprocessing
encoder = LabelEncoder()
features["sex"] = encoder.fit_transform(features["sex"])
features["asa"] = encoder.fit_transform(features["asa"])
features["emop"] = encoder.fit_transform(features["emop"])
features["preop_htn"] = encoder.fit_transform(features["preop_htn"])
features["preop_dm"] = encoder.fit_transform(features["preop_dm"])

# One-hot encode the "sex" feature
features = pd.get_dummies(features, columns=["sex"], drop_first=True)

# Train a decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(features, target)

# Function to get the most recent CSV file
def get_most_recent_csv():
    folder_path = "user_input/"
    files = os.listdir(folder_path)
    if len(files) == 0:
        return None
    files.sort(reverse=True)
    most_recent_csv = files[0]
    return os.path.join(folder_path, most_recent_csv)

# Get the path of the most recent CSV file
recent_csv_path = get_most_recent_csv()

if recent_csv_path is not None:
    # Load the most recent CSV file
    df = pd.read_csv(recent_csv_path)

    # Get the most recent user input data
    recent_user_input = df.iloc[-1]

     # Set the user input values
    date = recent_user_input["date"]
    age = recent_user_input["age"]
    sex = recent_user_input["sex"]
    bmi = recent_user_input["bmi"]
    asa = recent_user_input["asa"]
    emop = recent_user_input["emop"]
    preop_htn = recent_user_input["preop_htn"]
    preop_dm = recent_user_input["preop_dm"]
    preop_hb = recent_user_input["preop_hb"]
    preop_plt = recent_user_input["preop_plt"]
    preop_pt = recent_user_input["preop_pt"]
    preop_aptt = recent_user_input["preop_aptt"]
    preop_na = recent_user_input["preop_na"]
    preop_k = recent_user_input["preop_k"]
    preop_gluc = recent_user_input["preop_gluc"]
    preop_alb = recent_user_input["preop_alb"]
    preop_ast = recent_user_input["preop_ast"]
    preop_alt = recent_user_input["preop_alt"]
    preop_bun = recent_user_input["preop_bun"]
    preop_cr = recent_user_input["preop_cr"]

# Create a DataFrame for user input
    recent_user_input = pd.DataFrame([[age, sex, bmi, asa, emop, preop_htn, preop_dm, preop_hb, preop_plt, preop_pt,
                                       preop_aptt, preop_na, preop_k, preop_gluc, preop_alb, preop_ast, preop_alt,
                                       preop_bun, preop_cr]], columns=features.columns)
st.dataframe(df)
st.write("")
st.write("")
st.markdown("<h3 style='color: blue;'>1 = Low risk, 2-3 = Moderate risk, 4 = High risk</h3>", unsafe_allow_html=True)
#st.header("1 = Low risk, 2-3 = Moderate risk, 4 = High risk")

# Streamlit web application
st.title("1. Decision Tree")
#st.write("1 = Low risk, 2-3 = Moderate risk, 4 = High risk")

# Predict the risk based on user input
prediction = classifier.predict(recent_user_input)[0]
#st.write("<h1 style='color: blue; display: inline;'>Predicted preoperative risk:</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color: red; display: inline;'>Predicted preoperative risk: {}</h1>".format(prediction), unsafe_allow_html=True)

from sklearn.ensemble import HistGradientBoostingClassifier
# Train a HistGradientBoostingClassifier
classifier2 = HistGradientBoostingClassifier()
classifier2.fit(features, target)

# Make predictions
prediction2 = classifier2.predict(recent_user_input)[0]
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
st.write("")
st.write("")
st.title("2. Gradient Booster")
st.markdown("<h1 style='color: red; display: inline;'>Predicted preoperative risk: {}</h1>".format(prediction2), unsafe_allow_html=True)

