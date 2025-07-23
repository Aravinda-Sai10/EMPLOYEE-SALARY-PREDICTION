# Import required libraries
import streamlit as st
import pandas as pd
import joblib
import base64

# Load model and encoders
model = joblib.load("model/salary_model.joblib")
encoders = joblib.load("model/encoders.joblib")

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

#custom CSS styles
st.markdown("""
    <style>
        /* Transparent white container */
        .transparent-box {
            background-color: rgba(255, 255, 255, 0.4);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
        }

        /* Title - black */
        .transparent-box h1 {
            color: black;
            font-weight: bold;
            text-align: center;
        }

        /* Paragraph - black */
        .transparent-box p {
            color: black;
            font-size: 18px;
            text-align: center;
        }

        /* Predict Button (stay same - white with premium look) */
        div.stButton > button {
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            border: 2px solid #000;
            font-weight: bold;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        }

        div.stButton > button:hover {
            background-color: #f0f0f0;
            color: #000;
            border: 2px solid #333;
        }

        /* Sidebar stay dark or premium as before */
        section[data-testid="stSidebar"] {
            background-color: #202020;
            color: white;
        }

        /* Sidebar widgets (dropdowns etc.) */
        section[data-testid="stSidebar"] .stSelectbox, 
        section[data-testid="stSidebar"] .stNumberInput {
            background-color: #333333;
            color: white;
            border-radius: 10px;
        }

        /* Make dropdown text white */
        .css-1wa3eu0-placeholder {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
# title
st.markdown("""
    <div class="transparent-box">
        <h1>💼 Employee Salary Prediction</h1>
        <p>This app predicts whether an employee earns more than 50K based on the entered details.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for input
with st.sidebar:
    st.header("Enter Employee Details")
    age = st.number_input("Age", min_value=18, max_value=100)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"])
    education = st.selectbox("Education", ["Bachelors", "HS-grad", "Masters", "Some-college", "Assoc-acdm"])
    education_num = st.number_input("Education-Num", min_value=1, max_value=20)
    marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales"])
    relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child"])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander"])
    gender = st.selectbox("gender", ["Male", "Female"])
    hours_per_week = st.slider("Hours/Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany"])
    age = st.sidebar.slider("Age", 17, 90, 30)
    fnlwgt = st.sidebar.number_input("Fnlwgt", min_value=10000, max_value=1000000, value=50000)
    education_num = st.sidebar.slider("Education-Num", 1, 16, 10)
    capital_gain = st.sidebar.number_input("Capital Gain", value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", value=0)
    hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

# Build input data
input_data = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'education': encoders['education'].transform([education])[0],
    'education-num': education_num,
    'marital-status': encoders['marital-status'].transform([marital_status])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([native_country])[0],
}

input_df = pd.DataFrame([input_data])


if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    predicted_label = encoders['income'].inverse_transform([prediction])[0]
    st.success(f"✅ Predicted Salary Range: **{predicted_label}**")

    # Download CSV
    input_df['Predicted Salary'] = predicted_label
    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Prediction", csv, "salary_prediction.csv", mime='text/csv')

st.markdown("</div>", unsafe_allow_html=True)
