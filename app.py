import streamlit as st
import pandas as pd
import pickle
import joblib

# ---------------------- PAGE ----------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------------- LOAD MODEL ----------------------
model = joblib.load("model.pkl", "rb")
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------------- CSS ----------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg, #E0F7FA, #BBDEFB, #E8F5E9);
}

.main-title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#0b7285;
}

.sub{
text-align:center;
color:gray;
font-size:18px;
margin-bottom:25px;
}

[data-testid="stMetric"]{
background:white;
padding:15px;
border-radius:15px;
box-shadow:0px 5px 10px rgba(0,0,0,.12);
}

.stButton>button{
width:100%;
background:#009688;
color:white;
font-size:20px;
border-radius:10px;
height:55px;
border:none;
}

.stButton>button:hover{
background:#00695c;
color:white;
}

.result{
padding:20px;
border-radius:15px;
font-size:24px;
font-weight:bold;
text-align:center;
}

.good{
background:#2ecc71;
color:white;
}

.bad{
background:#e74c3c;
color:white;
}

</style>
""",unsafe_allow_html=True)

# ---------------------- TITLE ----------------------

st.markdown("<div class='main-title'>🩺 Diabetes Prediction Dashboard</div>",unsafe_allow_html=True)
st.markdown("<div class='sub'>Machine Learning Based Health Prediction System</div>",unsafe_allow_html=True)

# ---------------------- METRICS ----------------------

m1,m2,m3,m4=st.columns(4)

with m1:
    st.metric("Model","Random Forest")

with m2:
    st.metric("Inputs","8")

with m3:
    st.metric("Prediction","Binary")

with m4:
    st.metric("Status","Ready")

st.divider()

# ---------------------- INPUTS ----------------------

left,right=st.columns(2)

with left:

    st.subheader("👤 Personal Information")

    gender=st.selectbox(
        "Gender",
        ["Male","Female","Other"]
    )

    age=st.slider(
        "Age",
        1,
        100,
        25
    )

    hypertension=st.radio(
        "Hypertension",
        [0,1],
        horizontal=True
    )

    heart_disease=st.radio(
        "Heart Disease",
        [0,1],
        horizontal=True
    )

with right:

    st.subheader("🧪 Medical Information")

    smoking=st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "ever",
            "not current"
        ]
    )

    bmi=st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )

    hba1c=st.number_input(
        "HbA1c Level",
        3.0,
        15.0,
        5.5
    )

    glucose=st.number_input(
        "Blood Glucose",
        50,
        300,
        100
    )

st.divider()

# ---------------------- BUTTON ----------------------

if st.button("🔍 Predict Diabetes"):

    input_df=pd.DataFrame(
        [[
            gender,
            age,
            hypertension,
            heart_disease,
            smoking,
            bmi,
            hba1c,
            glucose
        ]],
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]
    )

    input_df=pd.get_dummies(
        input_df,
        columns=[
            "gender",
            "smoking_history"
        ],
        drop_first=True
    )

    input_df=input_df.reindex(
        columns=columns,
        fill_value=0
    )

    prediction=model.predict(input_df)

    st.divider()

    if prediction[0]==1:

        st.markdown(
        "<div class='result bad'>⚠️ Patient is Likely Diabetic</div>",
        unsafe_allow_html=True
        )

        st.progress(90)

        st.warning(
            "Consult a doctor for further diagnosis."
        )

    else:

        st.markdown(
        "<div class='result good'>✅ Patient is Not Diabetic</div>",
        unsafe_allow_html=True
        )

        st.progress(20)

        st.success(
            "Maintain a healthy lifestyle."
        )

st.divider()

st.caption("Developed using ❤️ Streamlit + Machine Learning")
