import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

try:
    model = joblib.load("titanic_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("❌ Model files not found!")
    st.stop()

st.title("🚢 Titanic Survival Prediction App")
st.markdown("---")

st.write("Enter passenger details:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0, 500, 100)

sex = 1 if sex == "Male" else 0

if st.button("Predict Survival"):

    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]],
                             columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    st.subheader("📊 Passenger Details")
    st.write(input_data)
    
    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")

    st.write(f"Survival Probability: {prob[0][1]:.2%}")

    st.progress(int(prob[0][1] * 100))

if st.button("Reset"):
    st.experimental_rerun()

st.markdown("---")
st.header("📊 Data Insights Dashboard")

if st.checkbox("Show Data Insights"):
    
    st.markdown("### Explore Titanic Data")
    
    st.subheader("📊 Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df, ax=ax)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Did Not Survive","Survived"])
    st.pyplot(fig)

    st.subheader("👨‍👩‍👧 Survival by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("🎟 Survival by Class")
    fig, ax = plt.subplots()
    sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)