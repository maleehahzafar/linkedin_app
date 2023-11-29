import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.markdown("# Do You Use LinkedIn?")
st.write("Find out whether you are likely to use LinkedIn or Not based on your qualities")

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = pd.DataFrame()
ss["sm_li"] = clean_sm(s.web1h)
ss["income"] = s[["income"]]
ss["education"] = s[["educ2"]]
ss["parent"] = clean_sm(s.par)
ss["married"] = clean_sm(s.marital)
ss["female"] = np.where(s.sex == 2, 1, 0)
ss["age"] = s[["age"]]
ss = ss[(ss["income"] <= 9) & (ss["education"] <= 8) & (ss["age"] < 98)]

y = ss.sm_li
X = ss[['income', 'age', 'education', 'parent', 'married', 'female']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=14)

log_reg = LogisticRegression(random_state=14, class_weight="balanced")
log_reg.fit(X_train, y_train)

def app(age, income, education, parent, married, female):
    data = pd.DataFrame([[income, age, education, parent, married, female]],
                    columns=['income','age','education','parent','married','female'])
    prediction = log_reg.predict(data)
    if prediction[0] == 0:
        st.write("You would NOT be classified as a LinkedIn user")
    else:
        st.write("You would be classified as a LinkedIn user")
    probability = log_reg.predict_proba(data)

    labels = 'Does Not Use LinkedIn', 'Does Use LinkedIn'
    sizes = [round(probability[0][0]*100,2),round(probability[0][1]*100,2)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    return st.pyplot(fig1)

st.markdown("## Answer the following questions to see if our predictions are correct.")
st.write("Go to the **Qualities** section for explanations on the values.")

age = st.number_input("Enter a number for your age:", step=1)
income = st.selectbox(
   "Select your income level",
   (1,2,3,4,5,6,7,8,9),
   index=0,
   placeholder="Select income level...",
)
education = st.selectbox(
   "Select your education level",
   (1,2,3,4,5,6,7,8),
   index=0,
   placeholder="Select education level...",
)
parent = st.radio(
    "Are you a parent?",
    [0,1],
    index=0,
)
married = st.radio(
    "Are you married?",
    [0,1],
    index=0,
)
female = st.radio(
    "Are you male or female?",
    [0,1],
    index=0,
)

app(age, income, education, parent, married, female)

st.markdown("## Qualities:")
st.write(pd.DataFrame({
    'Income': ["Less than $10,000", "10 to under $20,000", "20 to under $30,000", 
               "30 to under $40,000", "40 to under $50,000", "50 to under $75,000",
               "75 to under $100,000", "100 to under $150,000", "$150,000 or more"],
    'Income Value': [1,2,3,4,5,6,7,8,9],
}))
st.write(pd.DataFrame({
    'Education': ["Less than high school", "High school incomplete", "High school graduate",
                  "Some college, no degree", "Two-year associate degree froom a college or university", 
                  "Four-year college or university degree/Bachelor's degree",
                  "Some postgraduate or professional schooling, no postgraduate degree",
                  "Postgraduate or professional degree, including master’s, doctorate, medical or law degree"],
    'Education Value': [1,2,3,4,5,6,7,8],
}))
st.write(pd.DataFrame({
    'Gender': ["male", "female"],
    'Gender Value': [0,1],
}))
st.write(pd.DataFrame({
    'Married': ["not married", "married"],
    'Married Value': [0,1],
}))
st.write(pd.DataFrame({
    'Parent': ["no children", "children"],
    'Parent Value': [0,1],
}))
st.write("Age")