import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

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
        st.write("You :red[**WOULD NOT**] be classified as a LinkedIn user :thumbsdown:")
    else:
        st.write("You :green[**WOULD**] be classified as a LinkedIn user :thumbsup:")
    probability = log_reg.predict_proba(data)

    labels = ['Does Not Use LinkedIn', 'Does Use LinkedIn']
    fig = go.Figure(
        data=[go.Pie(
            labels = labels,
            values=probability[0]
        )]
    )
    fig = fig.update_traces(
        hoverinfo='label+percent',
        textinfo='percent',
        textfont_size=15
    )
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("### Answer these questions to see if you would be classified as a LinkedIn User")

age = st.sidebar.number_input("Enter a number for your age:", step=1, help="Any whole number can be entered")

def convert_df_to_string(df):
    str = df.columns[0] + "=" + df.columns[1] + "\n\n"
    for i in range(df.shape[0]):
        str = str + df[df.columns[0]][i] + "=" + df[df.columns[1]][i] + "\n\n"
    return str

income_df = pd.DataFrame({
    'Income': ["Less than $10,000", "10 to under $20,000", "20 to under $30,000", 
               "30 to under $40,000", "40 to under $50,000", "50 to under $75,000",
               "75 to under $100,000", "100 to under $150,000", "$150,000 or more"],
    'Income Value': ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
})
income = st.sidebar.selectbox(
    "Select your income level",
    [1,2,3,4,5,6,7,8,9],
    help=convert_df_to_string(income_df)
)

education_df = pd.DataFrame({
    'Education': ["Less than high school", "High school incomplete", "High school graduate",
                  "Some college, no degree", "Two-year associate degree froom a college or university", 
                  "Four-year college or university degree/Bachelor's degree",
                  "Some postgraduate or professional schooling, no postgraduate degree",
                  "Postgraduate or professional degree, including master’s, doctorate, medical or law degree"],
    'Education Value': ["1","2","3","4","5","6","7","8"],
})
education = st.sidebar.selectbox(
    "Select your education level",
    [1,2,3,4,5,6,7,8],
    help=convert_df_to_string(education_df)
)

parent_df = pd.DataFrame({
    'Parent': ["no children", "children"],
    'Parent Value': ["0","1"],
})
parent = st.sidebar.selectbox(
    "Are you a parent?",
    [0,1],
    help=convert_df_to_string(parent_df)
)

married_df = pd.DataFrame({
    'Married': ["not married", "married"],
    'Married Value': ["0","1"],
})
married = st.sidebar.selectbox(
    "Are you married?",
    [0,1],
    help=convert_df_to_string(married_df)
)

gender_df = pd.DataFrame({
    'Gender': ["male", "female"],
    'Gender Value': ["0","1"],
})
female = st.sidebar.selectbox(
    "Are you male or female?",
    [0,1],
    help=convert_df_to_string(gender_df)
)

app(age, income, education, parent, married, female)