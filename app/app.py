import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🚢 Titanic Dashboard")

df = pd.read_csv("data/cleaned_titanic.csv")

# Filters
sex = st.sidebar.multiselect("Gender", df["Sex"].unique(), df["Sex"].unique())
pclass = st.sidebar.multiselect("Class", df["Pclass"].unique(), df["Pclass"].unique())

filtered_df = df[(df["Sex"].isin(sex)) & (df["Pclass"].isin(pclass))]

st.write(filtered_df.head())

# Charts
st.subheader("Survival Count")
fig1, ax1 = plt.subplots()
sns.countplot(x="Survived", data=filtered_df, ax=ax1)
st.pyplot(fig1)

st.subheader("Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df["Age"], kde=True, ax=ax2)
st.pyplot(fig2)

st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(filtered_df.select_dtypes(include=["number"]).corr(), annot=True, ax=ax3)
st.pyplot(fig3)