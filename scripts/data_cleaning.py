import pandas as pd
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/titanic.csv")

# Drop irrelevant columns
df = df.drop(columns=["Cabin", "Ticket", "Name"])

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["AgeGroup"] = pd.cut(df["Age"],
                       bins=[0, 12, 18, 35, 60, 100],
                       labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

df.to_csv("data/cleaned_titanic.csv", index=False)

print("✅ Data cleaned successfully!")