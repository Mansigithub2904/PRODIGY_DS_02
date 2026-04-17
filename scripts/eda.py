import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/cleaned_titanic.csv")

sns.set(style="whitegrid")

# Survival
sns.countplot(x="Survived", data=df)
plt.savefig("outputs/survival.png")
plt.close()

# Gender vs Survival
sns.barplot(x="Sex", y="Survived", data=df)
plt.savefig("outputs/gender_survival.png")
plt.close()

# Age Distribution
sns.histplot(df["Age"], bins=30, kde=True)
plt.savefig("outputs/age_distribution.png")
plt.close()

# Class vs Survival
sns.barplot(x="Pclass", y="Survived", data=df)
plt.savefig("outputs/class_survival.png")
plt.close()

# Heatmap
corr = df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr, annot=True)
plt.savefig("outputs/heatmap.png")
plt.close()

# Pairplot
sns.pairplot(df[["Survived", "Age", "Fare", "Pclass"]])
plt.savefig("outputs/pairplot.png")
plt.close()

print("✅ EDA completed!")