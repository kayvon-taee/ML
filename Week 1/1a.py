import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")

airpoll = pd.read_csv("C://Users/pm1afj/Desktop/airpoll.csv")
airpoll

airpoll = pd.read_csv("C://Users/pm1afj/Desktop/airpoll.csv", index_col="City")
airpoll

rownames = airpoll.index
rownames

colnames = airpoll.columns
colnames

airpoll.shape

airpoll["NOX"]

features = ["Rainfall", "Education", "Popden", "Nonwhite", "NOX", "SO2"]
X = airpoll[features]
y = airpoll["Mortality"]

X

y

airpoll["SO2"] > 100

airpoll.head(4)

airpoll.iloc[3, 5]

airpoll.iloc[0, 0]

airpoll.iloc[[1, 3], 2:6]

airpoll.iloc[3]

airpoll.loc[airpoll["SO2"] > 100]

airpoll.describe()

airpoll.describe(include="all")

airpoll.cov()

airpoll.corr()
