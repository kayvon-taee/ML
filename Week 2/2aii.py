%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

irisnf=pd.read_csv("C://Users/pm1afj/Desktop/irisnf.csv")
irisnf

sns.set()

sns.scatterplot(x="Petal.l", y="Sepal.l", data=irisnf)

sns.scatterplot(x="Petal.l", y="Sepal.l", hue="Variety", data=irisnf)

sns.scatterplot(x="Petal.l", y="Sepal.l", style="Variety", data=irisnf)

sns.scatterplot(x="Petal.l", y="Sepal.l", hue="Variety", style="Variety", data=irisnf)

sns.lmplot(x="Petal.l", y="Sepal.l", data=irisnf)

sns.regplot(x="Petal.l", y="Sepal.l", marker="o", fit_reg=True, data=irisnf)

p=sns.scatterplot(x="Petal.l", y="Sepal.l", marker=".", data=irisnf)
for line in range(0,irisnf.shape[0]):
    l=irisnf.iloc[line]
    p.text(l["Petal.l"],l["Sepal.l"],int(l["Variety"]))

p=sns.scatterplot(x="Petal.l", y="Sepal.l", data=irisnf)
p.figure.suptitle("Plot of lengths", fontsize=16)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Sepal length", fontsize=14)

sns.scatterplot(x="Petal.l", y="Sepal.l", size="Petal.w", data=irisnf)

sns.pairplot(irisnf)

sns.pairplot(irisnf).savefig("plot.png")

p=sns.FacetGrid(data=irisnf,col="Variety")
p=(p.map(plt.scatter, "Petal.l", "Sepal.l").add_legend())

sns.jointplot(x="Petal.l", y="Sepal.l", data=irisnf)

sns.jointplot(x="Petal.l", y="Sepal.l", data=irisnf, kind="kde")

