%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

irisnf=pd.read_csv("C://Users/pm1afj/Desktop/irisnf.csv")
irisnf

plt.plot("Petal.l","Sepal.l",data=irisnf)

plt.plot("Petal.l","Sepal.l",data=irisnf,linestyle="none",marker="o")

plt.plot("Petal.l","Sepal.l",data=irisnf,alpha=0.5,linestyle="none",marker="o",markersize=10)

plt.plot("Petal.l","Sepal.l",data=irisnf,linestyle="none",marker="o",markeredgecolor="black",markerfacecolor="red")
plt.xlabel("Petal.l")
plt.ylabel("Sepal.l")
plt.xlim(0,7)
plt.ylim(3,10)
plt.xticks([])
plt.yticks([])
plt.title("Scatterplot")

plt.scatter("Petal.l","Sepal.l",data=irisnf,c="Variety")

plt.scatter("Petal.l","Sepal.l",data=irisnf,c="Variety",s="Sepal.w")

size=20*irisnf["Sepal.w"].values
plt.scatter("Petal.l","Sepal.l",data=irisnf,alpha=0.6,c="Variety",s=size)

plt.scatter("Petal.l","Sepal.l",data=irisnf,c="Variety")
plt.savefig("plot1.png")

plt.style.use("seaborn")
plt.scatter("Petal.l","Sepal.l",data=irisnf,c="Variety")

from pandas.plotting import andrews_curves
plt.figure()
andrews_curves(irisnf,"Variety")

