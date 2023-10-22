import matplotlib
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

hept=pd.read_csv("C://Users/pm1afj/Desktop/heptathlon2016.csv", index_col="Name")
hept

events=hept.T.index
events

sns.set()
sns.pairplot(hept)

pca=PCA()
pca.fit(hept)

pca.explained_variance_ratio_

np.cumsum(pca.explained_variance_ratio_)

pca.components_

scaler=StandardScaler()
scaler.fit(hept)
hept1=scaler.transform(hept)
hept1

pca=PCA()
pca.fit(hept1)

pca.explained_variance_ratio_

np.cumsum(pca.explained_variance_ratio_)

pca.components_

cumexp=np.concatenate([[0],pca.explained_variance_ratio_])
plt.plot(np.cumsum(cumexp))
plt.xlabel("number of components")
plt.ylabel("cumulative variance")

p=pca.fit_transform(hept1)
plt.scatter(x=p[:,0],y=p[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")

eh=[12.54,1.86,14.28,22.83,6.48,47.49,128.65]
eh=np.reshape(eh,(1,-1))
eh1=scaler.transform(eh)
eh1=pca.transform(eh1)

plt.scatter(p[:,0], y=p[:,1])
plt.plot(eh1[:,0],eh1[:,1],"r*")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC1")
plt.ylabel("PC2")
x=p[:,0]
y=p[:,1]
coeff=np.transpose(pca.components_[0:2,:])
n=hept.shape[1]
scalex=1.0/(x.max()-x.min())
scaley=1.0/(y.max()-y.min())
plt.scatter(x*scalex,y*scaley)
for i in range(n):
    plt.arrow(0,0,coeff[i,0],coeff[i,1],color="r",alpha=0.5)
    plt.text(coeff[i,0]*1.15,coeff[i,1]*1.15,events[i],color="g")

