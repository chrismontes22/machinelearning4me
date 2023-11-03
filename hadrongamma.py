import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


cols = ["fLength", "fwidth", "fsize", "fconc","fconc1", "fasym", "fM3Long","fm3Trans","falpha", "fdist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)
print(df.head())




for label in cols[:-1]:
  plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density='true')
  plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density='true')
  plt.title(label)
  plt.ylabel("probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()
  
  #working with datasets
train, valid, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

def scale_dataset(data_frame):
  x=dataframe[dataframe.cols[:-1]].values
  y=dataframe[dataframe.cols[-1]].values

  scarler = StandardScale()
  x = scaler.fit_transform(x)
  data = np.hstack((x,np.reshape(y, (-1,1))))

  return data, x,y

print(len(train[train["class"]==1]))
print(len(train[train["class"]==0]))

print(train)


#kNN

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)
print(y_pred)