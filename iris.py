import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


s = os.path.join("http://archive.ics.uci.edu", "ml",
                "machine-learning-databases",
                "iris", "iris.data")
df = pd.read_csv(s,
                header=None,
                encoding="utf8")
print(df.tail())

#산점도 그리기

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

X = df.iloc[:100, [0, 2]].values #iloc[x값 범위, column 범위]

plt.scatter(X[:50, 0], X[:50, 1], 
            color = "red", marker = "o", label = "setosa")
plt.scatter(X[50:100, 0], X[50:100, 1],
            color = "blue", marker="x", label= "versicolor" )

plt.xlabel("sepal length")
plt.ylabel("petal length")

plt.legend(loc ="upper left")
plt.show()

ppn = Perceptron(learning_rate = 0.1, epochs = 20)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1),
                ppn.errors_, marker = "o")
plt.xlabel("Epochs")
plt.ylabel("Number of updates")
plt.show()
