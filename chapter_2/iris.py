import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from perceptron import Perceptron
import sys
sys.path.insert(0, "/Users/seojiwon/Desktop/codingtest/MLReview/chapter_3")
# from logit import LogisticRegressionGD

# import wandb

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



from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def plot_decision_regions(X, y, classifier, resolution =  0.02):

    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha = 0.8,
                    c = colors[idx],
                    marker = markers[idx],
                    label = cl, 
                    edgecolor="black")

# ppn = Perceptron()
plot_decision_regions(X, y, classifier=ppn)
plt.show()
# wandb.finish()

