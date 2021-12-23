import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
import sys
sys.path.insert(0, "/Users/seojiwon/Desktop/codingtest/MLReview/chapter_2")
# from chapter_2.iris import plot_decision_regions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
# from iris import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('클래스 레이블:', np.unique(y))
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

 ############
X_combined_std =  np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# svm = SVC(kernel = "linear", C=1.0, random_state=1)
# svm.fit(X_train_std,y_train)
# plot_decision_regions(X = X_combined_std,
#                     y = y_combined,
#                     classifier=svm)
#                     # test_idx = range(105,150))
# plt.xlabel("petal length")
# plt.ylabel("petal width")
# plt.legend(loc = "upper left")
# plt.tight_layout()
# plt.show()







np.random.seed(1)
X_xor =  np.random.randn(200,2)
y_xor =  np.logical_xor(X_xor[:, 0] > 0,
                        X_xor[:, 1] > 0 )
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == -1, 1],
            c= "b", marker="x",
            label = "1")
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c= "r", 
            marker="s",
            label = "-1")

"""
def scatter(
        x, y, s=None, c=None, marker=None, cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None, *,
        edgecolors=None, plotnonfinite=False, data=None, **kwargs):
    __ret = gca().scatter(
        x, y, s=s, c=c, marker=marker, cmap=cmap, norm=norm,
        vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths,
        edgecolors=edgecolors, plotnonfinite=plotnonfinite,
        **({"data": data} if data is not None else {}), **kwargs)
    sci(__ret)
    return __ret"""
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc = "best")
plt.tight_layout()
plt.show()

svm = SVC(kernel = "rbf", random_state = 1, gamma = 0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X = X_combined_std,
                    y = y_combined,
                    classifier=svm)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()




