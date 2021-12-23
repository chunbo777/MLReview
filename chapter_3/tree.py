from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('클래스 레이블:', np.unique(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # 마커와 컬러맵을 설정합니다.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor=None if idx==1 else 'black')
def gini(p): #지니 불순도
    return p * (1 - p) + (1 - p) *(1 - (1 - p))

def entropy(p): #엔트로피
    return -p*np.log2(p) -(1-p)*np.log((1-p))

def error(p): #분류 오차
    return 1 - np.max([p, 1-p])

x = np.arange(0.0, 1.0, 0.01) # 0부터 1까지 0.01 간격의 array 형성 

ent = [entropy(p) if p!= 0 else None for p in x]
sc_ent = [ e*0.5 if e else None for e in ent]
err = [error(i) for i in x]

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

# fig = plt.figure()
# ax = plt.subplot(111)
# for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
#                           ['Entropy', 'Entropy (scaled)', 
#                            'Gini impurity', 'Misclassification error'],
#                           ['-', '-', '--', '-.'],
#                           ['black', 'lightgray', 'red', 'green', 'cyan']):
#     line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#           ncol=5, fancybox=True, shadow=False)

# ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
# ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
# plt.ylim([0, 1.1])
# plt.xlabel('p(i=1)')
# plt.ylabel('impurity index')
# plt.savefig('images/03_19.png', dpi=300, bbox_inches='tight')
# plt.show()


# 결정 트리 만들기

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(criterion="gini",
                                    max_depth=4,
                                    random_state=1)
tree_model.fit(X_train, y_train)

X_combined =  np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree_model,
                      test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/03_20.png', dpi=300)
plt.show()


# from sklearn import tree

# plt.figure(figsize=(10,10))
# tree.plot_tree(tree_model)
# plt.show()