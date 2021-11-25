from typing import no_type_check
from sklearn import datasets
import numpy as np

iris =  datasets.load_iris()
X = iris.data[:, [2,3]]
y =  iris.target


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify = y
) #stratify => 훈련 데이터셋과 테스트 데이터셋의 클래스 레이블 비율을 입력 데이터셋과 동일하게 만듬

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""StandardScaler의 fit method는 훈련 데이터셋의 각 피쳐 차원마다 샘플 평균과 표준 편차를 계산, 훈련데이터셋을 표준화, """

from sklearn.linear_model import Perceptron

ppn =  Perceptron(eta0 = 0.1, random_state = 1)
ppn.fit(X_train_std, y_train)
y_pred =  ppn.predict(X_test_std)
print( "잘못 분류된 샘플 개수: %d" %((y_test != y_pred).sum()))

from sklearn.metrics import accuracy_score
print("정확도 : %.3f" % accuracy_score(y_test, y_pred))
print("정확도 : %.3f" % ppn.score(X_test_std, y_pred))

