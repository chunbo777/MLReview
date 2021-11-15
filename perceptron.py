import numpy as np

class Perceptron(object):
    def __init__(self, learning_rate = 0.01, epochs =50, random_state =1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        randomgen = np.random.RandomState(self.random_state)
        self.w_ = randomgen.normal(loc = 0.0, scale = 0.01,
                                    size = 1 + X.shape[1]) #사이즈 1+ 훈련트레인셋 칼럼 갯수
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate*(target - self.predict(xi))
                self.w_[1:] +=update*xi #가중치의 변화율
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
