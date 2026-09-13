# this ai is predicting how OR gate is working
import numpy as np

class AI:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.b = -0.5

    def predict(self, x1, x2):
        return self.sigmoid(self.w1 * x1 + self.w2 * x2 + self.b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, a, b, output, epochs=100000):
        for _ in range(epochs):
            for a_out, b_out, real in zip(a, b, output):
                    y_pred = self.predict(a_out, b_out)
                    error = y_pred - real

                    delta = error * y_pred * (1 - y_pred)

                    self.w1 -= 0.1 * delta * a_out
                    self.w2 -= 0.1 * delta * b_out
                    self.b -= 0.1 * delta

a = [0, 0, 1, 1]
b = [0, 1, 0, 1]
output = [0, 1, 1, 1]

ai = AI()

ai.train(a, b, output)

print(ai.predict(0, 0))