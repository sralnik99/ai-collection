# this ai convers farenheit to celsius
import numpy as np

class AI:
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def predict(self, x):
        return self.w * x + self.b

    def teach(self, celsius, real, epochs=100000):
            for _ in range(epochs):
                for x, y_real in zip(celsius, real):
                    y_pred = self.predict(x)
                    error = y_pred - y_real

                    self.w -= 0.0001 * error * x
                    self.b -= 0.0001 * error

x_bad = [32, 50, 68, 212, 302, 392]
y = [0, 10, 20, 100, 150, 200]

x = []

for x_norm in x_bad:
    x.append(x_norm / max(x_bad))

ai = AI()

ai.teach(x, y)

while True:
    celsius = int(input("Enter farenheit: "))
    print("In celsius: ", ai.predict(celsius / 392))