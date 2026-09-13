# this ai converts celcius to farenheit
import numpy as np

class AI:
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def predict(self, x):
        return self.w * x + self.b

    def teach(self, celsius, real, epochs=10000):
            for _ in range(epochs):
                for x, y_real in zip(celsius, real):
                    y_pred = self.predict(x)
                    error = y_pred - y_real

                    self.w -= 0.0001 * error * x
                    self.b -= 0.0001 * error

x = [0, 10, 20, 100, 150, 200]
y = [32, 50, 68, 212, 302, 392]

ai = AI()

ai.teach(x, y)

while True:
    farenheit = int(input("Enter celsius: "))
    print("In farenheit: ", ai.predict(farenheit))