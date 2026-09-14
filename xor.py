# this ai is predicting how XOR gate is working
import numpy as np

class AI:
    def __init__(self):
        # hidden layer
        # first neuron
        self.w11 = np.random.randn() 
        self.w12 = np.random.randn() 

        # second neuron
        self.w21 = np.random.randn() 
        self.w22 = np.random.randn() 

        # for first neuron
        self.b1 = np.random.randn() 
        # for second neuron
        self.b2 = np.random.randn() 

        # output layer
        self.w31 = np.random.randn() 
        self.w32 = np.random.randn() 
        self.b3 = np.random.randn() 

    def predict(self, x1, x2):
        self.h1 = self.sigmoid(self.w11 * x1 + self.w21 * x2 + self.b1)
        self.h2 = self.sigmoid(self.w12 * x1 + self.w22 * x2 + self.b2)
        return self.sigmoid(self.w31 * self.h1 + self.w32 * self.h2 + self.b3)

    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, a, b, output, epochs=100000, lr=0.100):
        for _ in range(epochs):
            for a_out, b_out, real in zip(a, b, output):
                    y_pred = self.predict(a_out, b_out)
                    delta_out = (y_pred - real) * y_pred * (1 - y_pred)

                    delta_h1 = (delta_out * self.w31) * self.h1 * (1 - self.h1)
                    delta_h2 = (delta_out * self.w32) * self.h2 * (1 - self.h2)

                    self.w11 -= lr * delta_h1 * a_out
                    self.w12 -= lr * delta_h2 * a_out

                    self.w21 -= lr * delta_h1 * b_out
                    self.w22 -= lr * delta_h2 * b_out

                    self.w31 -= lr * delta_out * self.h1
                    self.w32 -= lr * delta_out * self.h2

                    self.b1 -= lr * delta_h1
                    self.b2 -= lr * delta_h2
                    self.b3 -= lr * delta_out

a = [0, 0, 1, 1]
b = [0, 1, 0, 1]
output = [0, 1, 1, 0]

ai = AI()

ai.train(a, b, output)

print(ai.predict(0, 1))
