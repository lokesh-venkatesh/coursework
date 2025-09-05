import numpy as np
class LinReg:
    def __init__(self, data):
        self.x = np.array([i[0] for i in data])
        self.y = np.array([i[1] for i in data])
        self.weights = np.zeros(1)

    def get_weights(self) -> np.ndarray:
        return self.weights
# model output
    def forward(self, x):
        return self.weights * x

# loss = MSE
    @staticmethod
    def loss(y, y_pred) -> float:
        return np.mean((y-y_pred)**2)

# Gradient of loss with respect to weights.
    @staticmethod
    def gradient(x, y, y_pred) -> float:
        return -2*np.mean((y-y_pred)*x)
        
    def fit(self, learning_rate: float = 0.05, n_iters: int = 100) -> None:
        for i in range(n_iters):
            y_pred = self.forward(self.x)
            gradient = self.gradient(self.x, self.y, y_pred)
            self.weights += learning_rate*-gradient

