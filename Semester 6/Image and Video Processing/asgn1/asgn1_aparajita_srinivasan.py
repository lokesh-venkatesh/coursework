import numpy as np 

class LinReg:
    def __init__(self, data):
        '''
        Data is in the form [(x1,y1),...,(xn,yn)]
        '''
        self.data = np.array(data) #convert the input data into a numpy array
        self.weights = np.zeros(1) #initialise weights into a 1D numpy array with zeros with one entry

    def get_weights(self) -> np.ndarray:
        '''
        Function that returns the weights as an n-dimensional numpy array.
        The -> np.ndarray is a type hint indicating that what is returned is an ndarray.
        '''
        return self.weights

    def forward(self,x):
        '''
        Function for forward pass: Computes the output from the model y = wx for a given x.
        '''
        return x*self.weights[0] 

    @staticmethod
    def loss(y,y_pred) -> float:
        '''
        Function to calculate loss (MSE in this case).
        Output is a float.
        '''
        return np.mean((y-y_pred) ** 2)

    @staticmethod
    def gradient(x,y,y_pred) -> float:
        '''
        Computes gradient of loss wrt the weights.
        '''
        return -2 * np.mean(x * (y - y_pred)) 

    def fit(self, lr:float = 0.01, n_iters: int = 20) -> None:
        '''
        Use the weights and learning rate (lr) to train the model and get the fit
        '''
        for _ in range(n_iters):
            x,y = self.data[:,0], self.data[:,1] #split the input and output data 
            y_pred = self.forward(x) #get the predicted values
            print(self.weights)
            grad = self.gradient(x,y,y_pred) #compute the gradient wrt the weights
            self.weights -= lr * grad #update the weights based on grad and lr
            
if __name__ == '__main__':
    data = [(1,2),(2,4),(3,6)]

    model = LinReg(data)
    model.fit(lr=0.1, n_iters=20)
    print("Final weight:", model.get_weights()[0])

    #testing
    test_x = 4
    y_pred = model.forward(test_x)
    print(f"Prediction for x={test_x}: {y_pred}")