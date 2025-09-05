import numpy as np
import json
#only library used was numpy==2.2.2

class LinReg:
    def __init__(self, data):
        """
        Initializes the object with data and sets initial weights.
        
        Parameters:
            data (any): The input data to be stored in the object.
        """

        self.data = data
        self.x_data = np.array([pair[0] for pair in self.data])
        self.y_data = np.array([pair[1] for pair in self.data])
        self.parameters = np.array([0.0])
    

    def get_weights(self) -> np.ndarray:
        """
        Returns the weights of the model that have been stored in the object.

        Returns:
            np.ndarray: The weights of the model.
        """

        return self.parameters


    def forward(self, x) -> np.ndarray:
        """
        Does one forward pass and returns the corresponding output, which is simply the prediction.

        Parameters:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the model.
        """

        return self.parameters*x


    @staticmethod
    def loss(y, y_pred) -> float:
        """
        Calculates the MSE (Mean Square Error) loss function for a given prediction and the original data.

        Parameters:
            y (np.ndarray): The original data.
            y_pred (np.ndarray): The predicted data.

        Returns:
            float: The loss function value.
        """

        return np.mean((y-y_pred)**2)
    

    @staticmethod
    def gradient(x, y, y_pred) -> float:
        """
        Finds the gradient of the loss function with respect to the weights.

        Parameters:
            x (np.ndarray): The input data.
            y (np.ndarray): The original data.
            y_pred (np.ndarray): The predicted data.

        Returns:
            float: The gradient of the loss function with respect to the weights.
        """

        return -2*np.mean(x*(y-y_pred))


    def fit(self, learning_rate:float=0.001, n_iters:int=10000) -> None:
        """
        Fits the model to the data by updating the weights.

        Parameters:
            learning_rate (float): The learning rate for the model.
            n_iters (int): The number of iterations to run the model for.
        """

        for _ in range(n_iters):
            x = self.x_data
            y = self.y_data
            y_pred = self.forward(x)

            grad = self.gradient(x, y, y_pred)
            #print(self.weights)
            self.parameters = self.parameters - learning_rate*grad

if __name__ == "__main__":
    with open('asgn1_data_publish.json', 'r') as file:
        master_dataset = json.load(file)

    all_weights = []

    for i in range(len(master_dataset)):
        example_data = master_dataset[i]
        example_model = LinReg(example_data)
        example_model.fit()
        all_weights.append(example_model.get_weights().item())
    print(all_weights)

    with open('asgn1_results.json', 'w') as outfile:
        json.dump(all_weights, outfile)