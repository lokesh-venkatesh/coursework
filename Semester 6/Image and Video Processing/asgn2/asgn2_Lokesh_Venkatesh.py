import torch

class LinReg:
    def __init__(self, data):
        """
        Initializes the object with data and sets initial weights.
        
        Parameters:
            data (any): The input data to be stored in the object.
        """
        self.data = torch.tensor(data, dtype=torch.float64)
        self.x_data = torch.tensor([tup[0] for tup in data], dtype=torch.float64)
        self.y_data = torch.tensor([tup[1] for tup in data], dtype=torch.float64)
        self.weights = torch.randn(1, requires_grad=True)  # Initialize weight randomly
    
    def get_weights(self) -> torch.tensor:
        """
        Returns the weights of the model as a torch tensor.

        Returns:
            torch.tensor: The weights of the model.
        """
        return self.weights

    def forward(self, x):
        """
        Computes the forward pass of the model.

        Parameters:
            x (float or numpy.ndarray): Input data.
        Returns:
            float or numpy.ndarray: The result of the model's linear transformation on the input data.
        """
        return self.weights*x

    def fit(self, learning_rate: float = 0.001, n_iters: int = 10000) -> None:
        """
        Trains the model using gradient descent with torch.optim.SGD.

        Args:
            learning_rate (float): The learning rate for the optimizer. Default is 0.001.
            n_iters (int): The number of iterations for training. Default is 10000.
        Returns:
            None
        """
        optimizer = torch.optim.SGD([self.weights], lr=learning_rate)
        criterion = torch.nn.MSELoss()

        for _ in range(n_iters):
            loss = 0
            y_pred = self.forward(self.x_data)
            loss += criterion(y_pred, self.y_data)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()