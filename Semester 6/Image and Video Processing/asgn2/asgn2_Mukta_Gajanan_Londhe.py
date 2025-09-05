import torch

class LinReg:
    def __init__(self, data):
        tensor_data = torch.tensor(data, dtype=torch.float32)  # Convert entire data at once
        self.x, self.y = tensor_data[:, 0].to(torch.float), tensor_data[:, 1].to(torch.float)
        self.weights = torch.randn(1, requires_grad=True)

    def get_weights(self) -> torch.tensor:
        return self.weights


# model output
    def forward(self, x):
        return self.weights * x

    def fit(self, learning_rate: float = 0.005, n_iters: int = 10000) -> None:
        optimizer = torch.optim.SGD([self.weights], lr=learning_rate) 
        loss_fn = torch.nn.MSELoss() 

        for _ in range(n_iters):
            loss = loss_fn(self.forward(self.x), self.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()