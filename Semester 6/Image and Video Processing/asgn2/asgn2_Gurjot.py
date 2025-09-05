import torch

class LinReg:
  def __init__(self, data):
    self.data=data
    K_x=[]
    K_y=[]

    for i in range(len(data)):
      K_x.append(data[i][0])
      K_y.append(data[i][1])

    self.x=torch.tensor(K_x,dtype=torch.float)
    self.y=torch.tensor(K_y,dtype=torch.float)
    self.weights=torch.tensor([0],dtype=torch.float,requires_grad=True)

  def get_weights(self) -> torch.tensor:
    return self.weights
    
# model output
  def forward(self, x):
    x=torch.tensor(x,dtype=torch.float)
    return self.weights*x


  def fit(self, learning_rate: float = 0.05, n_iters: int = 50000) -> None:
    loss=torch.nn.MSELoss()
    optimizer=torch.optim.SGD([self.weights], lr=learning_rate)
    #optimizer = torch.optim.Adam([self.weights], learning_rate)
    loss_track=[]
    epsilon=.00001

    for i in range(n_iters):
      optimizer.zero_grad()
      loss(self.forward(self.x),self.y).backward()
      optimizer.step()
      loss_track.append(loss(self.forward(self.x),self.y).item())

      if i>10:
        if loss_track[i-10]-loss_track[i]<epsilon:
          break

    return self.weights