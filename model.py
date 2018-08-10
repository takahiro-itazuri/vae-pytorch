import torch
from torch import nn
from torch.autograd import Variable

class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()

    self.fc1 = nn.Linear(784, 512)
    self.fc21 = nn.Linear(512, 2) # mu
    self.fc22 = nn.Linear(512, 2) # logvar

    self.fc3 = nn.Linear(2, 512)
    self.fc4 = nn.Linear(512, 784)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  
  def encode(self, x):
    h = self.relu(self.fc1(x))
    return self.fc21(h), self.fc22(h)
  
  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = Variable(std.data.new(std.size()).normal_())
      return eps.mul(std).add_(mu)
    else:
      return mu
  
  def decode(self, z):
    h = self.relu(self.fc3(z))
    return self.sigmoid(self.fc4(h))
  
  def forward(self, x):
    x = x.view(-1, 784)
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar
