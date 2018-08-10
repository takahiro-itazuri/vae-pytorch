import os
import torch
import torchvision
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model import VAE

def loss_function(xhat, x, mu, logvar):
  bce = F.binary_cross_entropy(xhat, x.view(-1, 784), size_average=False)
  kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return bce + kld

def train(epoch, model, optimizer, data_loader):
  model.train()
  train_loss = 0
  for img, _ in data_loader:
    if use_gpu:
      x = Variable(img).cuda()
    else:
      x = Variable(img)
    
    optimizer.zero_grad()
    xhat, mu, logvar = model(x)
    loss = loss_function(xhat, x, mu, logvar)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  
  train_loss /= len(data_loader.dataset)
  return train_loss

def test(epoch, model, data_loader):
  model.eval()
  test_loss = 0
  for idx, (img, _) in enumerate(data_loader):
    if use_gpu:
      x = Variable(img).cuda()
    else:
      x = Variable(img)
    
    xhat, mu, logvar = model(x)
    loss = loss_function(xhat, x, mu, logvar)
    test_loss += loss.item()

    if epoch % 10 == 0:
      if idx == 0:
        n = 10
        comparison = torch.cat([x[:n], xhat.view(batch_size, 1, 28, 28)[:n]])
        save_image(comparison.data.cpu(), '{}/reconstruction_{}.png'.format(log_dir, epoch), nrow=n)

  test_loss /= len(data_loader.dataset)
  return test_loss

use_gpu = torch.cuda.is_available()
batch_size = 128
num_epochs = 100
learning_rate = 1e-3
seed = 1
log_dir = 'logs'
data_dir = 'data'

if not os.path.exists(log_dir):
  os.makedirs(log_dir)

if not os.path.exists(data_dir):
  os.makedirs(data_dir)

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor()), 
  batch_size=batch_size, 
  shuffle=True
)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor()),
  batch_size=batch_size,
  shuffle=True
)

torch.manual_seed(seed)
if use_gpu:
  torch.cuda.manual_seed(seed)

model = VAE()
if use_gpu:
  model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_list = []
test_loss_list = []
for epoch in range(num_epochs+1):
  # training
  train_loss = train(epoch, model, optimizer, train_loader)
  loss_list.append(train_loss)

  # test
  test_loss = test(epoch, model, test_loader)
  test_loss_list.append(test_loss)

  print('epoch [{}/{}], loss: {:.4f}, test_loss: {:.4f}'.format(epoch, num_epochs, train_loss, test_loss))

# visualize loss
plt.plot(loss_list, label='loss')
plt.plot(test_loss_list, label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.legend()
plt.savefig('./{}/loss.png'.format(log_dir))

# visualize latent space
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST(data_dir, train=False, download=True, transform = transforms.ToTensor()),
  batch_size=10000,
  shuffle=True
)

x, labels = iter(test_loader).next()
x = x.view(10000, -1)
if use_gpu:
  x = Variable(x).cuda()
else:
  x = Variable(x)

z = model.encode(x)
mu, logvar = z
if use_gpu:
  mu, logvar = mu.cpu().data.numpy(), logvar.cpu().data.numpy()
else:
  mu, logvar = mu.data.numpy(), logvar.data.numpy()

plt.figure(figsize=(10, 10))
plt.scatter(mu[:, 0], mu[:, 1], marker='.', c=labels.numpy(), cmap=plt.cm.jet)
plt.colorbar()
plt.grid()
plt.savefig('./{}/latent_space.png'.format(log_dir))

np.save('loss_list.npy', np.array(loss_list))
np.save('test_loss_list.npy', np.array(test_loss_list))
torch.save(model.state_dict(), 'model_weights.pth')
