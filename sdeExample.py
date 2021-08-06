import torch
from torch import nn

import matplotlib.pyplot as plt

import torchsde

def plot(ts, samples, xlabel, ylabel, title=''):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, marker='x', label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    #plt.show()

## SDE 1D:
class SDE_1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # Scalar parameter.
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        return torch.sin(t) + self.theta * y

    def g(self, t, y):
        return 0.3 * torch.sigmoid(torch.cos(t) * torch.exp(-y))

batch_size, state_size, t_size = 3, 1, 100
sde = SDE_1D()
ts = torch.linspace(0, 1, t_size)
y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).

plot(ts, ys, xlabel='$t$', ylabel='$Y_t$')

## SDE 3D:
batch_size, state_size, brownian_size = 4, 3, 2
t_size = 200

class SDE_3D(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()

        self.mu1 = torch.nn.Linear(1, 1)
        self.noiseStd = torch.tensor([1, 10, 100]) # for state_size=3

    # Drift
    def f(self, t, y):
        state_size = y.shape[1]
        for s in range(state_size):
            if s == 0:
                out = self.mu1(y[:, s:s+1])
            else:
                out = torch.cat((out, self.mu1(y[:, s:s+1])), dim=1)
        return out  # shape (batch_size, state_size)


    # Diffusion
    def g(self, t, y):
        batch_size = y.shape[0]
        return self.noiseStd[None, :].repeat(batch_size,1)

sde = SDE_3D()
y0 = torch.full((batch_size, state_size), 0.1)
ts = torch.linspace(0, 1, t_size)
# Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
# ys will have shape (t_size, batch_size, state_size)
with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts)

for s in range(state_size):
    plot(ts, ys[:, :, s], xlabel='$t$', ylabel=f'Y_t[{s}]')


plt.show()
