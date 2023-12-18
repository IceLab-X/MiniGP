import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class cdf(nn.Module):
    def __init__(self,concentration1_prior, concentration0_prior):
        super().__init__()
        self.concentration1_prior=concentration1_prior
        self.concentration0_prior=concentration0_prior
        self.low = torch.full_like(self.concentration0_prior, 0)
        self.high = torch.full_like(self.concentration0_prior, 1)

    def forward(self, x):
        x= -1 * (x.clamp(0, 1) - 0.5) + 0.5
        x = x * (self.high - self.low) + self.low
        x= x.pow(self.concentration0_prior.reciprocal())
        x= 1-x
        x=x.pow(self.concentration1_prior.reciprocal())
        return x

    def inverse(self, y):
        y= y.pow(1 / self.concentration1_prior.reciprocal())
        y= 1-y
        y= y.pow(1 / self.concentration0_prior.reciprocal())
        y= -1 * (y - 0.5) + 0.5
        return y

class WarpLayer(nn.Module):
    def __init__(self, warp_func, if_trainable=False):
        super(WarpLayer, self).__init__()
        self.warp_func = warp_func
        self.if_trainable = if_trainable

        if self.if_trainable:
            for param in self.warp_func.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.warp_func(x)

    def inverse(self, y):
        return self.warp_func.inverse(y)

if __name__ == '__main__':
    fontdict = {"fontsize": 15}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    torch.manual_seed(1)
    c1 = torch.rand(6, dtype=dtype, device=device) * 3 + 0.1
    c0 = torch.rand(6, dtype=dtype, device=device) * 3 + 0.1
    x = torch.linspace(0, 1, 101, dtype=dtype, device=device)

    mywrap=WarpLayer(cdf(c1,c0))
    k_icdfs=mywrap.forward(x.unsqueeze(1).expand(101, 6))

    k_cdfs=mywrap.inverse(k_icdfs)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for i in range(6):
        ax.plot(x.cpu(), k_icdfs[:, i].cpu())
    ax.set_xlabel("Raw Value", **fontdict)
    ax.set_ylabel("Transformed Value", **fontdict)
    plt.show()

