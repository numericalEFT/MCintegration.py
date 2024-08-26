import torch
import numpy as np
from scipy.special import erf, gamma
import normflows as nf


# from nf import distributions
class Sharp(nf.distributions.Target):
    def __init__(self, batchsize):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        self.ndims = 2
        self.targetval = 4.0
        self.batchsize = batchsize
        self.samples = batchsize
    def prob(self, x):
        return -torch.log(x[:, 0]) / torch.sqrt(x[:, 0])

    def log_prob(self, x):
        return torch.log(torch.abs(self.prob(x)))


class Tight(nf.distributions.Target):
    def __init__(self, batchsize):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        self.ndims = 3
        self.targetval = 1.3932
        self.batchsize = batchsize

    def prob(self, x):
        integrand = torch.prod(torch.cos(x * np.pi), axis=-1)
        integrand = 1 / (1 - integrand)
        return integrand

    def log_prob(self, x):
        return torch.log(torch.abs(self.prob(x)))


class Gauss(nf.distributions.Target):
    def __init__(self, batchsize, ndims=2, alpha=0.2):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        self.ndims = ndims
        self.alpha = alpha
        self.log_const = -self.ndims * (np.log(self.alpha) + 0.5 * np.log(np.pi))
        self.targetval = erf(1 / (2.0 * self.alpha)) ** self.ndims
        self.batchsize = batchsize
        self.samples = batchsize
    def log_prob(self, x):
        return -1.0 * torch.sum((x - 0.5) ** 2 / self.alpha**2, -1) + self.log_const

    def prob(self, x):
        return torch.exp(self.log_prob(x))


class Camel(nf.distributions.Target):
    # Target value not implemented
    def __init__(self, batchsize, ndims=2, alpha=0.2, pos=1.0 / 8.0):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        self.ndims = ndims
        self.alpha = alpha
        self.pos = pos
        self.pre1 = np.exp(-self.ndims * (np.log(self.alpha) + 0.5 * np.log(np.pi)))
        self.pre2 = np.exp(-self.ndims * (np.log(self.alpha) + 0.5 * np.log(np.pi)))
        self.targetval = (
            0.5 * (0.5 * (erf(1 / (3.0 * alpha)) + erf(2 / (3.0 * alpha)))) ** ndims
            + 0.1
            / 16.0
            * (0.5 * (erf(1 / (3.0 * alpha / 4.0)) + erf(2 / (3.0 * alpha / 4.0))))
            ** ndims
        )
        self.batchsize = batchsize

    def log_prob(self, x):
        return torch.log(self.prob(x))

    def prob(self, x):
        exp1 = -1.0 * torch.sum((x - (self.pos)) ** 2 / self.alpha**2, -1)
        exp2 = -1.0 * torch.sum((x - (1.0 - self.pos)) ** 2 / self.alpha**2, -1)
        return 0.5 * (self.pre1 * torch.exp(exp1) + self.pre2 * torch.exp(exp2))


class Camel_v1(nf.distributions.Target):
    # Target value not implemented
    def __init__(self, batchsize, ndims=2, alpha=0.2, pos=1.0 / 8.0):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        self.ndims = ndims
        self.alpha = alpha
        self.pos = pos
        self.pre1 = np.exp(-self.ndims * (np.log(self.alpha) + 0.5 * np.log(np.pi)))
        self.pre2 = np.exp(-self.ndims * (np.log(self.alpha) + 0.5 * np.log(np.pi)))
        self.targetval = (
            0.5 * (0.5 * (erf(1 / (3.0 * alpha)) + erf(2 / (3.0 * alpha)))) ** ndims
            + 0.1
            / 16.0
            * (0.5 * (erf(1 / (3.0 * alpha / 4.0)) + erf(2 / (3.0 * alpha / 4.0))))
            ** ndims
        )
        self.batchsize = batchsize

    def log_prob(self, x):
        return torch.log(self.prob(x))

    def prob(self, x):
        exp1 = -1.0 * torch.sum((x - (self.pos)) ** 2 / self.alpha**2, -1)
        exp2 = -1.0 * torch.sum((x - (1.0 - self.pos)) ** 2 / self.alpha**2, -1)
        gx0 = (
            torch.exp(-((x[:, 0] - (self.pos)) ** 2) / self.alpha**2)
            + torch.exp(-((x[:, 0] - (1.0 - self.pos)) ** 2) / self.alpha**2)
            + 1e-4
        )
        gx1 = (
            torch.exp(-((x[:, 1] - (self.pos)) ** 2) / self.alpha**2)
            + torch.exp(-((x[:, 1] - (1.0 - self.pos)) ** 2) / self.alpha**2)
            + 1e-4
        )
        return (
            0.5
            * (self.pre1 * torch.exp(exp1) + self.pre2 * torch.exp(exp2))
            / (gx0 * gx1)
        )


class Sphere(nf.distributions.Target):
    def __init__(self, ndims=2):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        self.ndims = ndims
        self.targetval = np.pi ** ((ndims + 1) / 2.0) / (
            2 ** (ndims + 1) * gamma(((ndims + 1) / 2.0) + 1)
        )

    def log_prob(self, x):
        prob = torch.abs(self.prob(x))
        return torch.where(prob > 1e-16, torch.log(prob), torch.log(prob + 1e-16))

    def prob(self, x):
        integrand = torch.sum(torch.square(x), axis=-1)
        integrand = torch.sqrt(
            torch.maximum(1 - integrand, torch.zeros_like(integrand))
        )
        return integrand


class Polynomial(nf.distributions.Target):
    def __init__(self, batchsize, ndims=2):
        super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))
        self.ndims = ndims
        self.targetval = ndims / 6.0
        self.batchsize = batchsize

    def log_prob(self, x):
        return torch.log(self.prob(x))

    def prob(self, x):
        return -torch.sum(torch.square(x), axis=-1) + torch.sum(x, axis=-1)
