import numpy as np
import torch
from torch import nn
import mpmath
from mpmath import polylog, gamma, findroot
from funcs_sigma import *
from funcs_sigmadk import *
import pandas as pd
import os
import re

num_loops = [2, 6, 15, 39, 111, 448]


def chemical_potential(beta, dim=3):
    def g(mu):
        return float(
            mpmath.re(polylog(dim / 2, -mpmath.exp(beta * mu)))
            + 1 / gamma(1 + dim / 2) * beta ** (dim / 2)
        )

    return float(findroot(g, 0))


def _StringtoIntVector(s):
    pattern = r"[-+]?\d+"
    return [int(match) for match in re.findall(pattern, s)]


def load_leaf_info(root_dir, name, key_str):
    df = pd.read_csv(os.path.join(root_dir, f"leafinfo_{name}_{key_str}.csv"))
    with torch.no_grad():
        leaftypes = torch.tensor(df.iloc[:, 1].to_numpy())
        leaforders = torch.tensor([_StringtoIntVector(x)[:3] for x in df.iloc[:, 2]]).T
        inTau_idx = torch.tensor(df.iloc[:, 3].to_numpy() - 1)
        outTau_idx = torch.tensor(df.iloc[:, 4].to_numpy() - 1)
        loop_idx = torch.tensor(df.iloc[:, 5].to_numpy() - 1)
        leafvalues = torch.tensor(df.iloc[:, 0].to_numpy())
    return (leaftypes, leaforders, inTau_idx, outTau_idx, loop_idx), leafvalues


class FeynmanIntegrand(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        order,
        rs,
        beta,
        loopBasis,
        leafstates,
        leafvalues,
        batchsize,
        is_real=True,
        has_dk=False,
    ):
        super().__init__()
        # super().__init__(prop_scale=torch.tensor(1.0), prop_shift=torch.tensor(0.0))

        print("rs:", rs, "beta:", beta, "order:", order, "batchsize:", batchsize)

        if is_real:
            self.is_real = True
        else:
            self.is_real = False

        # Unpack leafstates for clarity
        lftype, lforders, leaf_tau_i, leaf_tau_o, leafMomIdx = leafstates

        # Register buffers for leaf state information
        self.register_buffer("lftype", lftype)
        self.register_buffer("lforders", lforders)
        self.register_buffer("leaf_tau_i", leaf_tau_i)
        self.register_buffer("leaf_tau_o", leaf_tau_o)
        self.register_buffer("leafMomIdx", leafMomIdx)

        # Physical constants setup
        pi = np.pi
        self.register_buffer("eps0", torch.tensor(1 / (4 * pi)))
        self.register_buffer("e0", torch.sqrt(torch.tensor(2.0)))
        self.register_buffer("mass2", torch.tensor(0.5))
        self.register_buffer("me", torch.tensor(0.5))
        self.register_buffer("spin", torch.tensor(2.0))
        self.register_buffer("rs", torch.tensor(2.0))
        self.dim = 3

        # Derived constants
        self.register_buffer("kF", (9 * pi / (2 * self.spin)) ** (1 / 3) / self.rs)
        self.register_buffer("EF", self.kF**2 / (2 * self.me))
        self.register_buffer("beta", beta / self.EF)
        self.register_buffer("mu", chemical_potential(beta) * self.EF)
        self.register_buffer("maxK", 10 * self.kF)

        print(
            "param:",
            self.dim,
            self.beta,
            self.me,
            self.mass2,
            self.mu,
            self.e0,
            self.eps0,
        )

        self.batchsize = batchsize
        self.innerLoopNum = order
        self.totalTauNum = order
        self.ndims = self.innerLoopNum * self.dim + self.totalTauNum - 1

        self.register_buffer(
            "loopBasis", loopBasis
        )  # size=(self.innerLoopNum + 1, loopBasis.shape[1])
        self.register_buffer(
            "loops", torch.empty((self.batchsize, self.dim, loopBasis.shape[1]))
        )
        self.register_buffer(
            "leafvalues",
            torch.broadcast_to(leafvalues, (self.batchsize, leafvalues.shape[0])),
        )
        # print("lvalue,", self.leafvalues.shape)
        self.register_buffer(
            "p", torch.zeros([self.batchsize, self.dim, self.innerLoopNum + 1])
        )
        self.register_buffer("tau", torch.zeros_like(self.leafvalues))
        self.register_buffer("kq2", torch.zeros_like(self.leafvalues))
        self.register_buffer("invK", torch.zeros_like(self.leafvalues))
        self.register_buffer("dispersion", torch.zeros_like(self.leafvalues))
        self.register_buffer(
            "isfermi", torch.full_like(self.leafvalues, True, dtype=torch.bool)
        )
        self.register_buffer(
            "isbose", torch.full_like(self.leafvalues, True, dtype=torch.bool)
        )
        self.register_buffer("leaf_fermi", torch.zeros_like(self.leafvalues))
        self.register_buffer("leaf_bose", torch.zeros_like(self.leafvalues))
        self.register_buffer("factor", torch.ones([self.batchsize]))
        # self.register_buffer("root", torch.ones([self.batchsize, num_roots[order - 1]]))

        self.register_buffer("samples", torch.zeros([self.batchsize, self.ndims]))
        self.register_buffer("log_q", torch.zeros([self.batchsize]))
        self.register_buffer("log_det", torch.zeros([self.batchsize]))
        self.register_buffer("val", torch.zeros([self.batchsize]))

        # Convention of variables: first totalTauNum - 1 variables are tau. The rest are momentums in shperical coordinate.
        self.p[:, 0, 0] += self.kF
        self.extk = self.kF
        self.extn = 0
        # self.targetval = 4.0

        if order == 1:
            if has_dk:
                self.eval_graph = torch.jit.script(eval_graph1001)
            else:
                self.eval_graph = torch.jit.script(eval_graph100)
        elif order == 2:
            if has_dk:
                self.eval_graph = torch.jit.script(eval_graph2001)
            else:
                self.eval_graph = torch.jit.script(eval_graph200)
        elif order == 3:
            if has_dk:
                self.eval_graph = torch.jit.script(eval_graph3001)
            else:
                self.eval_graph = torch.jit.script(eval_graph300)
        elif order == 4:
            if has_dk:
                self.eval_graph = torch.jit.script(eval_graph4001)
            else:
                self.eval_graph = torch.jit.script(eval_graph400)
        elif order == 5:
            if has_dk:
                self.eval_graph = torch.jit.script(eval_graph5001)
            else:
                self.eval_graph = torch.jit.script(eval_graph500)
        else:
            raise ValueError("Invalid order")

    @torch.no_grad()
    def kernelFermiT(self):
        sign = torch.where(self.tau > 0, 1.0, -1.0)

        a = torch.where(
            self.tau > 0,
            torch.where(self.dispersion > 0, -self.tau, self.beta - self.tau),
            torch.where(self.dispersion > 0, -(self.beta + self.tau), -self.tau),
        )
        b = torch.where(self.dispersion > 0, -self.beta, self.beta)

        # Use torch operations to ensure calculations are done on GPU if tensors are on GPU
        self.leaf_fermi[:] = sign * torch.exp(self.dispersion * a)
        self.leaf_fermi /= 1 + torch.exp(self.dispersion * b)

    @torch.no_grad()
    def extract_mom(self, var):
        p_rescale = var[
            :, self.totalTauNum - 1 : self.totalTauNum - 1 + self.innerLoopNum
        ]
        theta = (
            var[
                :,
                self.totalTauNum - 1 + self.innerLoopNum : self.totalTauNum
                - 1
                + 2 * self.innerLoopNum,
            ]
            * np.pi
        )
        phi = (
            var[
                :,
                self.totalTauNum - 1 + 2 * self.innerLoopNum : self.totalTauNum
                - 1
                + 3 * self.innerLoopNum,
            ]
            * 2
            * np.pi
        )
        # print((p_rescale / (1+1e-6 - p_rescale)**2)**2 * torch.sin(theta))
        # self.factor = torch.prod((p_rescale / (1+1e-6 - p_rescale)**2)**2 * torch.sin(theta), dim = 1)
        # print("factor:", self.factor)
        # p_rescale /= (1.0 + 1e-10 - p_rescale)

        self.factor[:] = torch.prod(
            (p_rescale * self.maxK) ** 2 * torch.sin(theta), dim=1
        )
        self.p[:, 0, 1:] = p_rescale * self.maxK * torch.sin(theta)
        self.p[:, 1, 1:] = self.p[:, 0, 1:]
        self.p[:, 0, 1:] *= torch.cos(phi)
        self.p[:, 1, 1:] *= torch.sin(phi)
        self.p[:, 2, 1:] = p_rescale * self.maxK * torch.cos(theta)

    @torch.no_grad()
    def _evalleaf(self, var):
        self.isfermi[:] = self.lftype == 1
        self.isbose[:] = self.lftype == 2
        # update momentum
        self.extract_mom(var)  # varK should have shape [batchsize, dim, innerLoopMom]
        torch.matmul(self.p, self.loopBasis, out=self.loops)

        self.tau[:] = torch.where(
            self.leaf_tau_o == 0, 0.0, var[:, self.leaf_tau_o - 1]
        )
        self.tau -= torch.where(self.leaf_tau_i == 0, 0.0, var[:, self.leaf_tau_i - 1])
        self.tau *= self.beta

        kq = self.loops[:, :, self.leafMomIdx]
        self.kq2[:] = torch.sum(kq * kq, dim=1)
        self.dispersion[:] = self.kq2 / (2 * self.me) - self.mu
        self.kernelFermiT()

        # print("kq", kq.shape)
        ad_factor = kq[:, 0, :] * self.loopBasis[0, self.leafMomIdx]

        self.leaf_fermi *= ad_factor / self.me * (self.lforders[2] == 1) + torch.ones(
            self.batchsize, len(self.leafMomIdx), device=self.tau.device
        ) * (self.lforders[2] != 1)

        # Calculate bosonic leaves
        self.invK[:] = 1.0 / (self.kq2 + self.mass2)
        self.leaf_bose[:] = ((self.e0**2 / self.eps0) * self.invK) * (
            self.mass2 * self.invK
        ) ** self.lforders[1]
        self.leaf_bose *= ad_factor * self.invK * -2 * (self.lforders[1] + 1) * (
            self.lforders[2] == 1
        ) + torch.ones(self.batchsize, len(self.leafMomIdx), device=self.tau.device) * (
            self.lforders[2] != 1
        )
        # self.leafvalues[self.isfermi] = self.leaf_fermi[self.isfermi]
        # self.leafvalues[self.isbose] = self.leaf_bose[self.isbose]
        self.leafvalues = torch.where(self.isfermi, self.leaf_fermi, self.leafvalues)
        self.leafvalues = torch.where(self.isbose, self.leaf_bose, self.leafvalues)

    @torch.no_grad()
    def __call__(self, var, root):
        self._evalleaf(var)
        # root[:] = self.eval_graph(self.leafvalues)
        self.eval_graph(self.leafvalues, root)
        root *= (
            self.factor
            * (self.maxK * 2 * np.pi**2) ** (self.innerLoopNum)
            * (self.beta) ** (self.totalTauNum - 1)
            / (2 * np.pi) ** (self.dim * self.innerLoopNum)
        ).unsqueeze(1)

        # phase = torch.ones((self.batchsize, 1), device=root.device)
        if self.is_real:
            phase = torch.ones((self.batchsize, 1), device=root.device)
        else:
            phase = torch.zeros((self.batchsize, 1), device=root.device)

        if self.totalTauNum > 1:
            if self.is_real:
                phase = torch.hstack(
                    [
                        phase,
                        torch.cos(
                            (2 * self.extn + 1)
                            * np.pi
                            / self.beta
                            * var[:, : self.totalTauNum - 1]
                        ),
                    ]
                )
            else:
                phase = torch.hstack(
                    [
                        phase,
                        torch.sin(
                            (2 * self.extn + 1)
                            * np.pi
                            / self.beta
                            * var[:, : self.totalTauNum - 1]
                        ),
                    ]
                )

        # print(self.totalTauNum, root.shape, phase.shape)

        root *= phase

        return root.sum(dim=1)


def init_feynfunc(order, rs, beta, batch_size, is_real=True, has_dk=False):
    if has_dk:
        name = "sigmadk"
        root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigmadk/")
        f_loopbasis = f"loopBasis_{name}_maxOrder5.csv"
        partition = [(order, 0, 0, 1)]
    else:
        name = "sigma"
        root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
        f_loopbasis = f"loopBasis_{name}_maxOrder6.csv"
        partition = [(order, 0, 0)]

    df = pd.read_csv(os.path.join(root_dir, f_loopbasis))
    with torch.no_grad():
        loopBasis = torch.Tensor(
            df.iloc[: order + 1, : num_loops[order - 1]].to_numpy()
        )
    leafstates = []
    leafvalues = []

    for key in partition:
        key_str = "".join(map(str, key))
        state, values = load_leaf_info(root_dir, name, key_str)
        leafstates.append(state)
        leafvalues.append(values)

    feynfunc = FeynmanIntegrand(
        order,
        rs,
        beta,
        loopBasis,
        leafstates[0],
        leafvalues[0],
        batch_size,
        is_real=is_real,
        has_dk=has_dk,
    )

    return feynfunc
