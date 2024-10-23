import vegas
import numpy as np
import gvar
import torch

dim = 4
nitn = 10
ninc = 1000


@vegas.lbatchintegrand
def f_batch(x):
    dx2 = 0.0
    for d in range(dim):
        dx2 += (x[:, d] - 0.5) ** 2
    return np.exp(-200 * dx2)
    # ans = np.empty((x.shape[0], 3), float)
    # dx2 = 0.0
    # for d in range(dim):
    #     dx2 += (x[:, d] - 0.5) ** 2
    # ans[:, 0] = np.exp(-200 * dx2)
    # ans[:, 1] = x[:, 0] * ans[:, 0]
    # ans[:, 2] = x[:, 0] ** 2 * ans[:, 0]
    # return ans


def smc(f, map, neval, dim):
    "integrates f(y) over dim-dimensional unit hypercube"
    y = np.random.uniform(0, 1, (neval, dim))
    jac = np.empty(y.shape[0], float)
    x = np.empty(y.shape, float)
    map.map(y, x, jac)
    fy = jac * f(x)
    return (np.average(fy), np.std(fy) / neval**0.5)


def mc(f, neval, dim):
    "integrates f(y) over dim-dimensional unit hypercube"
    y = np.random.uniform(0, 1, (neval, dim))
    fy = f(y)
    return (np.average(fy), np.std(fy) / neval**0.5)


m = vegas.AdaptiveMap(dim * [[0, 1]], ninc=ninc)
ny = 20000
# torch.manual_seed(0)
# y = torch.rand((ny, dim), dtype=torch.float64).numpy()
y = np.random.uniform(0.0, 1.0, (ny, dim))  # 1000 random y's

x = np.empty(y.shape, float)  # work space
jac = np.empty(y.shape[0], float)
f2 = np.empty(y.shape[0], float)

for itn in range(10):  # 5 iterations to adapt
    m.map(y, x, jac)  # compute x's and jac

    f2 = (jac * f_batch(x)) ** 2
    m.add_training_data(y, f2)  # adapt
    # if itn == 0:
    #     print(np.array(memoryview(m.sum_f)))
    #     print(np.array(memoryview(m.n_f)))
    m.adapt(alpha=0.5)


# with map
r = smc(f_batch, m, 50_000, dim)
print("   SMC + map:", f"{r[0]} +- {r[1]}")

# without map
r = mc(f_batch, 50_000, dim)
print("SMC (no map):", f"{r[0]} +- {r[1]}")


# vegas with adaptive stratified sampling
print("VEGAS using adaptive stratified sampling")
integ = vegas.Integrator(dim * [[0, 1]])
training = integ(f_batch, nitn=10, neval=20000)  # adapt grid

# final analysis
result = integ(f_batch, nitn=1, neval=50_000, adapt=False)
print(result)
result = integ(f_batch, nitn=5, neval=10_000, adapt=False)
print(result)
result = integ(f_batch, nitn=5, neval=10_000)
print(result)
# print("I[0] =", result[0], "  I[1] =", result[1], "  I[2] =", result[2])
# print("Q = %.2f\n" % result.Q)
# print("<x> =", result[1] / result[0])
# print(
#     "sigma_x**2 = <x**2> - <x>**2 =",
#     result[2] / result[0] - (result[1] / result[0]) ** 2,
# )
# print("\ncorrelation matrix:\n", gv.evalcorr(result))
