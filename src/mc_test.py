import numpy as np
import matplotlib.pyplot as plt


def target_function(x, y):
    """目标函数，例如 f(x, y) = x^2 + y^2"""
    return x**2 + y**2


def proposal_function(x, y, step_size):
    """生成新提议点"""
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, step_size)
    x_new = x + r * np.cos(theta)
    y_new = y + r * np.sin(theta)
    return x_new, y_new


def is_inside_circle(x, y):
    """检查点是否在单位圆内"""
    return x**2 + y**2 <= 1


def mcmc_integrate(num_samples, step_size):
    """MCMC积分计算单位圆的函数"""
    x_current, y_current = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
    while not is_inside_circle(x_current, y_current):
        x_current, y_current = np.random.uniform(-1, 1), np.random.uniform(-1, 1)

    accepted_samples = []

    for _ in range(num_samples):
        x_new, y_new = proposal_function(x_current, y_current, step_size)

        if is_inside_circle(x_new, y_new):
            # 计算接受概率
            acceptance_prob = min(
                1, target_function(x_new, y_new) / target_function(x_current, y_current)
            )
            if np.random.rand() < acceptance_prob:
                x_current, y_current = x_new, y_new

        accepted_samples.append((x_current, y_current))

    return np.array(accepted_samples)


# 参数设置
num_samples = 1000000
step_size = 0.1

# 运行MCMC积分
samples = mcmc_integrate(num_samples, step_size)

# 计算积分估计
integral_estimate = (
    np.mean([target_function(x, y) for x, y in samples]) * np.pi
)  # 圆面积
print("估计的积分值:", integral_estimate)

# 可视化接受的样本
plt.scatter(*zip(*samples), s=1, color="blue")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("MCMC Samples Inside Unit Circle")
plt.show()
