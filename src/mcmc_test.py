import numpy as np


# 定义建议分布函数
def proposal_distribution(x, y, sigma=0.1):
    """
    建议分布函数，这里使用正态分布
    """
    angle = np.random.uniform(0, 2 * np.pi)
    rad = np.random.normal(0, sigma)
    new_x = x + rad * np.cos(angle)
    new_y = y + rad * np.sin(angle)
    return new_x, new_y


# 定义目标分布函数
def target_distribution(x, y):
    """
    目标分布函数，对于单位圆，任何点 (x, y) 都符合条件
    如果点在圆外，则概率为0
    """
    if x**2 + y**2 <= 1:
        return 1
    else:
        return 0


def metropolis_hastings(samples=10000, burn_in=1000):
    """
    Metropolis-Hastings算法实现
    """
    # 初始点
    x, y = np.random.rand(2)

    # 存储样本
    samples_x = []
    samples_y = []

    for i in range(samples):
        # 从建议分布中抽取新样本
        new_x, new_y = proposal_distribution(x, y)

        # 计算接受概率
        acceptance_prob = min(
            1, target_distribution(new_x, new_y) / target_distribution(x, y)
        )

        # 决定是否接受新样本
        if np.random.rand() < acceptance_prob:
            x, y = new_x, new_y

        # 存储样本
        if i >= burn_in:
            samples_x.append(x)
            samples_y.append(y)

    return np.array(samples_x), np.array(samples_y)


def estimate_circle_area(samples_x, samples_y, sample_area=4):
    """
    估计单位圆的面积
    """
    inside_circle = np.sum(samples_x**2 + samples_y**2 <= 1)
    area_estimate = (inside_circle / len(samples_x)) * sample_area
    return area_estimate


# 使用MCMC估计单位圆的面积
samples_x, samples_y = metropolis_hastings()
estimated_area = estimate_circle_area(samples_x, samples_y, sample_area=4)
print(f"Estimated area of unit circle: {estimated_area}")
