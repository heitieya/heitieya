import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class NormalDistributionFitter:
    def __init__(self, data):
        self.data = data
        self.mu = None
        self.std = None
        self.ci_lower = None
        self.ci_upper = None

    def fit(self):
        # 对数据进行正态分布拟合
        self.mu, self.std = norm.fit(self.data)
        # 计算95%置信区间
        self.ci_lower = self.mu - 1.96 * self.std
        self.ci_upper = self.mu + 1.96 * self.std
        print(f"Fitted mean: {self.mu}")
        print(f"Fitted standard deviation: {self.std}")
        print(f"95% confidence interval: ({self.ci_lower}, {self.ci_upper})")

    def analyze_within_ci(self):
        # 统计在95%置信区间内的数据个数
        within_ci = np.sum((self.data >= self.ci_lower) & (self.data <= self.ci_upper))
        total_data_points = len(self.data)
        percentage_within_ci = within_ci / total_data_points * 100
        print(f"Number of data points within 95% confidence interval: {within_ci}")
        print(f"Percentage of data points within 95% confidence interval: {percentage_within_ci:.2f}%")

    def plot(self):
        # 绘制数据的直方图
        plt.hist(self.data, bins=50, density=True, alpha=0.6, color='g')

        # 使用拟合的参数绘制正态分布曲线
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, self.mu, self.std)
        plt.plot(x, p, 'k', linewidth=2)
        title = f"Fit results: mu = {self.mu:.2f},  std = {self.std:.2f}"
        plt.title(title)

        # 标记95%置信区间
        plt.axvline(self.ci_lower, color='r', linestyle='--', linewidth=2)
        plt.axvline(self.ci_upper, color='r', linestyle='--', linewidth=2)

        # 显示图表
        plt.show()


if __name__ == '__main__':
    # 示例数据
    np.random.seed(0)
    data = np.random.normal(loc=0, scale=1, size=5000)

    # 使用类进行拟合、分析和绘图
    fitter = NormalDistributionFitter(data)
    fitter.fit()
    fitter.analyze_within_ci()
    fitter.plot()
