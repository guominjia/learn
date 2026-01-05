import numpy as np

class UCB1:
    def __init__(self, n_actions, c=2.0):
        self.n_actions = n_actions   # 动作数量
        self.c = c                   # 探索系数
        self.Q = np.zeros(n_actions) # 动作价值估计
        self.N = np.zeros(n_actions) # 各动作被选择的次数
        self.t = 1                   # 当前时间步
    
    def select_action(self):
        # 计算UCB值，处理未被选过的动作（N=0时的无穷大）
        ucb_values = np.where(
            self.N == 0,
            np.inf,
            self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        )
        action = np.argmax(ucb_values)  # 选择最大UCB对应的动作
        return action
    
    def update(self, action, reward):
        # 更新动作的选中次数和估计价值
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        self.t += 1  # 时间步递增

# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 定义一个10臂老虎机环境，各臂的真实奖励服从不同均值
    np.random.seed(0)
    k_arms = 10

    class Bandit:
        def __init__(self, means):
            self.means = means
        
        def pull(self, action):
            return np.random.normal(self.means[action], 1)  # 高斯奖励
    
    num_samples = 2000
    num_steps = 1000
    rewards = np.zeros(num_steps)
    samples = np.random.normal(0, 1, (num_samples, k_arms))
    for sample in samples:
        bandit = Bandit(sample)
        ucb = UCB1(n_actions=k_arms, c=2)
        for i in range(num_steps):
            a = ucb.select_action()
            r = bandit.pull(a)
            ucb.update(a, r)
            rewards[i] += r
    
    # 可视化累计奖励
    plt.plot(rewards/num_samples)
    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.title('UCB1 Performance')
    plt.show()