import numpy as np

num_samples = 2000
num_steps = 1000
num_arms = 10

def softmax(V):
    return np.exp(V) / np.sum(np.exp(V))

class GradientSelection:
    def __init__(self, num_arms=num_arms, alpha=0.1, Rave=0):
        self.alpha = alpha
        self.Q = np.zeros(num_arms)
        self.H = np.zeros(num_arms)
        self.P = softmax(self.H)
        self.Rave = Rave
        self.N = 0

    def select_action(self):
        return np.argmax(self.P)

    def update(self, R, A):
        H_a = self.H[A]
        self.H = self.H - self.alpha*(R - self.Rave)*self.P
        self.H[A] = H_a + self.alpha*(R - self.Rave)*(1 - self.P[A])
        self.P = softmax(self.H)
        
        self.N += 1
        self.Rave = R/self.N + self.Rave * (self.N-1)/self.N

class Bandit:
    def __init__(self, means):
        self.means = means

    def pull(self, A):
        return np.random.normal(self.means[A], 1)

Rewards = np.zeros(num_steps)
O_ratios = np.zeros(num_steps)
samples = np.random.normal(0,1, (num_samples, num_arms))
for sample in samples:
    bandit = Bandit(sample)
    grad = GradientSelection()
    for step in range(num_steps):
        A = grad.select_action()
        R = bandit.pull(A)
        grad.update(R, A)

        Rewards[step] += R
        O_ratios[step] += 1 if A==np.argmax(sample) else 0
