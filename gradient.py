import numpy as np

num_samples = 2000
num_steps = 1000
num_arms = 10

def softmax(V):
    return np.exp(V) / np.sum(np.exp(V))

class GradientSelection:
    def __init__(self, num_arms=num_arms, alpha=0.1, Rave=0, baseline=1):
        self.alpha = alpha
        self.Q = np.zeros(num_arms)
        self.H = np.zeros(num_arms)
        self.P = softmax(self.H)
        self.Rave = Rave
        self.baseline=baseline
        self.N = 0

    def select_action(self):
        return np.argmax(self.P)

    def update(self, R, A):
        H_a = self.H[A]
        self.H = self.H - self.alpha*(R - self.baseline*self.Rave)*self.P
        self.H[A] = H_a + self.alpha*(R - self.baseline*self.Rave)*(1 - self.P[A])
        self.P = softmax(self.H)
        
        self.N += 1
        self.Rave = R/self.N + self.Rave * (self.N-1)/self.N

class Bandit:
    def __init__(self, means):
        self.means = means

    def pull(self, A):
        return np.random.normal(self.means[A], 1)

param_list = [
    r"$\alpha=0.1$ with baseline",
    r"$\alpha=0.1$ without baseline",
    r"$\alpha=0.4$ with baseline",
    r"$\alpha=0.4$ without baseline"
]
def get_grad(k, param_list=param_list):
    if k==param_list[0]:
        grad = GradientSelection(alpha=0.1, baseline=1)
    elif k==param_list[1]:
        grad = GradientSelection(alpha=0.1, baseline=0)
    elif k==param_list[2]:
        grad = GradientSelection(alpha=0.4, baseline=1)
    elif k==param_list[3]:
        grad = GradientSelection(alpha=0.4, baseline=0)
    else:
        raise KeyError("Error: Invalid k")
    return grad

Rewards = {k : np.zeros(num_steps) for k in param_list}
O_ratios = {k : np.zeros(num_steps) for k in param_list}
samples = np.random.normal(0,1, (num_samples, num_arms))
for k in param_list:
    for sample in samples:
        bandit = Bandit(sample)
        grad = get_grad(k)
        for step in range(num_steps):
            A = grad.select_action()
            R = bandit.pull(A)
            grad.update(R, A)

            Rewards[k][step] += R
            O_ratios[k][step] += 1 if A==np.argmax(sample) else 0