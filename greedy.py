import numpy as np

num_arms = 10
num_steps = 10000
num_samples = 2000
inits = (0., 5.)
epsilons = (0.1, 0.)

O_actions = [np.zeros(num_steps) for _ in epsilons]
R_averages = [np.zeros(num_steps) for _ in epsilons]

def choose(Q, epsilon):
    if np.random.random() < epsilon:
        idx = np.random.randint(0, num_arms)
    else:
        idx = np.argmax(Q)
    return idx

samples = np.random.normal(0,1, (num_samples, num_arms))
def e_greedy(init, epsilon, R_average, O_action, alpha=0.1):
    for sample in samples:
        Q = np.full(num_arms, init)
        N = np.zeros(num_arms)
        for step in range(num_steps):
            idx = choose(Q, epsilon)
            R = np.random.normal(sample[idx],1)
            for i in range(len(sample)): 
                sample[i] += np.random.normal(0, 0.01)
            N[idx] += 1
            Q[idx] = Q[idx] + 1/N[idx]*(R - Q[idx])

            R_average[step] += R
            O_action[step] += 1 if idx==np.argmax(sample) else 0

for init, epsilon, R_average, O_action in zip(inits, epsilons, R_averages, O_actions):
    e_greedy(init, epsilon, R_average, O_action)
    R_average /= num_samples
    O_action /= num_samples

import matplotlib.pyplot as plt

for init, epsilon, R_average in zip(inits, epsilons, R_averages):
    plt.plot(R_average, label=rf"$init={init}$, $\epsilon={epsilon}$")
plt.legend()