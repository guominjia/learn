import numpy as np
import matplotlib.pyplot as plt

class GradientBandit:
    """
    Gradient Bandit Algorithm implementation using softmax action selection
    """
    def __init__(self, n_actions, alpha=0.1, use_baseline=True):
        """
        Initialize the Gradient Bandit algorithm
        
        Parameters:
        - n_actions: number of actions
        - alpha: step size parameter
        - use_baseline: whether to use baseline (average reward)
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.use_baseline = use_baseline
        
        # Initialize preferences to zero
        self.H = np.zeros(n_actions)
        
        # Track average reward
        self.avg_reward = 0
        self.t = 0
        
    def softmax(self, preferences):
        """
        Compute softmax (Gibbs/Boltzmann) distribution
        """
        exp_H = np.exp(preferences - np.max(preferences))  # subtract max for numerical stability
        return exp_H / np.sum(exp_H)
    
    def select_action(self):
        """
        Select action according to softmax distribution
        """
        self.probs = self.softmax(self.H)
        action = np.random.choice(self.n_actions, p=self.probs)
        return action
    
    def update(self, action, reward):
        """
        Update preferences using gradient ascent
        """
        self.t += 1
        
        # Update average reward incrementally
        self.avg_reward += (reward - self.avg_reward) / self.t
        
        # Use baseline if enabled
        baseline = self.avg_reward if self.use_baseline else 0
        
        # Update preferences
        for a in range(self.n_actions):
            if a == action:
                # Increase preference for selected action if reward > baseline
                self.H[a] += self.alpha * (reward - baseline) * (1 - self.probs[a])
            else:
                # Decrease preference for non-selected actions
                self.H[a] -= self.alpha * (reward - baseline) * self.probs[a]


class TestBed:
    """
    k-armed bandit test bed
    """
    def __init__(self, n_actions=10, reward_mean=4.0):
        """
        Initialize test bed with true action values
        
        Parameters:
        - n_actions: number of actions
        - reward_mean: mean reward offset (baseline shift)
        """
        self.n_actions = n_actions
        self.reward_mean = reward_mean
        # True action values sampled from normal distribution with offset
        self.q_true = np.random.randn(n_actions) + reward_mean
        self.optimal_action = np.argmax(self.q_true)
        
    def get_reward(self, action):
        """
        Get reward for taking an action (sampled from normal distribution)
        """
        return np.random.randn() + self.q_true[action]


def run_experiment(n_runs=2000, n_steps=1000, n_actions=10, alphas=[0.1, 0.4], 
                   use_baseline=True, reward_mean=4.0):
    """
    Run bandit experiment with different step sizes
    
    Parameters:
    - reward_mean: offset for reward distribution (tests baseline importance)
    """
    results = {}
    
    for alpha in alphas:
        print(f"Running with alpha={alpha}, baseline={use_baseline}")
        avg_rewards = np.zeros(n_steps)
        optimal_actions = np.zeros(n_steps)
        
        for run in range(n_runs):
            if (run + 1) % 500 == 0:
                print(f"  Run {run + 1}/{n_runs}")
            
            # Create new bandit problem and agent
            bandit = TestBed(n_actions, reward_mean=reward_mean)
            agent = GradientBandit(n_actions, alpha=alpha, use_baseline=use_baseline)
            
            for step in range(n_steps):
                # Select action
                action = agent.select_action()
                
                # Get reward
                reward = bandit.get_reward(action)
                
                # Update agent
                agent.update(action, reward)
                
                # Track metrics
                avg_rewards[step] += reward
                if action == bandit.optimal_action:
                    optimal_actions[step] += 1
        
        # Average over runs
        avg_rewards /= n_runs
        optimal_actions /= n_runs
        
        results[alpha] = {
            'avg_rewards': avg_rewards,
            'optimal_actions': optimal_actions * 100  # Convert to percentage
        }
    
    return results


def plot_results(results_baseline, results_no_baseline):
    """
    Plot comparison of results
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot with baseline
    ax1.set_title('Gradient Bandit with Baseline', fontsize=14, fontweight='bold')
    for alpha, data in results_baseline.items():
        ax1.plot(data['optimal_actions'], label=f'α={alpha}', linewidth=2)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('% Optimal Action')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot without baseline
    ax2.set_title('Gradient Bandit without Baseline', fontsize=14, fontweight='bold')
    for alpha, data in results_no_baseline.items():
        ax2.plot(data['optimal_actions'], label=f'α={alpha}', linewidth=2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal Action')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_bandit_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Testing Gradient Bandit Algorithm")
    print("=" * 50)
    print("NOTE: Using reward offset of +4 to demonstrate baseline importance")
    print("=" * 50)
    
    # Test parameters
    n_runs = 2000
    n_steps = 1000
    n_actions = 10
    alphas = [0.1, 0.4]
    reward_mean = 4.0  # Important: non-zero mean to show baseline effect
    
    # Run with baseline
    print("\n### Running with Baseline ###")
    results_baseline = run_experiment(
        n_runs=n_runs, 
        n_steps=n_steps, 
        n_actions=n_actions, 
        alphas=alphas,
        use_baseline=True,
        reward_mean=reward_mean
    )
    
    # Run without baseline
    print("\n### Running without Baseline ###")
    results_no_baseline = run_experiment(
        n_runs=n_runs, 
        n_steps=n_steps, 
        n_actions=n_actions, 
        alphas=alphas,
        use_baseline=False,
        reward_mean=reward_mean
    )
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results_baseline, results_no_baseline)
    
    print("\nDone! Check 'gradient_bandit_results.png' for results.")
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("Final Performance (% Optimal Action at step 1000):")
    print("-" * 50)
    print("WITH Baseline:")
    for alpha, data in results_baseline.items():
        print(f"  α={alpha}: {data['optimal_actions'][-1]:.2f}%")
    print("\nWITHOUT Baseline:")
    for alpha, data in results_no_baseline.items():
        print(f"  α={alpha}: {data['optimal_actions'][-1]:.2f}%")
    print("\nDifference (WITH - WITHOUT):")
    for alpha in alphas:
        diff = results_baseline[alpha]['optimal_actions'][-1] - results_no_baseline[alpha]['optimal_actions'][-1]
        print(f"  α={alpha}: +{diff:.2f}%")