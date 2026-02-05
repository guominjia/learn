from typing import List, Tuple
from collections import defaultdict
import random

class BlackjackEnv:
    """Blackjack environment"""
    
    def __init__(self):
        self.action_space = ['hit', 'stick']
        
    def draw_card(self):
        """Draw a card from infinite deck (with replacement)"""
        card = random.randint(1, 13)
        if card > 10:
            card = 10  # Face cards count as 10
        return card
    
    def draw_hand(self):
        """Draw initial two cards"""
        return [self.draw_card(), self.draw_card()]
    
    def usable_ace(self, hand):
        """Check if hand has a usable ace (ace counted as 11 without busting)"""
        return 1 in hand and sum(hand) + 10 <= 21
    
    def sum_hand(self, hand):
        """Calculate hand sum, treating ace optimally"""
        total = sum(hand)
        if self.usable_ace(hand):
            total += 10  # Count ace as 11 instead of 1
        return total
    
    def is_bust(self, hand):
        """Check if hand is bust"""
        return self.sum_hand(hand) > 21
    
    def is_natural(self, hand):
        """Check if hand is a natural (21 with 2 cards)"""
        return sorted(hand) == [1, 10] and len(hand) == 2
    
    def reset(self):
        """Start a new game"""
        self.player_hand = self.draw_hand()
        self.dealer_hand = self.draw_hand()
        return self.get_state()
    
    def get_state(self):
        """Get current state: (player_sum, dealer_showing, usable_ace)"""
        player_sum = self.sum_hand(self.player_hand)
        dealer_showing = self.dealer_hand[0]
        usable_ace = self.usable_ace(self.player_hand)
        return (player_sum, dealer_showing, usable_ace)
    
    def step(self, action):
        """Execute action and return (state, reward, done)"""
        if action == 'hit':
            # Player hits
            self.player_hand.append(self.draw_card())
            if self.is_bust(self.player_hand):
                return self.get_state(), -1, True  # Player busts, loses
            else:
                return self.get_state(), 0, False  # Continue playing
        
        else:  # stick
            # Player sticks, dealer plays
            done = True
            
            # Dealer plays according to fixed strategy
            while self.sum_hand(self.dealer_hand) < 17:
                self.dealer_hand.append(self.draw_card())
            
            player_sum = self.sum_hand(self.player_hand)
            dealer_sum = self.sum_hand(self.dealer_hand)
            
            if self.is_bust(self.dealer_hand):
                reward = 1  # Dealer busts, player wins
            elif player_sum > dealer_sum:
                reward = 1  # Player sum closer to 21
            elif player_sum < dealer_sum:
                reward = -1  # Dealer sum closer to 21
            else:
                reward = 0  # Draw
            
            return self.get_state(), reward, done


class MonteCarlo:
    def __init__(self, status: List = None):
        """
        Initialize Monte Carlo learner
        status: list of all possible states (optional)
        """
        self.status = status or []
        self.V = defaultdict(float)  # State value function
        self.returns = defaultdict(list)  # Returns for each state
        
        # Initialize random values for given states
        for s in self.status:
            self.V[s] = random.random()
    
    def generate_episode(self, policy, env):
        """
        Generate an episode following the given policy
        Returns: list of (state, action, reward) tuples
        """
        episode = []
        state = env.reset()
        done = False
        
        # Handle natural blackjack
        if env.is_natural(env.player_hand):
            if env.is_natural(env.dealer_hand):
                return [(state, 'stick', 0)]  # Draw
            else:
                return [(state, 'stick', 1)]  # Player wins with natural
        
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def learn(self, policy, num_episodes=10000, first_visit=True):
        """
        Learn state values using Monte Carlo method
        
        Args:
            policy: function that takes state and returns action
            num_episodes: number of episodes to sample
            first_visit: if True, use first-visit MC; otherwise every-visit MC
        """
        env = BlackjackEnv()
        
        for episode_num in range(num_episodes):
            # Generate episode
            episode = self.generate_episode(policy, env)
            
            # Calculate returns and update values
            G = 0  # Return
            visited_states = set()
            
            # Process episode backwards to calculate returns
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + G  # No discounting (gamma = 1)
                
                # First-visit or every-visit MC
                if first_visit and state in visited_states:
                    continue
                
                visited_states.add(state)
                self.returns[state].append(G)
                self.V[state] = sum(self.returns[state]) / len(self.returns[state])
            
            # Print progress
            if (episode_num + 1) % 10000 == 0:
                print(f"Episode {episode_num + 1}/{num_episodes} completed")
        
        return self.V
    
    def print_values(self, min_visits=10):
        """Print learned state values"""
        print("\nLearned State Values (showing states with at least {} visits):".format(min_visits))
        print("State (player_sum, dealer_card, usable_ace) -> Value")
        print("-" * 60)
        
        # Sort states for better readability
        states_with_values = [(s, v, len(self.returns[s])) 
                              for s, v in self.V.items() 
                              if len(self.returns[s]) >= min_visits]
        states_with_values.sort()
        
        for state, value, visits in states_with_values[:20]:  # Show first 20
            print(f"{state} -> {value:.3f} (visits: {visits})")
        
        print(f"\n... (showing 20 of {len(states_with_values)} states)")


def simple_policy(state):
    """
    Simple policy: hit if player sum < 20, otherwise stick
    
    Args:
        state: (player_sum, dealer_showing, usable_ace)
    Returns:
        action: 'hit' or 'stick'
    """
    player_sum, dealer_showing, usable_ace = state
    
    if player_sum < 20:
        return 'hit'
    else:
        return 'stick'


def basic_strategy_policy(state):
    """
    Basic blackjack strategy policy
    
    Args:
        state: (player_sum, dealer_showing, usable_ace)
    Returns:
        action: 'hit' or 'stick'
    """
    player_sum, dealer_showing, usable_ace = state
    
    if usable_ace:
        # With usable ace (soft hand)
        if player_sum >= 19:
            return 'stick'
        else:
            return 'hit'
    else:
        # Without usable ace (hard hand)
        if player_sum >= 17:
            return 'stick'
        elif player_sum <= 11:
            return 'hit'
        else:  # 12-16
            if dealer_showing in [2, 3, 4, 5, 6]:
                return 'stick'  # Dealer likely to bust
            else:
                return 'hit'


if __name__ == "__main__":
    print("Blackjack Monte Carlo Learning")
    print("=" * 60)
    
    # Initialize all possible states
    states = []
    for player_sum in range(12, 22):  # 12-21
        for dealer_card in range(1, 11):  # A-10
            for usable_ace in [True, False]:
                states.append((player_sum, dealer_card, usable_ace))
    
    print(f"Total number of states: {len(states)}")
    
    # Create Monte Carlo learner
    mc = MonteCarlo(status=states)
    
    # Learn using basic strategy
    print("\nLearning state values using basic strategy...")
    mc.learn(policy=basic_strategy_policy, num_episodes=500000, first_visit=True)
    
    # Print some learned values
    mc.print_values(min_visits=100)
    
    # Analyze some specific states
    print("\n" + "=" * 60)
    print("Analysis of specific states:")
    print("-" * 60)
    
    interesting_states = [
        (20, 10, False),  # Strong hand vs dealer 10
        (12, 2, False),   # Weak hand vs dealer 2
        (16, 10, False),  # Difficult decision
        (18, 7, True),    # Soft 18 vs 7
    ]
    
    for state in interesting_states:
        if state in mc.V:
            visits = len(mc.returns[state])
            value = mc.V[state]
            print(f"State {state}: Value = {value:.3f}, Visits = {visits}")
    
    print("\n" + "=" * 60)
    print("Note: Positive values indicate favorable positions")
    print("      Negative values indicate unfavorable positions")
    print("      Values close to 0 indicate approximately even odds")