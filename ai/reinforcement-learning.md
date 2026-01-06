# RL

## Commands
```bash
sudo apt update
sudo apt install swig cmake python3-opengl ffmpeg xvfb
pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt
curl https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt
pip install gymnasium==0.28.1 moviepy==1.0.3
pip install pyvirtualdisplay
huggingface-cli login
git config --global credential.helper store
xvfb-run -s "-screen 0 1400x900x24" <python-file>
```

## Pythons
```python
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```

```python
import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Create environment
env = gym.make('LunarLander-v2')

# We added some parameters to accelerate the training
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)
# Train the agent
model.learn(total_timesteps=int(2e5))

# Save the model
model_name = "ppo-LunarLander-v2"
model.save(model_name)

# Evaluate the model
eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```

## References
- https://huggingface.co/learn/deep-rl-course/unit0/introduction
- https://github.com/huggingface/huggingface_sb3
- https://stable-baselines3.readthedocs.io/
- https://gymnasium.farama.org/introduction/basic_usage/
- http://incompleteideas.net/book/
- http://incompleteideas.net/