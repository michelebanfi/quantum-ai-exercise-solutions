import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from ising_hands_on import Environment

def main():
    # Define the environment and the initial seed (for debugging)
    np.random.seed(0)
    env = Environment(n_spins=10)

    # Define the RL model (RL algorithm).
    # Here two RL-models are given. Choose one and check the performance.
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[32, 32])

    model = DQN(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs, learning_rate=0.0003, buffer_size=100000, learning_starts=5000, batch_size=32, tau=1.0,
                gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=10000, exploration_fraction=0.1, exploration_initial_eps=1.0,
                exploration_final_eps=0.05, max_grad_norm=10, verbose=1)

    model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1)

    # Train the agent for a certain number of time-steps
    model.learn(total_timesteps=80000)

    # Let's test our model by making our agent play some episodes
    n_test_episodes = 10
    cum_reward = 0
    for _ in range(n_test_episodes):
        # The episode begins:
        done = False
        observation = env.reset()
        # Agent - Environment interaction
        while not done:
            # Predict the "best" action according to the policy
            action, _states = model.predict(observation, deterministic=True)
            # Perform the action on the environment
            observation, reward, done, info = env.step(action.item())
            # Store the reward
            cum_reward += reward
    print("The mean return over {} test episodes is: ".format(n_test_episodes), cum_reward/n_test_episodes)



if __name__ == '__main__':
    main()