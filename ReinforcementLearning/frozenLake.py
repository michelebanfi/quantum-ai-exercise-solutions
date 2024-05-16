import gym

# Define the environment
env = gym.make("CartPole-v1")

# We need to reset the environment before starting to interact with it
observation = env.reset()
done = False

# Continue to interact with the environment until the episode is done
while not done:
  # Sample a random action to perform
  action = env.action_space.sample()  # this is where you would insert your policy

  # Let's evolve the environment (apply the action)
  observation, reward, done, _, info = env.step(action)

  # render the env (ansi mode since we are on colab. "human" if we want to plot it)
  frame = env.render()

env.close()