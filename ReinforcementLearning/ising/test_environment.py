import numpy as np
from ising_hands_on import Environment

def main():
    # Setting the environment and the random seed
    np.random.seed(0)
    env = Environment(4)

    # Let's play a fake episode.
    # From the initial configuration, is the evolution as expected?
    env.reset()
    print("\nStarting state:")
    print('-'*40)
    env.render()
    print('-'*40)

    print("\nIteration 1:")
    print('-'*40)
    action = 0
    print("Action: ", action)
    state, reward, done, info = env.step(action)
    env.render()
    print("Reward: ", reward, "\nInfos: ", info)
    print('-'*40)

    print("\nIteration 2:")
    print('-'*40)
    action = 1
    print("Action: ", action)
    state, reward, done, info = env.step(action)
    env.render()
    print("Reward: ", reward, "\nInfos: ", info)
    print('-'*40)

    print("\nIteration 3:")
    print('-'*40)
    action = 2
    print("Action: ", action)
    state, reward, done, info = env.step(action)
    env.render()
    print("Reward: ", reward, "\nInfos: ", info)
    print('-'*40)

    print("\nIteration 4:")
    print('-'*40)
    action = 3
    print("Action: ", action)
    state, reward, done, info = env.step(action)
    env.render()
    print("Reward: ", reward, "\nInfos: ", info)
    print('-'*40)

    print("\nAfter reset:")
    print('-'*40)
    env.reset()
    env.render()
    print('-'*40)


if __name__ == '__main__':
    main()