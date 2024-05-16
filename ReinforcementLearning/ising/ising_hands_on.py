import gym
import numpy as np

# ----------------------------------------------------------------------------------------------------------------
"""
The aim of this exercise is to write a custom environment for RL.
The goal is to find the ground state of a spin-chain by deciding the right sequence of spin flips.
The spins must assume -1,+1 as values.
The agent at each time-step must choose a spin to flip. 
The reward is a sparse one based on the energy of the system.
The end of the episode is determined by achieving the lowest energy or by reaching the maximum number of timesteps.
"""
# -----------------------------------------------------------------------------------------------------------------

class Environment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_spins, j=-1, h=1):
        super(Environment, self).__init__()

        # Discrete action space
        # insert the number of possible actions between the parenthesis
        self.action_space = gym.spaces.Discrete(n_spins)

        # The observation space gives to the agent the spin chain format (the actual state of the environment)
        # Specify the range for the spin values
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(n_spins, ), dtype=np.float32)

        # Set the state of the system (np.ndarray)
        self.n = n_spins
        # Set the initial random state of the environment (a chain of ± 1)

        # create a random state of the environment
        self.state = np.random.choice([-1, 1], size=(n_spins,))

        # Physics parameters
        self.J = np.random.choice([-1, 1], size=(n_spins,))
        self.h = h

        self.perfectState = None
        self.perfectStateEnergy = 0
        self.bruteForce()
        print("the initial state is: ", self.state)
        print("the interactions are: ", self.J)
        print("state founded with brute force:", self.perfectState)

        # Information parameters
        # set the initial parameters of the episode
        self.timestep = 0
        self.max_timestep = self.n - 1
        self.solved = False
        self.done = False
        self.info = {}


    def bruteForce(self):
        #brute force all the possible states. All combinations of +1 and -1
        #return the state with the lowest energy
        for i in range(2**self.n):
            state = np.array([1 if i & (1 << j) else -1 for j in range(self.n)])
            coupling_energy = sum([self.J[i] * state[i] * state[i + 1] for i in range(self.n - 1)]) / (self.n - 1)
            transverse_field = self.h * sum(state) / self.n
            ising_energy = coupling_energy + transverse_field
            print(state, ising_energy)
            if ising_energy < self.perfectStateEnergy:
                self.perfectState = state
                self.perfectStateEnergy = ising_energy


    def reward(self, state):
        coupling_energy = sum([self.J[i] * state[i]* state[i+1] for i in range(self.n-1)])/(self.n-1)
        transverse_field = self.h*sum(state)/self.n
        ising_energy = coupling_energy + transverse_field

        if abs(ising_energy - self.perfectStateEnergy) > 0:
            return 0
        else:
            return 1

    def step(self, action):
        """
        Here you perform an action and evolve the environment
        """
        # set the spin-flip action (decide which spin to flip)
        self.state[action] = -self.state[action]

        # get the reward and increase the timestep
        reward = self.reward(self.state)
        self.timestep += 1

        # establish whether the environment is solved
        self.solved = self.is_solved(reward)

        if self.timestep == self.max_timestep or self.solved:
            self.done = True

        self.info = {"Solved": self.solved}

        return self.state, reward, self.done, self.info

    def is_solved(self, reward):
        # decide whether the environment is solved based on the reward value and return a bool value (Solved=True)
        if reward == 1:
            return True
        return False

    def reset(self):
        """
        Here you reset the environment
        """
        # reset a random initial state
        self.state = np.random.choice([-1, 1], size=(self.n,))

        # reset the information parameters as in the class constructor
        self.timestep = 0
        self.solved = False
        self.done = False
        self.info = {}

        return self.state

    def render(self):
        """
        Returns a graphical representation of the environment. Not mandatory
        """
        print('Configuration: ', ''.join(['↑ ' if e == 1 else '↓ ' for e in self.state]))
