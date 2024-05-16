import sys
from functools import cached_property

class Environment:
    def __init__(self, dimension, max_step):
        self.l = dimension #rows of the grid
        self.h = dimension #columns of the grid
        self.state = 0 #starting point on the grid
        self.n = 0 #number of steps at the beginning
        self.max_step = max_step #maximum number of steps per episode

    ## dictionary of possible actions ##
    @cached_property
    def actions(self):
        zero_dict = {i : i for i in range(self.l + 1)}
        zero_dict.update({i : i - self.l for i in range(self.l, self.l * self.h)})
        one_dict = {i : i -1 for i in range(0, self.h*self.l)}
        one_dict.update({i : i for i in range(0, self.h*self.l, self.l)})
        two_dict = {i : i +1 for i in range(0, self.h*self.l)}
        two_dict.update({i : i for i in range(self.l-1, self.l*self.h, self.l)})
        three_dict = {i : i + self.l for i in range(self.l * self.h)}
        three_dict.update({i : i for i in range(self.l*(self.h-1), self.l*self.h)})
        return {0 : zero_dict,
                1 : one_dict,
                2 : two_dict,
                3 : three_dict}

    ## selector for one action ##
    def action(self, a):
        return self.actions[a]

    '''
    This function returns the reward evaluation
    by the actual state of the environment
    '''
    def reward(self):
        ## Sparse reward ##
        return 1 if self.state == self.l*self.h-1 else 0


    '''
    This function returns if the environment is solved
    i.e. if the actual state matches the target state
    '''
    def is_solved(self):
        return True if self.state == self.l*self.h-1 else False

    '''
    This function returns if the episode is ended
    i.e. if the actual state matches the target state
    OR if the number of max steps has been exceeded 
    '''
    def is_ended(self):
        return True if self.is_solved() or self.n>=self.max_step else False

    '''
    The step function takes as input an action
    then it modifies the state of the environment.
    It returns the Markovian chain of the process:
    (state at time t, action, reward, state at time t+1)
    '''
    def step(self, action):
        state = self.state
        self.state = self.actions[action][self.state]
        self.n+=1
        return (state, action, self.reward(), self.state, self.is_ended())


    '''
    The reset function set to the initial state the current state
    of the environment, therefore to 0 the counter of steps.
    '''
    def reset(self):
        self.state = 0
        self.n = 0

    '''
    The render function is optional.
    It returns a graphical rendering of the environment.
    '''
    def render(self):
        pass

