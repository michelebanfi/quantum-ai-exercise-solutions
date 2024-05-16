import numpy as np
from functools import cached_property
from prettytable import PrettyTable

class Agent:
    def __init__(self, dim, n_actions=4, learning_rate=.001, discount_factor=.99):
        self.n_actions = n_actions
        self.dim = dim**2
        self.lr = learning_rate
        self.df = discount_factor
        self.actions = np.arange(0,self.n_actions)
        self.Q_table = np.zeros((self.dim, self.n_actions))


    def update_Qfunc(self, elements):
        ## element is a tuple, containing:
        ## 0) state old
        ## 1) action
        ## 2) reward
        ## 3) state new
        ## TODO: Update the Q-function
        ## Q(s,a) = Q(s,a) + lr * (r + df * max_a(Q(s',a)) - Q(s,a))
        self.Q_table[elements[0], elements[1]] += self.lr * (elements[2] + self.df * max(self.Q_table[elements[3]]) - self.Q_table[elements[0], elements[1]])

    def Qtrain(self, batch):
        [self.update_Qfunc(info) for info in batch]


    def __repr__(self):
        table = PrettyTable(['', 'a=0', 'a=1', 'a=2', 'a=3'])
        [table.add_row([f"s={i}"] + list(row)) for i, row in enumerate(self.Q_table)]
        return table.get_string()


    def value_func(self):
        return np.array([max(row) for row in self.Q_table]).reshape(3,3)




if __name__=="__main__":
    A = Agent()
    print(A)
    print(A.value_func())



