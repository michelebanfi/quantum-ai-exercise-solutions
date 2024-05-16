#######################################################
## LIBRARY SYSTEM ##                                  #
import sys, os                                        #
import numpy as np                                    #
import random, pandas                                 #
from moving_average import plot_movingAverage as plot #
                                                      #
from file_manager import csv_writer                   #
from environment_hands_on import Environment          #
from agent_hands_on import Agent                               #
#######################################################

'''
EPISODE FUNCTION
The core of the Reinforcement Learning is the episode.
The episode is composed by many steps.
---------------------------------------
STEP:
Given a s_t state, an action a is taken
returning a s_(t+1) state.
---------------------------------------
The episode ends when some conditions are satisfied:
1) the target state is reached
2) the number of step
The action could be random
or dictated by the Q-function 
'''
def episode(environment, agent, eps_, decay_rate):
    batch_ = []
    ## EPISODE ##
    while not environment.is_ended():
        ## STEP ##
        '''
        select a random number r
        if r < rate of random actions:
        -) take a random action
        -) otherwise the agent takes the best action
        '''
        if np.random.rand() < eps_:
            a = random.randrange(0, 4)
        else:
            ## best action = max value from the Q-table
            ## with state (row) fixed
            a = np.argmax(agent.Q_table[environment.state])
        info = environment.step(a)
        eps_ *= decay_rate
        batch_.append(info)
        ## END OF STEP ##
    ## END OF EPISODE ##

    ## Train the agent with the batch of info=(s_t, a, r, s_(t+1))
    ## returned by the module environment.step
    agent.Qtrain(batch=batch_)

    ## Useful flags for training:
    ## 1) the episode has been solved?
    ## 2) how many steps did it take?
    ## 3) update the rate of random action
    return environment.is_solved(), environment.n, eps_



def main():
    '''
    REINFORCEMENT LEARNING PARAMETERS
    1) eps:    the rate for the agent to take a random action. It decreases during the training.
               At the beginning, it is mandatory to have randomness in order to explore the environment.
    2) e:      exploration decay. It sets how much the randomness of an action should decrease by each step.
    3) lr:     the learning rate for the update of the Q-function.
    4) df:     the discount factor to evaluate the cumulative reward function. See the update for the Q-function.
    5) n_ep:   number of episodes to train the agent.
    6) n_step: number of steps n per episode. The episode ends when n=n_steps or the goal state is reached.
    7) dim:    linear size of the grid. Set it to 3 or 10
    '''
    eps = 1
    e = 0.9999
    lr =  0.01
    df = 0.99
    n_ep = 5000
    n_step = 100
    dim = 4

    ##################
    ## DATA PATTERN ##
    ##################

    '''
    Create a pattern to store the data
    in order to monitor the training
    '''
    pattern = f"data_{dim}/"
    if not os.path.exists(pattern):
        os.makedirs(pattern)
    ## 1) file tracking victories ##
    file_episodes = pattern + 'episode_data.csv'
    ## 2) parameters ##
    params_sim = pattern + 'params_simulation.csv'
    ## 3) data pattern ##
    graph_pattern = pattern + "Graphs/"

    #################
    ## RL SETTINGS ##
    #################
    '''
    Install the environment and the agent
    Environment -> grid dimension, max number of steps per episode
    Agent -> grid dimension, learning rate, discount factor
    '''
    ## 1) Environment ##
    E = Environment(dimension=dim, max_step=n_step)
    ## 2) Agent ##
    A = Agent(dim=dim, learning_rate=lr, discount_factor=df)


    ## Save in a csv file the parameters of the RL simulation ##
    with csv_writer(params_sim, 'w') as f:
        f.writerow(['learning_rate', 'discount_factor', 'n_episodes', 'steps_x_episode'])
        f.writerow([lr, df, n_ep, n_step])
    with csv_writer(file_episodes, 'w') as f:
        f.writerow(["victories", "length"])

    ##############
    ## TRAINING ##
    ##############
    for step in range(1, n_ep):
        ## run a single episode ##
        ## track if it's solved ##
        ## its length           ##
        ## update the eps value ##
        *episode_info, eps = episode(environment=E, agent=A, eps_=eps, decay_rate=e)

        ## At the end of the episode: ##
        ## reset the environment      ##
        E.reset()
        ## Write in an external file: ##
        ## 1) length of episode       ##
        ## 2) victory of episode      ##
        with csv_writer(file_episodes, 'a') as f:
            f.writerow(episode_info)
        #print(f"Victory: {bool(episode_info[0])}")


    '''
    Save the Q table in a .npy format
    '''
    np.save(pattern+'Q', A.Q_table)

    '''
    Plot the trend for the length of the episodes
    '''
    data = pandas.read_csv(file_episodes)
    length = pandas.DataFrame(data, columns=['length'])
    plot(length, 'length', 1000, graph_pattern)

    '''
    Plot the trend of victories
    '''
    data = pandas.read_csv(file_episodes)
    victories = pandas.DataFrame(data, columns=['victories'])
    plot(100*victories, 'victories', 1000, graph_pattern)




if __name__=='__main__':
    main()
