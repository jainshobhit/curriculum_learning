# -*- coding: utf-8 -*-
'''
A simple implementation of Q-Learning for a 2X2 grid
'''
import pandas as pd
import random

class Qlearning:
    _qmatrix = None
    _learn_rate = None
    _discount_factor = None

    def __init__(self,
                 possible_states,
                 possible_actions,
                 initial_reward,
                 learning_rate,
                 discount_factor):
        """
        Initialise the q learning class with an initial matrix and the parameters for learning.
        :param possible_states: list of states the agent can be in
        :param possible_actions: list of actions the agent can perform
        :param initial_reward: the initial Q-values to be used in the matrix
        :param learning_rate: the learning rate used for Q-learning
        :param discount_factor: the discount factor used for Q-learning
        """
        # Initialize the matrix with Q-values
        init_data = [[float(initial_reward) for _ in possible_states]
                     for _ in possible_actions]
        self._qmatrix = pd.DataFrame(data=init_data,
                                     index=possible_actions,
                                     columns=possible_states)

        # Save the parameters
        self._learn_rate = learning_rate
        self._discount_factor = discount_factor

    def get_best_action(self, state):
        """
        Retrieve the action resulting in the highest Q-value for a given state.
        :param state: the state for which to determine the best action
        :return: the best action from the given state
        """
        # Return the action (index) with maximum Q-value
        return self._qmatrix[[state]].idxmax().iloc[0]

    def update_model(self, state, action, reward, next_state):
        """
        Update the Q-values for a given observation.
        :param state: The state the observation started in
        :param action: The action taken from that state
        :param reward: The reward retrieved from taking action from state
        :param next_state: The resulting next state of taking action from state
        """
        # Update q_value for a state-action pair Q(s,a):
        # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
        q_sa = self._qmatrix.ix[action, state]
        max_q_sa_next = self._qmatrix.ix[self.get_best_action(next_state), next_state]
        r = reward
        alpha = self._learn_rate
        gamma = self._discount_factor

        # Do the computation
        new_q_sa = q_sa + alpha * (r + gamma * max_q_sa_next - q_sa)
        self._qmatrix.set_value(action, state, new_q_sa)

class MDP:
    '''
    This represents a 2X2 gridworld problem in the form of an MDP
    '''
    def __init__(self, reward, discount_factor):
        self.states = [0, 1, 2, 3]
        self.actions = ['up', 'right', 'down', 'left']
        self.rewards = reward
        self.discount_factor = discount_factor

    def sample_state(self):
        return random.choice(self.states)

    def select_action(self, init_state):
        return random.choice(self.actions)

    def next_state(self, init_state, action):
        if init_state == 0:
            if action == 'up':
                return 2
            elif action == 'right':
                return 1
            elif action == 'down':
                return 0
            elif action == 'left':
                return 0
        if init_state == 1:
            if action == 'up':
                return 3
            elif action == 'right':
                return 1
            elif action == 'down':
                return 1
            elif action == 'left':
                return 0
        if init_state == 2:
            if action == 'up':
                return 2
            elif action == 'right':
                return 3
            elif action == 'down':
                return 0
            elif action == 'left':
                return 2
        if init_state == 3:
            if action == 'up':
                return 3
            elif action == 'right':
                return 3
            elif action == 'down':
                return 1
            elif action == 'left':
                return 2

    def get_reward(self, init_state, action):
        return self.rewards[init_state][self.actions.index(action)]


if __name__ == "__main__":
    rewards = [[-1, -1, -1, -1], 
                [5, -1, -1, -1],
                [-1, 5, -1, -1],
                [5, 5, -1 ,-1]]
    discount_factor = 0.5
    mdp = MDP(rewards, discount_factor)
    initial_reward = 0
    learning_rate = 0.9
    nepisodes = 400
    qlearn = Qlearning(mdp.states, mdp.actions, initial_reward,
            learning_rate, mdp.discount_factor)

    #Run the QLearning for nepisodes number of iterations. 
    #For the given values of discount factor and learning reward, the problem converges after 400 iterations
    for i in range(nepisodes):
        s = mdp.sample_state()
        a = mdp.select_action(s)
        next_s = mdp.next_state(s, a)
        r = mdp.get_reward(s, a)
        qlearn.update_model(s,a,r,next_s)
        print(qlearn._qmatrix)