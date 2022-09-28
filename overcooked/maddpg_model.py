from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
import tensorflow as tf
import numpy as np
import random
import gym

class DQNAgent(Agent):

    def __init__(self, env, agent_index):
        # - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        # self.env.lossless_state_encoding_mdp(state) has shape (5,4,26) for one agent, so obs[0] or obs[1]
        self.epsilon = 0.05
        self.env = env
        self.agent_index = agent_index
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.hidden_state = [] # TODO
        self.model = QNet()

    # e greedy policy
    def action(self, obs):
        """
        Arguments: 
            - the q-values for all actions
        returns: 
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """
        # Preprocess the environment state
        #obs = self.env.lossless_state_encoding_mdp(state) # obs[0] and obs[1] has shape (5,4,26)
        #my_obs = np.expand_dims(obs[self.agent_index],axis=0)
        #print("obs shape: ", my_obs.shape)

        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_space)
        else:
            act_values = self.model.predict(state)
            action_idx = np.argmax(act_values[0])

        agent_action =  Action.INDEX_TO_ACTION[action_idx]
        #print("agent_action: ", agent_action)
        
        # Softmax in numpy to convert logits to normalized probabilities
        #action_probs = softmax(action_outcome.numpy())
        #print("action_probs: ", action_probs)

        #action_info = {'action_probs' : action_probs}

        return agent_action#, action_info


class QNet(tf.keras.Model):
    def __init__(self, obs_space, act_space):
        '''
        RNN that maintains an internal state h, an input network producing a task embedding z and
        an output network for the q-values
        '''

        super(QNet, self).__init__()

        # 3 conv layer and flatten as specific probelm MLP, like in the proposed ppo algo
        self.l1 = tf.keras.layers.Conv2D(filters=25,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial"
            )
        self.l2 = tf.keras.layers.Conv2D(filters=25,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_1"
        )
        self.l3 = tf.keras.layers.Conv2D(filters=25,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_2"
        )

    def predict(self, obs, action):
        '''
        input: (obs_t, act_t-1, agent_ind)
        '''
        out = self.flatten(input[0])
        out = self.l1(out)
        out = tf.keras.layers.concatenate([out,input[1]])




class Actor(tf.keras.Model):
    '''
    Each agent has its own actor model, which takes its own observations as input and outputs action probabilities
    '''
    def __init__(self, action_space):
        super(Actor, self).__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.l1 = tf.keras.layers.Dense(64, activation='relu')
        self.l2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_out = tf.keras.layers.Dense(action_space)


    def forward(self,input):
        out = self.flatten(input)
        out = self.l1(out)
        out = self.l2(out)
        out = self.layer_out(out)
        return out

class Critic(tf.keras.Model):
    '''
    Centralized Critic takes as input state (in overcooked state = observation) and actions (next actions according to policy)
    '''
    def __init__(self, action_space, obs_space):
        super(Critic, self).__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.l1 = tf.keras.layers.Dense(64, activation='relu')
        self.l2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_out = tf.keras.layers.Dense(1)


    def forward(self,input):
        # input as tuple of (obs, actions)
        out = self.flatten(input[0])
        out = self.l1(out)
        out = tf.keras.layers.concatenate([out,input[1]])
        out = self.l2(out)
        out = self.layer_out(out)
        return out
        




        

