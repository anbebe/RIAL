from email import message
import gym
from gym.spaces import Box, Discrete
import pettingzoo
from pettingzoo.mpe import simple_spread_v2
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, GRUCell, RNN, Embedding, Add, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.activations import relu, linear
import numpy as np
import os
import imageio
import PIL
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt   
import random

import datetime
import time



class QNet(tf.keras.Model):
    def __init__(self, action_space, hidden_size):
        '''
        RNN that maintains an internal state h, an input network producing a task embedding z and
        an output network for the q-values
        '''

        super(QNet, self).__init__()

        # 2 dense layer as specific problem MLP: input space = 21, output dim = 128
        self.mlp = Sequential()
        self.mlp.add(Dense(64, activation='relu'))
        self.mlp.add(Dense(128, activation='relu'))

        # 1-layer MLP to process message
        self.mlp2 = Sequential()
        self.mlp2.add(Dense(128, activation='relu'))
        self.mlp2.add(BatchNormalization())

        # embedding for action
        self.emb_act = Embedding(input_dim=action_space, output_dim=128)

        # embedding for agent index
        self.emb_ind = Embedding(input_dim=2, output_dim= 128)

        self.add = Add(dynamic=True)
        self.concat = Concatenate(dynamic=True)

        # 2-layer RNN with GRUs that outputs internal state, approximates agent's action-observation history
        # work with GRU cell to input last hidden state
        self.hidden_size = hidden_size
        self.rnn1 = GRUCell(hidden_size)
        self.rnn2 = GRUCell(hidden_size)

        # the output of the second layer used as input for 2-layer MLP that outputs Q-value
        self.q_net = Sequential()
        self.q_net.add(Dense(64, activation='relu')) 
        self.q_net.add(Dense(act_space, activation='relu'))

        self.optimizer = RMSprop(learning_rate=0.0005, momentum=0.95, epsilon=0.05)


    @tf.function
    def call(self, input):
        '''
        input: (observation, last_action, last_message, agent_ind)
        each should have shape [ batch_size, specific ]
        instead: hidden states from last timestep for both rnn cells
        '''
        batch_size = input[0].shape[0]
        state = input[0]
        last_act = input[1]
        last_m = input[2]
        hidden = input[4]
        # assert that hidden has shape (batch_size, 2, 100)
        hidden = tf.transpose(hidden, [1,0,2])
        agent = input[3]      
        
        x = self.mlp(state)

        last_m = tf.cast(last_m, 'float')
        last_m = self.mlp2(last_m)

        last_act = tf.cast(last_act, 'float')
        last_act = self.emb_act(last_act)

        agent = tf.cast(agent, 'float')
        agent = self.emb_ind(agent)

        last_act = tf.reshape(last_act, [batch_size,128])
        agent =  tf.reshape(agent, [batch_size,128])
        
        z = self.add([x, last_act, last_m, agent])

        hidden_1, _  = self.rnn1(inputs=z, states = hidden[0])
        hidden_2,_ = self.rnn2(inputs=hidden_1, states = hidden[1])
        q = self.q_net(hidden_2)
        return q, hidden_1, hidden_2


class Agent():

    """ Implementation of deep q learning algorithm """

    def __init__(self, state_space, action_space):

        self.action_space = action_space
        self.message_space = Discrete(3).n
        self.state_space = state_space
        self.epsilon = 0.05                # initial value of epsilon for e-greedy policy
        self.gamma = 1                  # discount factor
        self.lr = 0.0005                   # learning rate of the model
        self.hidden_space = (100)           # size of hidden state from rnn model
        self.action_model = QNet(self.action_space, self.hidden_space)
        self.t_action_model = QNet(self.action_space, self.hidden_space)
        self.message_model = QNet(self.action_space, self.hidden_space)
        self.t_message_model = QNet(self.action_space, self.hidden_space)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    def choose_action(self, next_state, action, message, hidden, agent_ind):
        '''
        choose an action based on the epsilon-greedy policy
        '''
        if hidden == None:
            hidden = [self.action_model.rnn1.get_initial_state(batch_size=1, dtype=float).numpy(), self.action_model.rnn2.get_initial_state(batch_size=1, dtype=float).numpy()]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space), hidden
        hidden = np.asarray(hidden).swapaxes(0,1)
        input = tuple([np.asarray([next_state]), to_ext_numpy(action), to_ext_numpy(message), np.asarray([agent_ind])] + [hidden])
        act_values, hidden_1, hidden_2 = self.action_model(input, training=False)
        # use softmax to turn logit into probabilities
        return np.argmax(tf.nn.softmax(act_values[0])), [hidden_1.numpy(), hidden_2.numpy()]

    def choose_message(self, next_state, action, message, hidden, agent_ind):
        '''
        choose a message based on the epsilon-greedy policy
        '''
        if hidden == None:
            hidden = [self.message_model.rnn1.get_initial_state(batch_size=1, dtype=float).numpy(), self.message_model.rnn2.get_initial_state(batch_size=1, dtype=float).numpy()]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.message_space), hidden
        hidden = np.asarray(hidden).swapaxes(0,1)
        input = tuple([np.asarray([next_state]), to_ext_numpy(action), to_ext_numpy(message), np.asarray([agent_ind])] + [hidden])
        m_values, hidden_1, hidden_2 = self.message_model(input, training=False)
        return np.argmax(tf.nn.softmax(m_values[0])), [hidden_1.numpy(), hidden_2.numpy()]

    def update(self, memory, batch_size=5):
        '''
        input to model: (memory: batch_size , timesteps, [state, action, message, reward, next_state, hidden_message, hidden_action, dones)
        start batch from second step to have last action
        '''
        #memory = np.asarray(memory)
        T = np.asarray(memory).shape[1]
        # count backwards from T to 1 steps per episode, T = 50
        for s in range(T)[:1:-1]:
            states = np.asarray([b[s][0] for b in memory])
            actions = np.asarray([[b[s][1]] for b in memory])
            messages = np.asarray([[b[s][2]] for b in memory])
            rewards = np.asarray([b[s][3] for b in memory])
            next_states = np.asarray([b[s][4] for b in memory])
            hidden_message = np.asarray([b[s][5] for b in memory]).squeeze()
            hidden_action = np.asarray([b[s][6] for b in memory]).squeeze()
            dones = np.asarray([[b[s][7]] for b in memory])
            agent_inds = np.asarray([b[s][8] for b in memory])
            last_actions = np.asarray([[b[s-1][1]] for b in memory])
            last_messages = np.asarray([[b[s-1][2]] for b in memory])
            last_hidden_action = np.asarray([b[s-1][6] for b in memory]).squeeze()
            last_hidden_message = np.asarray([b[s-1][5] for b in memory]).squeeze()

            # update action QNet

            with tf.GradientTape() as tape:
                pred, _, _ = self.t_action_model((next_states, actions, messages, agent_inds, hidden_action))
                targets = rewards + self.gamma*(tf.math.reduce_max(pred, axis=1))*(1-dones.squeeze())
                q_vals, _, _ = self.action_model((states,last_actions, last_messages, agent_inds, last_hidden_action))
                q_inds = tf.one_hot(actions.squeeze(), self.action_space)
                exp_q = tf.reduce_sum(tf.multiply(q_vals, q_inds), axis=1)
                a_loss = self.mse_loss(targets, exp_q)

            gradients = tape.gradient(a_loss, self.action_model.trainable_variables)
            self.action_model.optimizer.apply_gradients(zip(gradients, self.action_model.trainable_variables))
                
            with tf.GradientTape() as tape:   
                pred, _, _ = self.t_message_model((next_states, actions, messages, agent_inds, hidden_message))
                targets = rewards + self.gamma*(tf.math.reduce_max(pred, axis=1))*(1-dones.squeeze())
                q_vals, _, _ = self.message_model((states,last_actions, last_messages, agent_inds, last_hidden_message))
                q_inds = tf.one_hot(messages.squeeze(), self.action_space)
                exp_q = tf.reduce_sum(tf.multiply(q_vals, q_inds), axis=1)
                m_loss = self.mse_loss(targets, exp_q)

            gradients = tape.gradient(m_loss, self.message_model.trainable_variables)
            self.message_model.optimizer.apply_gradients(zip(gradients, self.message_model.trainable_variables))

        return a_loss, m_loss

    
    def update_target_networks(self):
        self.t_action_model.set_weights(self.action_model.get_weights())
        self.t_message_model.set_weights(self.message_model.get_weights())

def to_ext_numpy(feature):
    '''
    convert list to expanded numpy array for forward pass
    '''
    return np.expand_dims(np.asarray([feature]), axis=0)

def render_one_episode(rial, episode):
    frames = []
    env.reset()
    test_memory = []
    done = False
    start = True
    last_message = 0
    last_action = 0
    score = 0
    for agent in env.agent_iter():
        if done:
            break
        agent_ind = [int(agent[-1])]
        state = env.last(observe=True)[0]
            
        # if the episode starts, there is no memory from last states:
        if start:
            message = rial.choose_first_message()
            action = rial.choose_first_action()
            hidden_message = None
            hidden_action = None
            start = False
        else:
            last_action = test_memory[-1][1]
            last_message = test_memory[-1][2]
            hidden_message = test_memory[-1][5]
            hidden_action = test_memory[-1][6]

            message, hidden_message = rial.choose_message(state,last_action, last_message, hidden_message, agent_ind)
            action, hidden_action = rial.choose_action(state,last_action, last_message, hidden_action, agent_ind)

        env.step(action)
        next_state, reward, done, _ = env.last(observe=True)
        score += reward
        #save each feature as list, not numpy array
        test_memory.append([state.tolist(), action, message, reward, next_state.tolist(), hidden_message, hidden_action, done, agent_ind])
        frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))
    env.close()
    print("test score: ", score)

    path_name = "random_agent_episode_" + str(episode) + ".gif"
    imageio.mimwrite(os.path.join('./videos/', path_name), frames)
    



def train_rial(episode, obs_space, act_space, batch_size):
    loss = []
    rial = Agent(obs_space, act_space)
    for e in range(episode):
        print("episode: ", e+1)
        # memory should have shape (batch_size, timesteps, episode_info)
        memory = []
        b_score = 0
        #TODO: reset agents?
        # sample batch size through finishing this amount of episodes
        for b in range(batch_size):
            env.reset()
            score = 0
            done = False
            hidden_message = None
            hidden_action = None
            last_message = 0
            last_action = 0
            batch_memory = []
            # env.agent_iter iterates over the two agents until both are finished or max_cycles is reached
            step = 1
            for agent in env.agent_iter():

                if done:
                    break
                agent_ind = [int(agent[-1])]
                state = env.last(observe=True)[0]

                # only get history if there is history
                if len(batch_memory) > 0:
                    last_action = batch_memory[-1][1]
                    last_message = batch_memory[-1][2]
                    hidden_message = batch_memory[-1][5]
                    hidden_action = batch_memory[-1][6]

                message, hidden_message = rial.choose_message(state,last_action, last_message, hidden_message, agent_ind)
                action, hidden_action = rial.choose_action(state,last_action, last_message, hidden_action, agent_ind)

                env.step(action)
                next_state, reward, done, _ = env.last(observe=True)
                score += reward
                #save each feature as list, not numpy array
                batch_memory.append([state.tolist(), action, message, reward, next_state.tolist(), hidden_message, hidden_action, done, agent_ind])
                step += 1
            b_score += score
            memory.append(batch_memory)

        b_score = b_score/batch_size

        print("----- begin update ------")
        a_loss, m_loss = rial.update(memory, batch_size=batch_size)
            
        print("episode: {}/{}, score: {}".format(e, episode, b_score))
        loss.append(b_score)

        if (e+1) % 2 == 0 :
            print("100 episodes reached")
            rial.update_target_networks()
            # TODO:show one episode in one environment for visualization of training
            #render_one_episode(rial, e)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss

batch_size = 5 # normally 32 for testing 5
env = simple_spread_v2.env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False)

obs_space = env.observation_space('agent_0').shape
act_space = env.action_space('agent_0').n
episodes = 10
loss = train_rial(episodes, obs_space, act_space, batch_size=batch_size)

        