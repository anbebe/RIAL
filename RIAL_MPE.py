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
import PIL
from PIL import Image
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


    @tf.function
    def call(self, input, training=False):
        '''
        input: (observation, last_action, last_message, agent_ind)
        each should have shape [ batch_size, specific ]
        instead: hidden states from last timestep for both rnn cells
        '''
        state = input[0]
        last_act = input[1]
        last_m = input[2]
        hidden = input[3]
        agent = input[4]        
        
        x = self.mlp(state)

        last_m = tf.cast(last_m, 'float')
        last_m = self.mlp2(last_m)
        #last_m = tf.reshape(last_m, [1,])

        last_act = tf.cast(last_act, 'float')
        last_act = self.emb_act(last_act)

        agent = tf.cast(agent, 'float')
        agent = self.emb_ind(agent)

        if training:
            # for testing batch_size=32, now 5
            last_act = tf.reshape(last_act, [5,128])
            agent =  tf.reshape(agent, [5,128])
        else:
            last_act = tf.reshape(last_act, [1,128])
            agent =  tf.reshape(agent, [1,128])
        
        z = self.add([x, last_act, last_m, agent])

        hidden_1, _  = self.rnn1(inputs=z, states = hidden[0])
        hidden_2,_ = self.rnn2(inputs=hidden_1, states = hidden[1])
        q = self.q_net(hidden_2)
        return q, hidden_1, hidden_2


class Agent():

    """ Implementation of deep q learning algorithm """

    def __init__(self, state_space, action_space):

        self.action_space = action_space
        #TODO: look at communication space
        #self.message_space = Box(low=-1.0, high=2.0, shape=(action_space.n), dtype=np.float32)
        self.message_space = Discrete(action_space).n
        self.state_space = state_space
        self.epsilon = 0.05                # initial value of epsilon for e-greedy policy
        self.gamma = 1                  # discount factor
        self.lr = 0.0005                   # learning rate of the model
        self.hidden_space = (100)           # size of hidden state from rnn model
        self.action_model = QNet(self.action_space, self.hidden_space)
        self.t_action_model = QNet(self.action_space, self.hidden_space)
        self.message_model = QNet(self.action_space, self.hidden_space)
        self.t_message_model = QNet(self.action_space, self.hidden_space)
        #self.update_target_networks()

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    def choose_action(self, next_state, action, message, hidden, agent_ind):
        '''
        choose an action based on the epsilon-greedy policy
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space), hidden
        if hidden == None:
            hidden = [self.message_model.rnn1.get_initial_state(batch_size=1, dtype=float).numpy(), self.message_model.rnn2.get_initial_state(batch_size=1, dtype=float).numpy()]
        input = (np.asarray(next_state), action, message, hidden, agent_ind)
        act_values, hidden_1, hidden_2 = self.action_model(input, training=False)
        # use softmax to turn logit into probabilities
        return np.argmax(tf.nn.softmax(act_values[0])), [hidden_1, hidden_2]

    def choose_first_action(self):
        '''
        choose an action based on the epsilon-greedy policy for the first step
        '''
        #TODO make it work with model
        '''
        last_action = np.asarray([0])
        last_message = np.asarray([0])
        hidden_action = np.asarray([[]])
        hidden_message = np.asarray([[]])
        action = np.expand_dims(action, axis=0)
        act_values, hidden_1, hidden_2 = self.action_model.predict((next_state, action, last_m, hidden, agent_ind))
        # use softmax to turn loit into probabilities
        return np.argmax(tf.nn.softmax(act_values[0])), [hidden_1, hidden_2]
        '''
        return random.randrange(self.action_space)

    def choose_message(self, next_state, action, message, hidden, agent_ind):
        '''
        choose a message based on the epsilon-greedy policy
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.message_space), hidden
        if hidden == None:
            hidden = [self.message_model.rnn1.get_initial_state(batch_size=1, dtype=float).numpy(), self.message_model.rnn2.get_initial_state(batch_size=1, dtype=float).numpy()]
        input = (np.asarray(next_state), action, message, hidden, agent_ind)
        m_values, hidden_1, hidden_2 = self.message_model(input, training=False)
        return np.argmax(tf.nn.softmax(m_values[0])), [hidden_1, hidden_2]

    def choose_first_message(self):
        '''
        choose a message based on the epsilon-greedy policy
        '''
        '''if np.random.rand() <= self.epsilon:
            
        #message = np.expand_dims(message, axis=0)
        m_values, hidden_1, hidden_2 = self.message_model.predict((next_state, action, message, hidden, agent_ind))
        return np.argmax(tf.nn.softmax(m_values[0])), [hidden_1, hidden_2]
        '''
        return random.randrange(self.message_space)

    def update(self, memory, batch_size=1):
        '''
        input to model: (memory: batch_size , timesteps, [state, action, message, reward, next_state, hidden_message, hidden_action, dones)
        start batch from second step to have last action
        '''
        memory = np.asarray(memory)
        # count backwards from T to 1 steps per episode
        for s in range(memory.shape[1])[:1:-1]:

            states = memory[:,s,0]
            actions = np.asarray(memory[:,s,1]).astype('float32')
            messages = np.asarray(memory[:,s,2]).astype('float32')
            rewards = memory[:,s,3]
            next_states = memory[:,s,4]
            hidden_message = memory[:,s,5]
            hidden_action = memory[:,s,6]
            dones = memory[:,s,7]
            agent_inds = np.asarray(memory[:,s,8]).astype('float32')
            last_actions = memory[:,s-1,1]
            last_messages = memory[:,s-1,2]
            last_hidden_action = memory[:,s-1,6]
            last_hidden_message = memory[:,s-1,5]

            # update action QNet

            with tf.GradientTape() as tape:
                pred = self.t_action_model.predict((next_states, actions, messages, hidden_action, agent_inds))
                pred = tf.nn.softmax(pred[0])
                max_ind = np.argmax(pred)

                targets = rewards + self.gamma*(np.amax(pred, axis=1))*(1-dones)
                q_vals = self.action_model.predict((states,last_actions, last_messages, last_hidden_action))
                #error = targets - q_vals[:,max_ind]
                loss = tf.keras.losses.MeanSquaredError()(targets, q_vals[:,max_ind])

            #self.action_model.train_on_batch((states,last_actions, ), targets, batch_size=batch_size, validation_data=((states,last_actions, agent_ind), targets), callbacks=[self.tensorboard_callbacks])
            gradients = tape.gradient(loss, self.action_model.trainable_variables)
            self.action_model.optimizer.apply_gradients(zip(gradients, self.action_model.trainable_variables))

            # TODO: update message network

        return loss

    
    def update_target_networks(self):
        self.t_action_model.set_weights(self.action_model.get_weights())
        self.t_message_model.set_weights(self.message_model.get_weights())


def train_rial(episode, obs_space, act_space, batch_size):
    loss = []
    rial = Agent(obs_space, act_space)
    for e in range(episode):
        print("episode: ", e)
        # memory should have shape (batch_size, timesteps, episode_info)
        memory = []
        #TODO: reset agents?
        # sample batch size through finishing this amount of episodes
        for b in range(batch_size):
            env.reset()
            score = 0
            done = False
            start = True
            last_message = None
            last_action = None
            batch_memory = []
            # env.agent_iter iterates over the two agents until both are finished or max_cycles is reached
            step = 1
            for agent in env.agent_iter():
                if done:
                    break
            
                agent_ind = np.asarray([int(agent[-1])])
                agent_ind = np.array([agent_ind])
                print("step ", step)

                state = env.last(observe=True)[0]
                state = state[np.newaxis, :]
            
                # if the episode starts, there is no memory from last states:
                if start:
                    message = rial.choose_first_message()
                    action = rial.choose_first_action()
                    hidden_message = None
                    hidden_action = None
                    start = False
                else:
                    last_action = batch_memory[-1][1]
                    last_message = batch_memory[-1][2]
                    hidden_message = batch_memory[-1][5]
                    hidden_action = batch_memory[-1][6]

                    last_action = np.expand_dims(np.array([last_action]), axis=0)
                    last_message = np.expand_dims(np.array([last_message]), axis=0)
                    

                    message, hidden_message = rial.choose_message(state,last_action, last_message, hidden_message, agent_ind)
                    action, hidden_action = rial.choose_action(state,last_action, last_message, hidden_action, agent_ind)

                #print("action: ", action)
                #print("message: ", message)

                env.step(action)
                next_state, reward, done, _ = env.last(observe=True)
                next_state = np.expand_dims(next_state, axis=0)
                reward = np.expand_dims(np.asarray([reward]), axis=0)
                done = np.expand_dims(np.asarray([done]), axis=0)
                action = np.expand_dims(np.asarray([action]), axis=0)
                message = np.expand_dims(np.asarray([message]), axis=0)
                score += reward
                batch_memory.append([state, action, message, reward, next_state, hidden_message, hidden_action, done, agent_ind])
                step += 1
            memory.append(batch_memory)

        # TODO
        print("----- begin update ------")
        error = rial.update(memory, batch_size=batch_size)
            
        print("episode: {}/{}, score: {}".format(e, episode, score/episode))
        loss.append(score)

        if e % 100 == 0:
            print("100 episodes reached")
            #TODO: update to training process - timesteps, message
            rial.update_target_network()
            # show one episode in one environment for visualization of training
            frame_list = []
            env.reset()
            for agent in env.agent_iter():
              state_ = env.last()[0]
              state_ = np.expand_dims(state_, axis=0)
              action_ = rial.choose_action(state,action_, agent_ind)
              env.step(action_)
              env.render(mode='rgb_array')
              time.sleep(0.05)


        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss

batch_size = 5 # normally 32 for testing 5
env = simple_spread_v2.env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False)

obs_space = env.observation_space('agent_0').shape
act_space = env.action_space('agent_0').n
episodes = 200
loss = train_rial(episodes, obs_space, act_space, batch_size=batch_size)

        