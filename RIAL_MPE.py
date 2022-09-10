import gym
import pettingzoo
from pettingzoo.mpe import simple_reference_v2
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, GRU, Embedding, Add, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.activations import relu, linear
import numpy as np
import PIL
from PIL import Image
import random

import datetime
import time

ACTIONS = ["no_action", "move_left", "move_right", "move_down", "move_up"]
MESSAGES = ["say_0", "say_1", "say_2", "say_3", "say_4", "say_5", "say_6", "say_7", "say_8", "say_9"]



class QNet(tf.keras.Model):
    def __init__(self, action_space, state_space):
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
        self.emb_ind = Embedding(input_dim=(None,2), output_dim= 128)

        self.add = Add(dynamic=True)
        self.concat = Concatenate(dynamic=True)

        # 2-layer RNN with GRUs that outputs internal state, approximates agent's action-observation history
        self.rnn = Sequential()
        self.rnn.add(GRU(units=64, time_major=True))
        self.rnn.add(GRU(units=128, time_major=True))

        # the output of the second layer used as input for 2-layer MLP that outputs Q-value
        self.q_net = Sequential()
        self.q_net.add(Dense(64, activation='relu')) 
        self.q_net.add(Dense(act_space, activation='relu'))


    @tf.function
    def call(self, input):
        '''
        input: (last_zs, observation, last_action, last_message agent_ind)
        should have shape [timesteps, batch_size, ..]
        instead: save z of previous steps to keep timesteps for rnn
        '''
        last_zs = input[0]
        state = input[1]
        last_act = input[2]
        last_m = input[3]
        agent = input[4]

        print("state: ", state)
        x = tf.cast(state, 'float')
        x = self.mlp(x)
        print("x: ", x)

        last_m = tf.cast(last_m, 'float')
        last_m = self.mlp2(last_m)

        last_act = tf.cast(last_act, 'float')
        last_act = self.emb_act(last_act)
        last_act = tf.squeeze(last_act, axis=1)
        print("last_act: ", last_act)

        agent = tf.cast(agent, 'float')
        agent = self.emb_ind(agent)
        agent = tf.squeeze(agent, axis=1)
        print("agent: ", agent)
        

        z = self.add([x, last_act, last_m, agent])
        print("z: ", z)
        z = self.concat(z, last_zs)
        rnn_output = self.rnn(z)
        q = self.q_net(rnn_output)
        return q, z


class Agent():

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.05                # initial value of epsilon for e-greedy policy
        self.gamma = 1                  # discount factor
        self.lr = 0.0005                   # learning rate of the model
        self.action_model = QNet(self.action_space, self.state_space)
        self.target_model = QNet(self.action_space, self.state_space)
        self.update_target_network()

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    def choose_action(self,last_zs, next_state, action, agent_ind):
        '''
        choose an action based on the epsilon-greedy policy, input: (observation, action, agent_ind)
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        action = np.expand_dims(action, axis=0)
        act_values = self.action_model.predict((last_zs, next_state, action, agent_ind))
        return np.argmax(act_values[0])

    def update(self, memory, agent_ind, batch_size=1):
        '''
        input to model: (observation, last_action, agent_ind)
        '''
        states = memory[:,:,0]
        actions = memory[:,:,1]
        rewards = memory[:,:,2]
        next_states = memory[:,:,3]
        last_actions = memory[:,:,4]
        dones = memory[:,:,5]
        pred = self.target_model.predict((next_states, actions, agent_ind))
        max_ind = np.argmax(pred)

        targets = rewards + self.gamma*(np.amax(pred, axis=1))*(1-dones)

        #with tf.GradientTape() as tape:
        q_vals = self.action_model.predict_on_batch((states,last_actions, agent_ind))
        error = targets - q_vals[0][max_ind]
        #loss = tf.keras.losses.MeanSquaredError()(targets, q_vals[0][max_ind])

        #TODO: now fits all values, but only the max of q and the target are necessary
        self.action_model.fit((states,last_actions, agent_ind), targets, batch_size=batch_size, validation_data=((states,last_actions, agent_ind), targets), callbacks=[self.tensorboard_callbacks])

        #gradients = tape.gradient(loss, self.action_model.trainable_variables)
        #self.action_model.optimizer.apply_gradients(zip(gradients, self.action_model.trainable_variables))

        return error

    
    def update_target_network(self):
        self.target_model.set_weights(self.action_model.get_weights())


def train_rial(episode, obs_space, act_space):
    loss = []
    rial = Agent(act_space, obs_space)
    for e in range(episode):
        print("episode: ", e)
        #TODO: reset agents?
        for env in envs:
          env.reset()
        time_memory = [] # to save the timsteps
        score = 0
        done = False
        render = False
        env_ind = 0
        env = envs[0]
        start = True
        # memory should have shape (timesteps, batch_size, episode_info)
        memory = []
        # env.agent_iter iterates over the two agents until both are finished
        # take only one action per agent, save in memory and update after every env is done
        for agent in env.agent_iter():
            if done:
                break
            
            agent_ind = np.asarray([[[int(agent[-1])]]])
            print("env: ", env_ind, "agent: ", agent_ind)
            

            # if the episode starts, there is no memory from last states:
            if start:
                state = env.last(observe=True)[0]
                state = state[np.newaxis, np.newaxis, :]
                last_action = np.array([[[]]])
            else:
                state = time_memory[:,env_ind,3]
                last_action = time_memory[:,env_ind,1]
                

            action = rial.choose_action(state,last_action, agent_ind)
            env.step(action)
            next_state, reward, done, info = env.last(observe=True)
            next_state = np.expand_dims(next_state, axis=0)
            score += reward
            memory.append([state, action, reward, next_state, last_action, done])
            if env_ind < len(envs) - 1:
                env_ind += 1
            else:
                time_memory.append(memory)
                error = rial.update(time_memory, agent_ind)
                env_ind = 0
                memory = []
                start = False
            env = envs[env_ind]
            
        print("episode: {}/{}, score: {}".format(e, episode, score/episode))
        loss.append(score)

        if e % 100 == 0:
            print("100 episodes reached")
            #TODO: update to timesteps saved
            rial.update_target_network()
            # show one episode in one environment for visualization of training
            frame_list = []
            envs[0].reset()
            for agent in envs[0].agent_iter():
              state_ = env.last()[0]
              state_ = np.expand_dims(state_, axis=0)
              action_ = rial.choose_action(state,action_, agent_ind)
              env.step(action_)
              envs[0].render(mode='rgb_array')
              time.sleep(0.05)


        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss

batch_size = 32
envs = [simple_reference_v2.env(local_ratio=0.5, max_cycles=25, continuous_actions=False) for _ in range(batch_size)]
print(envs[0].observation_space('agent_0').shape)
print(envs[0].action_space('agent_0').n)
obs_space = envs[0].observation_space('agent_0').shape
act_space = envs[0].action_space('agent_0').n
episodes = 200
loss = train_rial(episodes, obs_space, act_space)

        