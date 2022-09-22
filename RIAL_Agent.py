from model import RIAL
from gym.spaces import Discrete
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import datetime


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
        self.action_model = RIAL(self.action_space, self.hidden_space)
        self.t_action_model = RIAL(self.action_space, self.hidden_space)
        self.message_model = RIAL(self.action_space, self.hidden_space)
        self.t_message_model = RIAL(self.action_space, self.hidden_space)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.train_action_metric = keras.metrics.MeanSquaredError()
        self.train_message_metric = keras.metrics.MeanSquaredError()
        self.eval_action_metric = keras.metrics.MeanSquaredError()
        self.eval_message_metric = keras.metrics.MeanSquaredError()


        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    def to_ext_numpy(self, feature):
        '''
        convert list to expanded numpy array for forward pass
        '''
        return np.expand_dims(np.asarray([feature]), axis=0)


    def choose_action(self, next_state, action, message, hidden, agent_ind):
        '''
        choose an action based on the epsilon-greedy policy
        '''
        if hidden == None:
            hidden = [self.action_model.rnn1.get_initial_state(batch_size=1, dtype=float).numpy(), self.action_model.rnn2.get_initial_state(batch_size=1, dtype=float).numpy()]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space), hidden
        hidden = np.asarray(hidden).swapaxes(0,1)
        input = tuple([np.asarray([next_state]), self.to_ext_numpy(action), self.to_ext_numpy(message), np.asarray([agent_ind])] + [hidden])
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
        input = tuple([np.asarray([next_state]), self.to_ext_numpy(action), self.to_ext_numpy(message), np.asarray([agent_ind])] + [hidden])
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

            # update action RIAL

            with tf.GradientTape() as tape:
                pred, _, _ = self.t_action_model((next_states, actions, messages, agent_inds, hidden_action))
                targets = rewards + self.gamma*(tf.math.reduce_max(pred, axis=1))*(1-dones.squeeze())
                q_vals, _, _ = self.action_model((states,last_actions, last_messages, agent_inds, last_hidden_action))
                q_inds = tf.one_hot(actions.squeeze(), self.action_space)
                exp_q = tf.reduce_sum(tf.multiply(q_vals, q_inds), axis=1)
                a_loss = self.mse_loss(targets, exp_q)

            gradients = tape.gradient(a_loss, self.action_model.trainable_variables)
            self.action_model.optimizer.apply_gradients(zip(gradients, self.action_model.trainable_variables))
            self.train_action_metric.update_state(targets, exp_q)
                
            with tf.GradientTape() as tape:   
                pred, _, _ = self.t_message_model((next_states, actions, messages, agent_inds, hidden_message))
                targets = rewards + self.gamma*(tf.math.reduce_max(pred, axis=1))*(1-dones.squeeze())
                q_vals, _, _ = self.message_model((states,last_actions, last_messages, agent_inds, last_hidden_message))
                q_inds = tf.one_hot(messages.squeeze(), self.action_space)
                exp_q = tf.reduce_sum(tf.multiply(q_vals, q_inds), axis=1)
                m_loss = self.mse_loss(targets, exp_q)

            gradients = tape.gradient(m_loss, self.message_model.trainable_variables)
            self.message_model.optimizer.apply_gradients(zip(gradients, self.message_model.trainable_variables))
            self.train_message_metric.update_state(targets, exp_q)

        return a_loss, m_loss

    
    def update_target_networks(self):
        self.t_action_model.set_weights(self.action_model.get_weights())
        self.t_message_model.set_weights(self.message_model.get_weights())

    def display_metrics(self):
        action_acc = self.train_action_metric.result()
        message_acc = self.train_message_metric.result()
        # Display metrics at the end of each epoch
        print("Training acc for actions over epoch: %.4f" % (float(action_acc),))
        print("Training acc for messages over epoch: %.4f" % (float(message_acc),))
        # Reset training metrics at the end of each epoch
        self.train_action_metric.reset_states()
        self.train_message_metric.reset_states()


        