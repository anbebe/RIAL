from model import RIAL
from gym.spaces import Discrete
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random


class Agent():

    """ Agent for Reinforced Inter-Agent Learning (RIAL) """

    def __init__(self, state_space, action_space, args):
        '''
        Parameters:
            state_space:    [int]
                            shape of the state space (here observation of one agent)
            action_space:   int
                            number of possible actions for one agent
            args:           dict
                            arguments from parser for model and training
        '''
        self.state_space = state_space  
        self.action_space = action_space            
        self.message_space = Discrete(args.agents).n    # define the message space as integer numbers for representing the landmarks       
        self.epsilon = args.epsilon                 # initial value of epsilon for e-greedy policy
        self.gamma = args.discount_factor           # discount factor
        self.hidden_space = (100)                   # size of hidden state from rnn model

        # loads a pre-trained model or initialized new models for policies and targets
        if args.load_model:
            self.load_models(args.action_model_dir, args.message_model_dir)
        else:
            self.action_model = RIAL(self.action_space, self.hidden_space, lr=args.learning_rate, moment=args.momentum)
            self.t_action_model = RIAL(self.action_space, self.hidden_space, lr=args.learning_rate, moment=args.momentum)
            self.message_model = RIAL(self.action_space, self.hidden_space, lr=args.learning_rate, moment=args.momentum)
            self.t_message_model = RIAL(self.action_space, self.hidden_space, lr=args.learning_rate, moment=args.momentum)

        # define loss
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        # define metrics for training and evaluation for both models (action and message policy)
        self.train_action_metric = keras.metrics.MeanSquaredError()
        self.train_message_metric = keras.metrics.MeanSquaredError()
        self.eval_action_metric = keras.metrics.MeanSquaredError()
        self.eval_message_metric = keras.metrics.MeanSquaredError()

        # compile the policy and target models to build them
        self.action_model.compile(optimizer=self.action_model.optimizer, loss=self.mse_loss)
        self.t_action_model.compile(optimizer=self.t_action_model.optimizer, loss=self.mse_loss)
        self.message_model.compile(optimizer=self.message_model.optimizer, loss=self.mse_loss)
        self.t_message_model.compile(optimizer=self.t_message_model.optimizer, loss=self.mse_loss)


    def to_ext_numpy(self, feature):
        '''
        Convert list to expanded numpy array for forward pass
        Parameters:
            feature:    list of a feature
        Returns:
            numpy array of the feature with at least two dimensions
        '''
        return np.expand_dims(np.asarray([feature]), axis=0)


    def choose_action(self, state, action, message, hidden, agent_ind):
        '''
        Choose an action based on the epsilon-greedy policy.
        Parameters:
            state:      [int]
                        current state/observation 
            action:     int
                        index of the last action done by the previous agent
            message:    int
                        index of the message received, send by the previous agent
            hidden:     [int]
                        hidden states of the action model's rnn cells, from the last timestep
            agent_ind:  int
                        index of the current agent
        Returns:
            action chosen by the policy, current hidden states of the rnn
        '''
        # get initial hidden states if there are no states in history (for the first step in each episode)
        if hidden == None:
            hidden = [self.action_model.rnn1.get_initial_state(batch_size=1, dtype=float).numpy(), self.action_model.rnn2.get_initial_state(batch_size=1, dtype=float).numpy()]
        # epsilon greedy policy, to balance exploitation-exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space), hidden

        # preprocess data to the correct shapes, expand dimension, because the batch_size is simply one
        hidden = np.asarray(hidden).swapaxes(0,1)
        input = tuple([np.asarray([state]), self.to_ext_numpy(action), self.to_ext_numpy(message), np.asarray([agent_ind])] + [hidden])
        
        # choose action from policy
        act_values, hidden_1, hidden_2 = self.action_model(input, training=False)

        # use softmax to turn logit into probabilities
        return np.argmax(tf.nn.softmax(act_values[0])), [hidden_1.numpy(), hidden_2.numpy()]


    def choose_message(self, state, action, message, hidden, agent_ind):
        '''
        Choose a message based on the epsilon-greedy policy.
        Parameters:
            state:      [int]
                        current state/observation 
            action:     int
                        index of the last action done by the previous agent
            message:    int
                        index of the message received, send by the previous agent
            hidden:     [int]
                        hidden states of the message model's rnn cells, from the last timestep
            agent_ind:  int
                        index of the current agent
        Returns:
            message chosen by the policy, current hidden states of the rnn
        '''
        if hidden == None:
            hidden = [self.message_model.rnn1.get_initial_state(batch_size=1, dtype=float).numpy(), self.message_model.rnn2.get_initial_state(batch_size=1, dtype=float).numpy()]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.message_space), hidden
        hidden = np.asarray(hidden).swapaxes(0,1)
        input = tuple([np.asarray([state]), self.to_ext_numpy(action), self.to_ext_numpy(message), np.asarray([agent_ind])] + [hidden])
        m_values, hidden_1, hidden_2 = self.message_model(input, training=False)
        return np.argmax(tf.nn.softmax(m_values[0])), [hidden_1.numpy(), hidden_2.numpy()]

    def update(self, memory):
        '''
        One training step that updates the policy models by going backwards in the generated trajectories and
        updating after each step.
        Parameters:
            memory:     list containing a batch of sampled episodes (shape [batch, timesteps, features]) with
                        features saved as [state, action, message, reward, next_state, hidden_message, hidden_action, dones]
        Return:
            a_loss, m_loss      loss for each action and message model
        '''
        T = np.asarray(memory, dtype=object).shape[1]
        # count backwards from T to 1 steps per episode, T = 50
        for s in range(T)[:1:-1]:
            # preprocess data for the model
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

            # update action model
            # compute targets from target model and get the corresponding chosen actions from the policy model
            # to compute the loss
            with tf.GradientTape() as tape:
                pred, _, _ = self.t_action_model((next_states, actions, messages, agent_inds, hidden_action))
                targets = rewards + self.gamma*(tf.math.reduce_max(pred, axis=1))*(1-dones.squeeze())
                q_vals, _, _ = self.action_model((states,last_actions, last_messages, agent_inds, last_hidden_action))
                q_inds = tf.one_hot(actions.squeeze(), self.action_space)
                exp_q = tf.reduce_sum(tf.multiply(q_vals, q_inds), axis=1)
                a_loss = self.mse_loss(targets, exp_q)

            # get gradients from the loss, apply them via the rmsprop optimizer and update the metrics
            gradients = tape.gradient(a_loss, self.action_model.trainable_variables)
            self.action_model.optimizer.apply_gradients(zip(gradients, self.action_model.trainable_variables))
            self.train_action_metric.update_state(targets, exp_q)

            # update message model
            # compute targets from target model and get the corresponding chosen messages from the policy model
            # to compute the loss
            with tf.GradientTape() as tape:   
                pred, _, _ = self.t_message_model((next_states, actions, messages, agent_inds, hidden_message))
                targets = rewards + self.gamma*(tf.math.reduce_max(pred, axis=1))*(1-dones.squeeze())
                q_vals, _, _ = self.message_model((states,last_actions, last_messages, agent_inds, last_hidden_message))
                q_inds = tf.one_hot(messages.squeeze(), self.action_space)
                exp_q = tf.reduce_sum(tf.multiply(q_vals, q_inds), axis=1)
                m_loss = self.mse_loss(targets, exp_q)

            # get gradients from the loss, apply them via the rmsprop optimizer and update the metrics
            gradients = tape.gradient(m_loss, self.message_model.trainable_variables)
            self.message_model.optimizer.apply_gradients(zip(gradients, self.message_model.trainable_variables))
            self.train_message_metric.update_state(targets, exp_q)

        return a_loss, m_loss

    
    def update_target_networks(self):
        '''
        Update the target networks through applying the current weights from the policy models to them
        '''
        self.t_action_model.set_weights(self.action_model.get_weights())
        self.t_message_model.set_weights(self.message_model.get_weights())

    def display_metrics(self):
        '''
        Get the stored metrics during training from the action and message models, print them and reset them.
        '''
        action_acc = self.train_action_metric.result()
        message_acc = self.train_message_metric.result()
        # Display metrics at the end of each epoch
        print("Training acc for actions over epoch: %.4f" % (float(action_acc),))
        print("Training acc for messages over epoch: %.4f" % (float(message_acc),))
        # Reset training metrics at the end of each epoch
        self.train_action_metric.reset_states()
        self.train_message_metric.reset_states()

    def save_models(self, dir):
        '''
        Save the current action and message model in a given directory or else in the current.
        '''
        self.action_model.save(dir+"action", include_optimizer=True)
        self.message_model.save(dir+"message", include_optimizer=True)

    def load_models(self, a_dir, m_dir):
        '''
        Load the models from given directory.
        '''
        self.action_model = keras.models.load_model(a_dir)
        self.t_action_model = keras.models.load_model(a_dir)
        self.message_model = keras.models.load_model(m_dir)
        self.t_message_model = keras.models.load_model(m_dir)

        