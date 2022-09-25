import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, GRUCell, RNN, Embedding, Add, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.activations import relu, linear



class RIAL(tf.keras.Model):
    def __init__(self, action_space, hidden_size, lr, moment):
        '''
        RNN that maintains an internal state h, an input network producing a task embedding z and
        an output network for the q-values
        '''

        super(RIAL, self).__init__()

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
        self.q_net.add(Dense(action_space, activation='relu'))

        self.optimizer = RMSprop(learning_rate=lr, momentum=moment)


    @tf.function
    def call(self, input):
        '''
        input: (observation, last_action, last_message, agent_ind)
        each should have shape [ batch_size, specific ]
        instead: hidden states from last timestep for both rnn cells
        '''
        #batch_size = input[0].shape[0]
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

        last_act = tf.reshape(last_act, [-1,128])
        agent =  tf.reshape(agent, [-1,128])
        
        z = self.add([x, last_act, last_m, agent])

        hidden_1, _  = self.rnn1(inputs=z, states = hidden[0])
        hidden_2,_ = self.rnn2(inputs=hidden_1, states = hidden[1])
        q = self.q_net(hidden_2)
        return q, hidden_1, hidden_2