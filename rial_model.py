import tensorflow as tf
from tf.keras.optimizers import Adam
import numpy as np
import random
import gym

import pettingzoo
from pettingzoo.butterfly import cooperative_pong_v4

env = cooperative_pong_v4.parallel_env(ball_speed=9, left_paddle_speed=12, right_paddle_speed=12, cake_paddle=True, max_cycles=900, bounce_randomness=False, max_reward=100, off_screen_penalty=-10)


class RIAL:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.05                # value of epsilon for e-greedy policy
        self.gamma = .99                  # discount factor
        self.lr = 0.0005                   # learning rate of the model
        self.momentum = 0.95               # mometum of RMSProp optimizer
        self.obs_model = self.build_obs_model()
        self.model = self.build_model()

    def build_obs_model(self):
        model = tf.keras.Sequential()

        # 3 conv layer and flatten as specific problem MLP
        model.add(tf.keras.layers.Conv2D(filters=25,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial"
            ))
        model.add(tf.keras.layers.Conv2D(filters=25,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.leaky_relu,
            name="conv_1"
        ))
        model.add(tf.keras.layers.Conv2D(filters=25,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.leaky_relu,
            name="conv_2"
        ))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.action_space, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        adam = Adam(lr=self.lr)
        model.compile(optimizer=adam, loss='mse')
        return model

    def build_model(self):

        # 2-layer RNN with GRUs that outputs internal state, approximates agent's action-observation history
        rnn = tf.keras.Sequential()
        rnn.add(tf.keras.layers.GRU(units=64))
        rnn.add(tf.keras.layers.GRU(units=128))

        # the output of the second layer used as input for 2-layer MLP that outputs Q-value
        rnn.add(tf.keras.layers.Dense(64, activation='relu'))
        rnn.add(tf.keras.layers.Dense(self.action_space, activation='relu'))

        opt = tf.keras.optimizers.RMSProp(learning_rate=self.lr, momentum=self.momentum, epsilon=self.epsilon)
        rnn.compile(optimizer=opt, loss='mse') #TODO:MSE loss?

        rnn.summary() 


        return 

    def choose_action(self, state):
        '''
        choose an action based on the epsilon-greedy policy
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.buffer.b_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_rial(model):

    loss = []
    episodes = 5000
    for e in range(episodes):
        state = env.reset()
        #TODO: state reshapen?
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = model.choose_action(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            #TODO: reshape score?
            state = next_state
            model.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss

if __name__ == "__main__":
    shared_model = RIAL(env.action_space.n, env.observation_space.shape[0])
    train_rial(shared_model)