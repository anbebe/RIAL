from cgi import test
from RIAL_Agent import Agent
from pettingzoo.mpe import simple_spread_v2
import numpy as np
import os
import imageio
import random
import time
from PIL import Image


def sample_trajectory(rial, render=False):
    env.reset()
    score = 0
    done = False
    hidden_message = None
    hidden_action = None
    last_message = 0
    last_action = 0
    batch_memory = []
    # env.agent_iter iterates over the two agents until both are finished or max_cycles is reached
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
        if render:
            env.render()
            time.sleep(0.1)
        next_state, reward, done, _ = env.last(observe=True)
        score += reward
        batch_memory.append([state.tolist(), action, message, reward, next_state.tolist(), hidden_message, hidden_action, done, agent_ind])
    return batch_memory, score



def train_rial(epochs, episode, obs_space, act_space, batch_size):
    loss = []
    rial = Agent(obs_space, act_space)
    for epoch in range(epochs):
        print("epoch ", epoch)
        start_time = time.time()
        for e in range(episode):
            # memory should have shape (batch_size, timesteps, episode_info)
            memory = []
            score = 0
            # sample batch size through finishing this amount of episodes
            for b in range(batch_size):
                b_memory, b_score = sample_trajectory(rial)
                score += b_score
                memory.append(b_memory)

            score = score/batch_size
            a_loss, m_loss = rial.update(memory, batch_size=batch_size)
            
            print("episode: {}/{}, score: {}".format(e, episode, score))
            loss.append(score)
            
            # update target networks after 100 episodes
            if (e) % 2 == 0 :
                rial.update_target_networks()
        rial.display_metrics()
        test_data, test_score = sample_trajectory(rial, render=True)
        print("epoch processed in {}".format(time.time() - start_time))

    return loss



if __name__ == "__main__":
    batch_size = 5 # normally 32 for testing 5
    env = simple_spread_v2.env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False)

    obs_space = env.observation_space('agent_0').shape
    act_space = env.action_space('agent_0').n
    # 5000 episodes in total - 20 epochs, 250 epsidoes
    episodes = 5 # 5000
    epochs = 2
    loss = train_rial(epochs, episodes, obs_space, act_space, batch_size=batch_size)
    env.close()

    

