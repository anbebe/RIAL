from RIAL_Agent import Agent
from pettingzoo.mpe import simple_spread_v2
import numpy as np
import os
import imageio
from PIL import Image


def sample_trajectory(rial, episode, render=False):
    frames = []
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
        next_state, reward, done, _ = env.last(observe=True)
        score += reward
        batch_memory.append([state.tolist(), action, message, reward, next_state.tolist(), hidden_message, hidden_action, done, agent_ind])
        if render:
            frame = env.render(mode='rgb_array')
            frames.append(Image.fromarray(frame))
    if render:
        env.close()
        print("test score: ", score)
        path_name = "random_agent_episode_" + str(episode) + ".gif"
        imageio.mimwrite(os.path.join('./videos/', path_name), frames)

    return batch_memory, score
    



def train_rial(episode, obs_space, act_space, batch_size):
    loss = []
    rial = Agent(obs_space, act_space)
    for e in range(episode):
        print("episode: ", e+1)
        # memory should have shape (batch_size, timesteps, episode_info)
        memory = []
        score = 0
        #TODO: reset agents?
        # sample batch size through finishing this amount of episodes
        for b in range(batch_size):
            b_memory, b_score = sample_trajectory(rial, e)
            score += b_score
            memory.append(b_memory)

        score = score/batch_size

        print("----- begin update ------")
        a_loss, m_loss = rial.update(memory, batch_size=batch_size)
            
        print("episode: {}/{}, score: {}".format(e, episode, score))
        loss.append(score)

        if (e+1) % 2 == 0 :
            print("100 episodes reached")
            rial.update_target_networks()
            # TODO:show one episode in one environment for visualization of training
            # _, score sample_trajectory(rial, e, render=True)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss



if __name__ == "__main__":
    batch_size = 5 # normally 32 for testing 5
    env = simple_spread_v2.env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False)

    obs_space = env.observation_space('agent_0').shape
    act_space = env.action_space('agent_0').n
    episodes = 10
    loss = train_rial(episodes, obs_space, act_space, batch_size=batch_size)