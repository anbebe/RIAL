from RIAL_Agent import Agent
from pettingzoo.mpe import simple_spread_v2
import argparse
import time
import os
import pathlib

parser = argparse.ArgumentParser()


def sample_trajectory(env, rial, render=False):
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



def train_rial(env, epochs, episode, obs_space, act_space, batch_size, args):
    loss = []
    rial = Agent(obs_space, act_space, args)
    for epoch in range(epochs):
        print("epoch ", epoch)
        start_time = time.time()
        for e in range(episode):
            # memory should have shape (batch_size, timesteps, episode_info)
            memory = []
            score = 0
            # sample batch size through finishing this amount of episodes
            for b in range(batch_size):
                b_memory, b_score = sample_trajectory(env, rial)
                score += b_score
                memory.append(b_memory)

            score = score/batch_size
            a_loss, m_loss = rial.update(memory, batch_size=batch_size)
            
            print("episode: {}/{}, score: {}".format(e, episode, score))
            loss.append(score)
            
            # update target networks after 100 episodes
            if (e) % args.update_target_network == 0 :
                rial.update_target_networks()
        rial.display_metrics()
        test_data, test_score = sample_trajectory(env, rial, render=True)
        print("epoch processed in {}".format(time.time() - start_time))
    # save models at the end of training
    rial.save_models(args.log_dir)

    return loss

def main(args):

    #check if given directories for the models are valid
    if args.load_model:
        if not os.path.isdir(args.action_model_dir):
            a_path = str(os.path.dirname(os.path.abspath(__file__))) + str(args.action_model_dir)
            if not os.path.isdir(a_path):
                print("directory for action model does not exist")
                return
            else:
                args.action_model_dir = a_path
        if not os.path.isdir(args.message_model_dir):
            m_path = str(os.path.dirname(os.path.abspath(__file__))) + str(args.message_model_dir)
            if not os.path.isdir(m_path):
                print("directory for message model does not exist")
                return
            else:
                args.message_model_dir = m_path

    env = simple_spread_v2.env(N=args.agents, local_ratio=0.5, max_cycles=args.max_episode_length, continuous_actions=False)
    obs_space = env.observation_space('agent_0').shape
    act_space = env.action_space('agent_0').n

    loss = train_rial(env, args.epochs, args.episodes, obs_space, act_space, batch_size=args.batch_size, args=args)
    env.close()




if __name__ == "__main__":

    parser.add_argument('--mode', default="train", type=str, help="train or test")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--episodes', default=250, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--agents', default=2, type=int)
    parser.add_argument('--max_episode_length', default=25, type=int)
    parser.add_argument('--update_target_network', default=100 , type=int)

    parser.add_argument('--epsilon', default=0.05, type=float)
    parser.add_argument('--discount_factor', default=1.0, type=float)
    parser.add_argument('--learning_rate', default=0.0005 , type=int)
    parser.add_argument('--momentum', default=0.95 , type=int)

    parser.add_argument('--log_dir',  default="", type=str)
    parser.add_argument('--load_model',  default=False, type=bool)
    parser.add_argument('--action_model_dir', default="" , type=str)
    parser.add_argument('--message_model_dir', default="" , type=str)
    

    args = parser.parse_args()

    main(args)

    

