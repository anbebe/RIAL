from RIAL_Agent import Agent
from pettingzoo.mpe import simple_spread_v2
import argparse
import time
import os

parser = argparse.ArgumentParser()


def sample_trajectory(env, rial, test=False, render=False):
    '''
    Samples one episode for the given environment using the model from the agent as policy with
    the agents taking actions one after one.
    In the test case load the same scenario, else a random one is used. In the render case,
    each step of one agent is rendered and shown in a pygame window.
    Parameters:

        env:        the already existing environment
        rial:       RIAL_Agent
                    the common agent class, that contains the shared models for each agent
        test:       bool
                    indicates the test case
        render:     bool
                    indicates the render case

    Returns:
        memory containing states, actions, messages, rewards, next_states, hidden states for message rnn,
        hidden states for message action, dones, indices of the agents for all timesteps in the episode
    '''
    # for test case: use the same scenario, else random one
    if test:
        env.reset(seed=100)
    else:
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

        # use the model as policy for choosing action and message and save hidden states of the rnn cells
        message, hidden_message = rial.choose_message(state,last_action, last_message, hidden_message, agent_ind)
        action, hidden_action = rial.choose_action(state,last_action, last_message, hidden_action, agent_ind)
        env.step(action)

        # render the actions taken by the agents for visualization
        if render:
            env.render()
            time.sleep(0.1)
        next_state, reward, done, _ = env.last(observe=True)
        score += reward
        # save the data as list, for further processing in the update
        batch_memory.append([state.tolist(), action, message, reward, next_state.tolist(), hidden_message, hidden_action, done, agent_ind])
    return batch_memory, score



def train_rial(env, epochs, episode, obs_space, act_space, batch_size, args):
    '''
    Creates the Agent for training and trains the model for given epochs, episodes and batch_size.
    To generate a batch of samples for the update, the model samples whole trajectories due to the
    Pettingzoo API of agent.iter(), which stops when an episode is done.
    The target networks are updated after given time and the model is evaluated after each epoch on
    the same scenario for better visualization of training process.

    Parameters:

        env:        the already existing environment
        epochs:     int
                    amount of epochs to train, consisting of number of episodes
        episodes:   int
                    amount of episodes in each epoch, where a batch of samples is
                    generated and used for an update of the RIAL models
        obs-space:  [int]
                    shape of the observation space from the environment
        obs-space:  int
                    number of possible actions in the environment (possible, because the action_space is discrete)
        batch_size: number of samples for one batch
        args:       given arguments, including arguments for the model and agent

    Returns:

        loss over the whole training, saved in a list

    '''
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
            a_loss, m_loss = rial.update(memory)
            
            print("episode: {}/{}, score: {}".format(e, episode, score))
            loss.append(score)
            
            # update target networks after 100 episodes
            if (e) % args.update_target_network == 0 :
                rial.update_target_networks()

        # evaluate model through metrics and visualization of the steps taken in test scenario
        rial.display_metrics()
        test_data, test_score = sample_trajectory(env, rial, test=True, render=True)
        print("epoch processed in {}".format(time.time() - start_time))
    # save models at the end of training
    rial.save_models(args.log_dir)

    return loss

def test_rial(env, obs_space, act_space, args):
    '''
    Test a given or if not given, random model for one epsiode, that's always the same, and return the score.
    '''
    rial = Agent(obs_space, act_space, args)
    _, score = sample_trajectory(env, rial, test=True, render=True)
    print("Test score: ", score)

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

    if args.mode == "train":
        loss = train_rial(env, args.epochs, args.episodes, obs_space, act_space, batch_size=args.batch_size, args=args)
    elif args.mode == "test":
        test_rial(env, obs_space, act_space, args)
    else:
        print("No valid mode. Try train or test")
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

    

