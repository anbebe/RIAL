from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, EVENT_TYPES
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from human_aware_rl.rllib.utils import softmax, get_base_ae, get_required_arguments, iterable_equal

from maddpg_model import MADDPGModel, Critic

import gym
import random
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class ExperienceReplayBuffer(tf.Module):

    def __init__(self, max=1000000, batch_size=64):
      self.max = max
      self.b_size = batch_size
      self.memory = []

    def add_sample(self, state, action, reward, next_state, done):
      if len(self.memory) == self.max:
        self.memory = self.memory[1:]
      self.memory.append([state,action,reward, next_state, done])

    def sample_batch(self):
      return random.sample(population=self.memory, k=self.b_size)
    

        


class MADDPG():

    def __init__(self):

        self.num_episodes = 2 # 4000
        self.max_episode_len = 2 # 400
        self.sample_batch_size = 25
        self.train_batch_size = 1024
        self.update_params = 100 # network parameters are updated after this amount of samples added to buffer
        

        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }


        #cramped room layout includes 2 players
        mdp = OvercookedGridworld.from_layout_name(layout_name="cramped_room", rew_shaping_params=rew_shaping_params)
        self.env = OvercookedEnv.from_mdp(mdp, horizon=400)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.obs_space = self.setup_observation_space()

        #print(repr(self.env))
        #print(self.env.state)
        #print(self.env.state.players)

        # Init agents & buffer
        self.agents = AgentPair(MADDPGAgent(self.env, agent_index=0), MADDPGAgent(self.env, agent_index=1))
        self.buffer = ExperienceReplayBuffer()

        # Init centralized critic
        self.critic = Critic(self.action_space, self.obs_space)


    # function from rllib for ppo observation
    def setup_observation_space(self):
        dummy_state = self.env.mdp.get_standard_start_state()
        featurize_fn = lambda state: self.env.lossless_state_encoding_mdp(state)
        obs_shape = featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        
        return gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)



    def run(self):
        for _ in range(self.num_episodes):

            #TODO: initialize random process for action exploration or else some exploration factor (greedy)
            self.env.reset()
            print(repr(self.env))

            for t in range(self.max_episode_len):

                actions = self.agents.joint_action(self.env.state)
                actions_indx = np.asarray([np.argmax(actions[i][1]['action_probs']) for i in range(2)], dtype=float)
                actions_indx = np.expand_dims(actions_indx,axis=0)
                actions = (actions[0][0],actions[1][0])
                old_state = self.env.state
                next_state, reward, done, info = self.env.step(joint_action=actions)
                print(repr(self.env))
                self.buffer.add_sample(old_state, actions, reward, next_state, done)

                # for testing: use critic to compute value
                obs = self.env.lossless_state_encoding_mdp(old_state)
                obs = np.expand_dims(obs,axis=0)
                value = self.critic.forward((obs, actions_indx))
                print(value)


                for i in range(2):
                    if len(self.buffer.memory) < self.buffer.b_size:
                        break
                    else:
                        minibatch = self.buffer.sample_batch()
        print("finished")






if __name__ == '__main__':
    maddpg = MADDPG()
    maddpg.run()
    