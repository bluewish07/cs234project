import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import config
  
from pg import PG

class MultiAgentPG(object):
  """
    Owns multiple individual PG models, one per agent
  """

  def __init__(self, env, configuration=config, logger=None):
    # directory for training outputs
    if not os.path.exists(config.output_path):
      os.makedirs(config.output_path)

    # store hyper-params
    self.config = config
    self.logger = logger
    if logger is None:
      self.logger = get_logger(config.log_path)

    self.env = env

    # create n PG objects for n agents
    temp = []
    for i in range(self.env.n):
      temp.append(PG(self.env, configuration=self.config, logger=logger))
    self.agents = temp
    
  def build(self):
    for idx, agent_net in enumerate(self.agents):
      var_scope = "agent_" + str(idx)
      with tf.variable_scope(var_scope):
        agent_net.build()

  def initialize(self):
    sess = tf.Session()
    for network in self.agents:
        network.initialize(session=sess)

  def sample_paths_n(self, num_episodes=None):
    """
          Sample paths for all agents for the environment.

          Args:
                num_episodes:   the number of episodes to be sampled
                  if none, sample one batch (size indicated by config file)
          Returns:
                paths_n: paths_n[i] is a list of sampled paths/trajectories for agent i.
                  n represents number of agents
                  Each path in paths is a dictionary with
                  path["observation"] a numpy array of ordered observations in the path
                  path["actions"] a numpy array of the corresponding actions in the path
                  path["reward"] a numpy array of the corresponding rewards in the path

    """
    episode = 0
    paths_n = []
    t = 0
        
    for i in range(self.env.n):
      paths_n.append([])

    while (num_episodes or t < self.config.batch_size):
      obs_n = self.env.reset() # list of n observations after initial setup
      observations_n, actions_n, rewards_n = [], [], [] # holds values for each agent

      for i in range(self.env.n):
        observations_n.append([])
        actions_n.append([])
        rewards_n.append([])
        
      for step in range(self.config.max_ep_len):         
        action = []  # actions taken by all agents at this timestep
        for i in range(self.env.n):
          observations_n[i].append(obs_n[i])
          
          act = self.agents[i].get_sampled_action(obs_n[i])
          # act = self.env.action_space[0].sample() # for testing without policy network
          
          action_onehot = np.zeros((self.env.action_space[0].n))
          action_onehot[act] = 1
          
          action.append(action_onehot)
          actions_n[i].append(act) # append the non-onehot version for training

        obs_n, rew_n, done_n, info_n = self.env.step(action)
        for i in range(self.env.n):
          rewards_n[i].append(rew_n[i])
        t += 1
        if (done_n[0] or step == self.config.max_ep_len - 1):
          break
        if (not num_episodes) and t == self.config.batch_size:
          break

      # form a path for each agent
      for i in range(self.env.n):
        path = {"observation": np.array(observations_n[i]),
                "reward": np.array(rewards_n[i]),
                "action": np.array(actions_n[i])}
        paths_n[i].append(path)

      episode += 1
      if num_episodes and episode >= num_episodes:
        break

    return paths_n

  def train(self):
    for t in range(self.config.num_batches):
      self.logger.info("Batch " + str(t) + ":")
      paths_n = self.sample_paths_n(num_episodes=self.config.batch_size_in_episodes)
      for i in range(self.env.n):
        self.logger.info("training for agent " + str(i) + "...")
        agent_net = self.agents[i]
        agent_net.train_for_batch_paths(paths_n[i])

    self.logger.info("- Training all done.")


  def run(self):
    """
    Apply procedures of training for a MultiAgentPG.
    """
    # build model
    self.build()
    # initialize
    self.initialize()

    # model
    self.train()