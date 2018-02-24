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

    # create n PG object for n agents
    agent_networks = []
    for i in range(env.n):
      agent_networks.append(PG(env, logger=logger))

    self.agent_networks = agent_networks


  def build(self):
    for idx, agent_net in enumerate(self.agent_networks):
      var_scope = "agent_" + str(idx)
      with tf.variable_scope(var_scope):
        agent_net.build()

  def initialize(self):
    sess = tf.Session()
    for network in self.agent_networks:
        network.initialize(session=sess)

  def sample_paths_n(self, env, num_episodes=None):
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

    while (num_episodes or t < self.config.batch_size):
      obs_n = env.reset() # list of n observations after initial setup
      observations_n, actions_n, rewards_n = [], [], []

      for step in range(self.config.max_ep_len):
        act_n = []  # list of n actions for this step
        for i in env.n:
          obs = obs_n[i]
          observations_n[i].append(obs)
          act = self.agent_networks[i].get_sampled_action(obs)
          act_n.append(act)
          actions_n[i].append(act)

        obs_n, rew_n, done_n, info_n = env.step(act_n)
        for i in env.n:
          rewards_n[i].append(rew_n[i])
        t += 1
        if (done_n[0] or step == self.config.max_ep_len - 1):
          break
        if (not num_episodes) and t == self.config.batch_size:
          break

      # form a path for each agent
      for i in env.n:
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
      paths_n = self.sample_paths_n(self.env)
      for i in self.env.n:
        self.logger.info("training for agent " + str(i) + "...")
        agent_net = self.agent_networks[i]
        agent_net.train_for_batch(paths_n[i])

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
