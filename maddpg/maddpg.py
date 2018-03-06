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
from utils.replay_buffer import ReplayBuffer
from config import config

from ddpg_actor_critic import DDPGActorCritic

#TODO: evaluate

class MADDPG(object):
  """
    Owns multiple individual DDPGActorCritic models, one per agent
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
    self.action_dim = self.env.action_space[0].n
    self.observation_dim = self.env.observation_space[0].shape[0]

    # create n DDPGActorCritic object for n agents
    agent_networks = []
    for i in range(env.n):
      agent_networks.append(DDPGActorCritic(i, env, configuration=self.config, logger=logger))

    self.agent_networks = agent_networks


  def build(self):
    for idx, agent_net in enumerate(self.agent_networks):
      var_scope = "agent_" + str(idx)
      with tf.variable_scope(var_scope):
        agent_net.build(var_scope)

  def initialize(self):
    sess = tf.Session()
    for network in self.agent_networks:
        network.initialize(session=sess)

  def sample_n(self, env, replay_buffer, sample_freq, batch_size):
    """
          Sample for all agents for the environment.

          Args:
              sample_freq: only sample after sample_freq steps have been performed
              batch_size:  the number of steps to be sampled

          Returns:
            obs_batch: np.array of shape (None, num_agent, observation_dim)
            act_batch: np.array
            rew_batch: np.array
            next_obs_batch: np.array
            done_mask: np.array

    """
    t = 0

    obs_n = self.current_obs_n
    while (t < sample_freq or (not replay_buffer.can_sample(batch_size))):
      if self.config.render:
        self.env.render()
      idx = replay_buffer.store_frame(obs_n)
      act_n = []  # list of n actions for this step
      for i in range(env.n):
          obs = obs_n[i]
          act = self.agent_networks[i].get_sampled_action(obs)
          act_n.append(act)

      obs_n, rew_n, done_n, info_n = env.step(act_n)
      replay_buffer.store_effect(idx, act_n, rew_n, done_n)
      self.current_obs_n = obs_n

      t += 1
      self.current_episode_length += 1
      if (done_n[0] or self.current_episode_length >= self.config.max_ep_len):
        # reset
        self.current_obs_n = self.env.reset()
        self.current_episode_length = 0

    return replay_buffer.sample(batch_size)

  def train(self):
    replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.observation_dim, self.action_dim, self.env.n)
    self.current_obs_n = self.env.reset()
    self.current_episode_length = 0
    for t in range(self.config.num_batches):
      self.logger.info("Batch " + str(t) + ":")
      samples = self.sample_n(self.env, replay_buffer, self.config.train_freq, self.config.batch_size)
      for i in self.env.n:
        self.logger.info("training for agent " + str(i) + "...")
        agent_net = self.agent_networks[i]
        agent_net.train_for_batch_samples(samples)

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
