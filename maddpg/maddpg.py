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
from  utils.collisions import count_agent_collisions
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
    paths = []
    current_path = []
    obs_n = self.current_obs_n
    while (t < sample_freq or (not replay_buffer.can_sample(batch_size))):
      current_path = []
      # NV: Don't render during path sampling, only render during test runs
      # if self.config.render:
        # self.env.render()
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

  def test_run(self, env, num_episodes):
    """
      Do a test run to evaluate the network, and log statistics 
      Does NOT populate the experience replay buffer, as this is for evaluation purposes
    """
    j = 0
    total_rewards = []
    episode_reward = 0
    collisions = []
    episode_collisions = 0
    successes = 0
    obs_n = self.current_obs_n
    
    while j < num_episodes:
      if self.config.render:
        time.sleep(0.1)
        self.env.render()
        continue
      act_n = []  # list of n actions for this step
      for i in range(env.n):
          obs = obs_n[i]
          act = self.agent_networks[i].get_sampled_action(obs, is_evaluation=True)
          act_n.append(act)

      obs_n, rew_n, done_n, info_n = env.step(act_n)
      self.current_obs_n = obs_n
      temp = np.sum(np.clip(rew_n, -1e10, 1e10)) # for numerical stability
      episode_reward += temp # sum reward across agents to give episode reward
      
      episode_collisions += count_agent_collisions(self.env)
      
      # define a "successful" episode as one where every agent has a reward > -0.1
      # this definition comes from the benchmark_data function in multi-agent-envs simple_spread.py definition 
      # reward = -1 * distance from agent to a landmark
      if np.mean(rew_n) > -0.1:
        successes += 1
      

      self.current_episode_length += 1
      if (any(done_n) or self.current_episode_length >= self.config.max_ep_len):
        # end the existing episode
        total_rewards.append(episode_reward)
        collisions.append(episode_collisions)
        j += 1
        
        # reset
        self.current_obs_n = self.env.reset()
        self.current_episode_length = 0
        
        episode_reward = 0
        episode_collisions = 0
        
        
    # log average episode reward
    avg_reward = np.mean(total_rewards)
    sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
    msg = "Evaluating...Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    
    # log # of collisions
    avg_collisions = np.mean(collisions)
    sigma_collisions = np.sqrt(np.var(collisions) / len(collisions))
    msg = "Average collisions: {:04.2f} +/- {:04.2f}".format(avg_collisions, sigma_collisions)
    self.logger.info(msg)
    
    # log # of successes
    msg = "Successful episodes: {:d}".format(successes)
    self.logger.info(msg)
        
  def train(self):
    replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.observation_dim, self.action_dim, self.env.n)
    self.current_obs_n = self.env.reset()
    self.current_episode_length = 0
    for t in range(self.config.num_batches):
      self.logger.info("Batch " + str(t) + ":")
      samples = self.sample_n(self.env, replay_buffer, self.config.train_freq, self.config.batch_size)
      for i in range(self.env.n):
        self.logger.info("training for agent " + str(i) + "...")
        agent_net = self.agent_networks[i]
        agent_net.train_for_batch_samples(samples)
      
      # NV: every batch, do a test_run and print average reward)
      # change this if we want to sample more often
      if True:
        self.test_run(self.env, self.config.batch_size_in_episodes)
        

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

  ############### summary and logging ###############

  def record_summary(self, t):
      """
          Add summary to tfboard

          You don't have to change or use anything here.
          """

      fd = {
        self.avg_reward_placeholder: self.avg_reward,
        self.max_reward_placeholder: self.max_reward,
        self.std_reward_placeholder: self.std_reward,
        self.eval_reward_placeholder: self.eval_reward,
      }
      summary = self.sess.run(self.merged, feed_dict=fd)
      # tensorboard stuff
      self.file_writer.add_summary(summary, t)

  def add_summary(self):
      """
          Tensorboard stuff.

          You don't have to change or use anything here.
          """
      # extra placeholders to log stuff from python
      self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
      self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
      self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

      self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

      # extra summaries from python -> placeholders
      tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
      tf.summary.scalar("Max Reward", self.max_reward_placeholder)
      tf.summary.scalar("Std Reward", self.std_reward_placeholder)
      tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

      # logging
      self.merged = tf.summary.merge_all()
      self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)
