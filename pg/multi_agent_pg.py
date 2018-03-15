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
from utils.collisions import count_agent_collisions
from utils.distance_from_landmarks import get_distance_from_landmarks

  
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
      temp.append(PG(i, self.env, configuration=self.config, logger=logger))
    self.agents = temp
    
  def build(self):
    for idx, agent_net in enumerate(self.agents):
      var_scope = "agent_" + str(idx)
      with tf.variable_scope(var_scope):
        agent_net.build()

  def initialize(self):
    self.sess = tf.Session()
    self.add_summary()
    for network in self.agents:
        network.initialize(session=self.sess)

  def sample_paths_n(self, num_episodes=None):
    """
          Sample paths for all agents for the environment.
          Prefers num_episodes specified over batch_size defined

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
        
    self.total_rewards = []
    self.collisions = []
    self.agent_distance = []

    for i in range(self.env.n):
      paths_n.append([])

    while (t < num_episodes): # or t < self.config.batch_size):
      obs_n = self.env.reset() # list of n observations after initial setup
      observations_n, actions_n, rewards_n = [], [], [] # holds values for each agent

      for i in range(self.env.n):
        observations_n.append([])
        actions_n.append([])
        rewards_n.append([])
        
      self.episode_reward = 0
      self.episode_collisions = 0
      self.avg_distance_episode = 0
      for step in range(self.config.max_ep_len):         
        if self.config.render:
          self.env.render()
          # time.sleep(30)
        
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

        # collect stats
        temp = np.sum(np.clip(rew_n, -1e10, 1e10)) # for numerical stability
        self.episode_reward += temp # sum reward across agents to give episode reward
        self.episode_collisions += count_agent_collisions(self.env)
        self.avg_distance_episode = get_distance_from_landmarks(self.env)

        t += 1
        if (done_n[0] or step == self.config.max_ep_len - 1):
          break
        if (not num_episodes) and t == self.config.batch_size:
          break

      self.total_rewards.append(self.episode_reward)
      self.collisions.append(self.episode_collisions)
      self.agent_distance.append(self.avg_distance_episode)
        
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

  def evaluate(self, num_episodes = 1):
    self.logger.info("Evaluating ...")

    paths_n = self.sample_paths_n(num_episodes)
    rewards = [0] * num_episodes
    for i in range(self.env.n):
      paths_for_agent = paths_n[i]
      for eps_idx, path in enumerate(paths_for_agent):
        rewards[eps_idx] += np.sum(path["reward"])

    self.avg_reward = np.mean(self.total_rewards)
    self.avg_collisions = np.mean(self.collisions)
    self.avg_distance = np.mean(self.agent_distance)
    self.record_summary(self.current_batch_num)

    sigma_reward = np.sqrt(np.var(self.total_rewards) / len(self.total_rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(self.avg_reward, sigma_reward)
    self.logger.info(msg)

    sigma_collisions = np.sqrt(np.var(self.collisions) / len(self.collisions))
    msg = "Average collisions: {:04.2f} +/- {:04.2f}".format(self.avg_collisions, sigma_collisions)
    self.logger.info(msg)

    sigma_distance = np.sqrt(np.var(self.agent_distance) / len(self.agent_distance))
    msg = "Average distance: {:04.2f} +/- {:04.2f}".format(self.avg_distance, sigma_distance)
    self.logger.info(msg)

  def train(self):
    for t in range(self.config.num_batches):
      self.logger.info("Batch " + str(t) + ":")
      self.current_batch_num = t
      paths_n = self.sample_paths_n(num_episodes=self.config.batch_size_in_episodes)
      for i in range(self.env.n):
        self.logger.info("training for agent " + str(i) + "...")
        agent_net = self.agents[i]
        agent_net.train_for_batch_paths(paths_n[i])

      #if t % self.config.eval_freq == 0:
      self.evaluate(self.config.eval_freq)

    self.logger.info("- Training all done.")

   
  def test(self):
    """
    TODO - test-time running
    """
    pass

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


################################  RECORD AND SUMMARY  #############################################

  def record_summary(self, t):
    fd = {
      self.avg_reward_placeholder: self.avg_reward,
      self.avg_collsions_placeholder: self.avg_collisions,
      self.avg_distance_placeholder: self.avg_distance,
    }

    summary = self.sess.run(self.merged, feed_dict=fd)
    self.file_writer.add_summary(summary, t)

  def add_summary(self):
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.avg_collsions_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_collsions")
    self.avg_distance_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_distance")

    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
    tf.summary.scalar("Avg Collisions", self.avg_collsions_placeholder)
    tf.summary.scalar("Avg Distance", self.avg_distance_placeholder)

    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)

