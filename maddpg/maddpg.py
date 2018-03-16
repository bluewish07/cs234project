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
from utils.collisions import count_agent_collisions
from utils.distance_from_landmarks import get_distance_from_landmarks
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
    self.sess = tf.Session()
    self.add_summary()
    init = tf.global_variables_initializer()
    self.sess.run(init)
    for network in self.agent_networks:
      network.initialize(session=self.sess)

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
      idx = replay_buffer.store_frame(obs_n)
      act_n = []  # list of n actions for this step
      for i in range(env.n):
          obs = obs_n[i]
          act = self.agent_networks[i].get_sampled_action(obs)
          if self.config.action_clip:
            act = np.clip(act, -2, 2)
          act_n.append(act)

      obs_n, rew_n, done_n, info_n = env.step(act_n)
      # next_obs_n, rew_n, done_n, info_n = env.step(act_n)
      if self.config.scale_reward: rew_n = [rew * .1 for rew in rew_n]
      replay_buffer.store_effect(idx, act_n, rew_n, done_n, obs_n)
      # replay_buffer.remember(obs_n, act_n, rew_n, next_obs_n, done_n[0])

      t += 1
      self.current_episode_length += 1
      if (any(done_n) or self.current_episode_length >= self.config.max_ep_len):
        # reset
        # print(act_n)
        # print(rew_n)
        obs_n = self.env.reset()
        self.current_episode_length = 0
        
      self.current_obs_n = obs_n
    # return replay_buffer.sample(batch_size, env.n)
    return replay_buffer.sample(batch_size)

  def test_run(self, env, num_episodes):
    """
      Do a test run to evaluate the network, and log statistics 
      Does NOT populate the experience replay buffer, as this is for evaluation purposes
    
      NOTE: We do not clip the action here, since we do not want to explore, we want to just exploit
      the path we believe is best so far
    """
    j = 0
    total_rewards = []
    collisions = []
    agent_distance = []
    successes = 0

    obs_n = self.env.reset()
    episode_length = 0
    
    while j < num_episodes:
      if self.config.render:
        time.sleep(0.1)
        self.env.render()

      # initialize metrics before start of an episode
      episode_reward = 0
      episode_collisions = 0
      avg_distance_episode = 0
      
      #reset observation after every episode
      obs_n = self.env.reset()
      for i in range(self.config.max_ep_len):
        act_n = []  # list of n actions for this step

        for i in range(env.n):
          obs = obs_n[i]
          act = self.agent_networks[i].get_sampled_action(obs, is_evaluation=True)
          act_n.append(act)

        obs_n, rew_n, done_n, info_n = env.step(act_n)
        #episode_length += 1
        temp = np.sum(np.clip(rew_n, -1e10, 1e10)) # for numerical stability
        episode_reward += temp # sum reward across agents to give episode reward
      
        episode_collisions += count_agent_collisions(self.env)
      
        # define a "successful" episode as one where every agent has a reward > -0.1
        # this definition comes from the benchmark_data function in multi-agent-envs simple_spread.py definition 
        # reward = -1 * distance from agent to a landmark
        if np.mean(rew_n) > -0.1:
          successes += 1
      
        # distance of agents from landmarks are needed only at the end of an episode
        avg_distance_episode += get_distance_from_landmarks(self.env)

        total_rewards.append(episode_reward)
        collisions.append(episode_collisions)
        agent_distance.append(avg_distance_episode)

      #increment episode counter 
      j += 1
        
    # log average episode reward
    self.avg_reward = np.mean(total_rewards)
    sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(self.avg_reward, sigma_reward)
    self.logger.info(msg)
    
    # log # of collisions
    self.avg_collisions = np.mean(collisions)
    sigma_collisions = np.sqrt(np.var(collisions) / len(collisions))
    msg = "Average collisions: {:04.2f} +/- {:04.2f}".format(self.avg_collisions, sigma_collisions)
    self.logger.info(msg)
    
    # log of average agent distance
    self.avg_distance = np.mean(agent_distance)
    sigma_agent_distance = np.sqrt(np.var(agent_distance) / len(agent_distance))
    msg = "Average distance from landmarks: {:04.2f} +/- {:04.2f}".format(self.avg_distance, sigma_agent_distance)
    self.logger.info(msg)

    # log # of successes
    msg = "Successful episodes: {:d}".format(successes)
    self.logger.info(msg)

    self.record_summary(self.current_batch_num)
        
  def train(self):
    start = time.time()
    replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.observation_dim, self.action_dim, self.env.n)
    # replay_buffer = Memory(self.config.replay_buffer_size)
    self.current_obs_n = self.env.reset()
    self.current_episode_length = 0
    for t in range(self.config.num_batches):
      self.current_batch_num = t
      
      samples = self.sample_n(self.env, replay_buffer, self.config.train_freq, self.config.batch_size)
      for i in range(self.env.n):
        agent_net = self.agent_networks[i]
        agent_net.train_for_batch_samples(samples, agents_list=self.agent_networks)
      
      # NV: every batch, do a test_run and print average reward)
      # change this if we want to sample more often
      if t % self.config.eval_freq == 0:
        self.logger.info("Batch " + str(t) + ":")
        self.test_run(self.env, self.config.eval_episodes)

    self.logger.info("- Training all done.")
    self.logger.info("Total training time: " + str(time.time() - start) + " seconds.")
    

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
        self.avg_collsions_placeholder: self.avg_collisions,
        self.avg_distance_placeholder: self.avg_distance,
        #self.eval_reward_placeholder: self.eval_reward,
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
      self.avg_collsions_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_collsions")
      self.avg_distance_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_distance")

      # self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

      # extra summaries from python -> placeholders
      tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
      tf.summary.scalar("Avg Collisions", self.avg_collsions_placeholder)
      tf.summary.scalar("Avg Distance", self.avg_distance_placeholder)
      # tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

      # logging
      self.merged = tf.summary.merge_all()
      self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)
