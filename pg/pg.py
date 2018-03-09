# -*- coding: UTF-8 -*-

import os

import gym
import numpy as np
import tensorflow as tf
from config import config
from utils.general import get_logger, export_plot
from utils.network import build_mlp

def build_mlp(
          mlp_input, 
          output_size,
          scope, 
          n_layers=config.n_layers, 
          size=config.layer_size, 
          output_activation=None):
  with tf.variable_scope(scope):
    h = mlp_input # handle case where n_layers = 0 or 1
    
    for i in range(n_layers):
      h = tf.layers.dense(h, size, activation=tf.nn.relu)
      
    # make output layer
    out = tf.layers.dense(h, output_size, activation = output_activation)
  
  return out


class PG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, agent_idx, env, configuration, logger=None):
    self.agent_idx = agent_idx
    self.config = configuration

    # directory for training outputs
    if not os.path.exists(self.config.output_path):
      os.makedirs(self.config.output_path)
            
    # store hyper-params
    self.logger = logger
    if logger is None:
      self.logger = get_logger(self.config.log_path)
    self.env = env
  
    # action space for a given agent - is Discrete(5) for simple_spread
    # NOTE: assumes that all agents have the same action space for now
    # TODO: action_dim as a argument to this function, so it can vary by agent
    
    # TODO: for simple_spread we don't need to worry about communication space
    # however, for senarios with communication channels, we will need to re-look at this
    #  as action_space seems to be a tuple of movement space and comm space
    self.action_dim = self.env.action_space[0].n

    # observation space for a given agent - is Box(18) for simple_spread
    # NOTE: assumes that all agents have the same observation space for now
    # TODO: observation_dim as a argument to this function, so it can vary by agent
    self.observation_dim = self.env.observation_space[0].shape[0]
    
    self.lr = self.config.learning_rate

############### Building the model graph ####################
  
  
  def add_placeholders_op(self):
    self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
    
    # NOTE: action_placeholder is just a number, but the actual action outputted has to be converted into a one-hot vector before being used in the environment
    self.action_placeholder = tf.placeholder(tf.int32, shape=(None))
  
    # Define a placeholder for advantages
    self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None))
  
  
  def build_policy_network_op(self, scope = "policy_network"):
    """
    Builds the policy network. Note that sampled_action needs to be a onehot vector in order
    to work with multiagent environments.
    """
    action_logits =         build_mlp(self.observation_placeholder, self.action_dim, scope)
    self.sampled_action =   tf.squeeze(tf.multinomial(action_logits, 1), axis=1)
    self.logprob =          -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=self.action_placeholder)
            
  def add_loss_op(self):
    self.loss = -tf.reduce_mean(self.logprob*self.advantage_placeholder)
  
  
  def add_optimizer_op(self):
    opt1 = tf.train.AdamOptimizer(learning_rate = self.lr)
    self.train_op = opt1.minimize(self.loss)
  
  
  def add_baseline_op(self, scope = "baseline"):
    self.baseline = tf.squeeze(build_mlp(self.observation_placeholder, 1, scope))
    self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None))
    opt2 = tf.train.AdamOptimizer(learning_rate = self.lr)
    self.update_baseline_op = opt2.minimize(tf.losses.mean_squared_error(self.baseline_target_placeholder, self.baseline))
  
  def build(self):
    """
    Build model by adding all necessary variables

    You don't have to change anything here - we are just calling
    all the operations you already defined to build the tensorflow graph.
    """
  
    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()
  
    if self.config.use_baseline:
      self.add_baseline_op()


#################### Running the model ######################
  
  def initialize(self, session=None):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    You don't have to change or use anything here.
    """
    # create tf session if not given
    if session is None:
      self.sess = tf.Session()
    else:
      self.sess = session
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)
  
  
  def get_returns(self, paths):
    all_returns = []
    for path in paths:
      rewards = path["reward"]
      
      discounted_return = 0
      path_returns = []
      for i in range(len(rewards)):
        discounted_return += np.power(self.config.gamma, i)*rewards[i]
      path_returns.append(discounted_return)
      for i in range(len(rewards)-1):
        discounted_return = (discounted_return - rewards[i])/self.config.gamma
        path_returns.append(discounted_return)
      all_returns.append(path_returns)
    returns = np.concatenate(all_returns)
  
    return returns


  def calculate_advantage(self, returns, observations):
    adv = returns
    if self.config.use_baseline:
      base = self.sess.run(self.baseline, feed_dict={
                    self.observation_placeholder : observations, 
                    })
      adv = returns - base
    if self.config.normalize_advantage:
      mean = np.mean(adv)
      std = np.sqrt(np.var(adv))
      adv = (adv - mean)/std
    return adv
  
  
  def update_baseline(self, returns, observations):
    self.sess.run(self.update_baseline_op, feed_dict={
                self.observation_placeholder: observations,
                self.baseline_target_placeholder: returns
                })
  def get_sampled_action(self, observation):
    """
    Run self.sample_action op

    :param observation: single observation to run self.sampled_action with
    :return: action
    """
    batch = np.expand_dims(observation, 0)
    action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder: batch})[0]
    return action
                
                
  def train_for_batch_paths(self, paths):
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    rewards = np.concatenate([path["reward"] for path in paths])
    # compute Q-val estimates (discounted future returns) for each time step
    returns = self.get_returns(paths)
    advantages = self.calculate_advantage(returns, observations)

    # run training operations
    if self.config.use_baseline:
      self.update_baseline(returns, observations)
    self.sess.run(self.train_op, feed_dict={
      self.observation_placeholder: observations,
      self.action_placeholder: actions,
      self.advantage_placeholder: advantages})

    # compute reward statistics for this batch and log
    total_rewards = []
    for path in paths:
      path_rewards = path["reward"]
      total = 0
      for r in path_rewards:
        total += r
      total_rewards.append(total)

    avg_reward = np.mean(total_rewards)
    sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
    msg = "Agent {} average reward: {:04.2f} +/- {:04.2f}".format(self.agent_idx, avg_reward, sigma_reward)
    self.logger.info(msg)

##################### For running/training a single PG model only #######################

  def sample_path(self, env, num_episodes=None):
    """
    NV: Not sure this method is relevant at all, as any path generation has to involve
    multiple agents
    
    
    Sample path for the environment.

    Args:
            num_episodes:   the number of episodes to be sampled
              if none, sample one batch (size indicated by config file)
    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["observation"] a numpy array of ordered observations in the path
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"

    You do not have to implement anything in this function, but you will need to
    understand what it returns, and it is worthwhile to look over the code
    just so you understand how we are taking actions in the environment
    and generating batches to train on.
    """
    episode = 0
    episode_rewards = []
    paths = []
    t = 0

    while (num_episodes or t < self.config.batch_size):
      state = env.reset()
      states, actions, rewards = [], [], []
      episode_reward = 0

      for step in range(self.config.max_ep_len):
        states.append(state)
        action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder: states[-1][None]})[0]
        # DH TODO: here we need to modify the sample_path method so that we could accomodate a list of actions, rewards and observations
        state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if (done or step == self.config.max_ep_len - 1):
          episode_rewards.append(episode_reward)
          break
        if (not num_episodes) and t == self.config.batch_size:
          break

      path = {"observation": np.array(states),
              "reward": np.array(rewards),
              "action": np.array(actions)}
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break

    return paths, episode_rewards

  def train(self):
    """
    Performs training

    You do not have to change or use anything here, but take a look
    to see how all the code you've written fits together!
    """
    last_eval = 0 
    last_record = 0
    
    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time
  
    for t in range(self.config.num_batches):
  
      # collect a minibatch of samples
      paths, total_rewards = self.sample_path(self.env) 
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)
      advantages = self.calculate_advantage(returns, observations)

      # run training operations
      if self.config.use_baseline:
        self.update_baseline(returns, observations)
      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations, 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : advantages})
  
      # tf stuff
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)
  
      if  self.config.record and (last_record > self.config.record_freq):
        self.logger.info("Recording...")
        last_record =0
        self.record()
  
    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)


  def evaluate(self, env=None, num_episodes=1):
    """
    Evaluates the return for num_episodes episodes.
    Not used right now, all evaluation statistics are computed during training 
    episodes.
    """
    if env==None: env = self.env
    paths, rewards = self.sample_path(env, num_episodes)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    return avg_reward

  

  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # build model
    self.build()
    # initialize
    self.initialize()
    # record one game at the beginning
    if self.config.record:
        self.record()
    # model
    self.train()
    # record one game at the end
    if self.config.record:
      self.record()

################### Reward tracking ######################

  def init_averages(self):
    """
    Defines extra attributes for tensorboard.

    You don't have to change or use anything here.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.

  def update_averages(self, rewards, scores_eval):
    """
    Update the averages.

    You don't have to change or use anything here.

    Args:
            rewards: deque
            scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]

################### Summary and Recording etc #######################

#DH TODO: I don't think we need record here, instead we should plug in env.render() as appropriate
  def record(self):
     """
     Re create an env and record a video for one episode
     """
     pass
     #env = gym.make(self.config.env_name)
     #env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
     #self.evaluate(env, 1)

#DH TODO: summary stuff might need to be consolidated and moved to multi_agent_pg
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
