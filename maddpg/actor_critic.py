import os

import gym
import numpy as np
import tensorflow as tf

from pg import PG
from config import config
from utils.general import get_logger, export_plot
from utils.replay_buffer import ReplayBuffer

#TODO: we need to add target network

class ActionCritic(PG):
  """
    Class for implementing a simple stochastic actor-critic PG Algorithm with Q learning
    This is for a single agent.
    This class inherits most parts from PG but only changes how advantages are calculated.
    Another significant difference is that this class uses experience replay buffer, instead of sampled trajectories
  """

  ############### Building the model graph ####################

  def get_critic_network_op(self, scope="critic_network"):
    """
    Build critic network. Assign it to self.q
    TODO later: add a target network to self.target_q
    :param scope: variable scope used for parameters in this network
    :return: critic_network_op
    """
    #TODO

  def add_baseline_op(self, scope="baseline"):
    """
    In Actor-Critic,
    baseline is basically the current Q network.
    baseline target is the current state-action value given the sample
    To update baseline, we want to minimize MSE between baseline and baseline target
    :param scope: unused
    :return: None
    """
    self.baseline = self.get_critic_network_op()
    self.baseline_target_placeholder =  # TODO
    self.update_baseline_op =  # TODO


  #################### Running the model ######################

  def get_returns(self, samples):
    """
      Calculate the state-action value Q for this time step.
      The function name might be a bit misleading.

      Args:
        samples: a tuple of lists ([obs_t], [a_t], [r_t], [obs_t+1], done_mask)

    """

    q_values = []
    for sample in samples:
      #TODO

    return q_values


  def calculate_advantage(self, returns, observations, actions):
    """
    Calculate the advantage
    Should be very similar to calculate_advantage in PG except that calculating baseline here requires actions as well
    Args:
            returns: all discounted future returns for each step
            observations: observations
              Calculate the advantages, using baseline adjustment if necessary,
              and normalizing the advantages if necessary.
              If neither of these options are True, just return returns.

    TODO:
    If config.use_baseline = False and config.normalize_advantage = False,
    then the "advantage" is just going to be the returns (and not actually
    an advantage).

    if config.use_baseline, then we need to evaluate the baseline and subtract
      it from the returns to get the advantage.
      HINT: 1. evaluate the self.baseline with self.sess.run(...

    if config.normalize_advantage:
      after doing the above, normalize the advantages so that they have a mean of 0
      and standard deviation of 1.

    """
    adv = returns
    #######################################################
    #########   YOUR CODE HERE - 5-10 lines.   ############

    if self.config.normalize_advantage:
      # TODO
    #######################################################
    #########          END YOUR CODE.          ############
    return adv


  def update_baseline(self, returns, observations, actions):
    """
    Update the baseline
    Similar to update_baseline in PG but here we need actions as part of feed_dict as well
    TODO:
      apply the baseline update op with the observations and the returns.
    """
    #######################################################
    #########   YOUR CODE HERE - 1-5 lines.   ############
    pass # TODO
    #######################################################
    #########          END YOUR CODE.          ############


  def train_for_batch_samples(self, samples):
    """
        Train for a batch of samples

            Args:
              samples: a tuple of lists ([obs_t], [a_t], [r_t], [obs_t+1], done_mask)

    """
    observations, actions, rewards, _, _ = samples
    # compute Q-val estimates (discounted future returns) for each time step
    returns = self.get_returns(samples)
    advantages = self.calculate_advantage(returns, observations, actions)

    # run training operations
    if self.config.use_baseline:
      self.update_baseline(returns, observations, actions)
    self.sess.run(self.train_op, feed_dict={
      self.observation_placeholder: observations,
      self.action_placeholder: actions,
      self.advantage_placeholder: advantages})

    self.env.render()


  # def evaluate(self, env=None, num_episodes=1):
  #   """
  #       Evaluation with same procedure as the training
  #   """
  #   # log our activity only if default call
  #   self.logger.info("Evaluating...")
  #
  #   if env is None:
  #       env = self.env
  #
  #   # replay memory to play
  #   replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
  #   rewards = []
  #
  #   for i in range(num_episodes):
  #       total_reward = 0
  #       state = env.reset()
  #       while True:
  #           if self.config.render_test: env.render()
  #
  #           # store last state in buffer
  #           idx = replay_buffer.store_frame(state)
  #           q_input = replay_buffer.encode_recent_observation()
  #
  #           action = self.get_action(q_input)
  #
  #           # perform action in env
  #           new_state, reward, done, info = env.step(action)
  #
  #           # store in replay memory
  #           replay_buffer.store_effect(idx, action, reward, done)
  #           state = new_state
  #
  #           # count reward
  #           total_reward += reward
  #           if done:
  #               break
  #
  #       # updates to perform at the end of an episode
  #       rewards.append(total_reward)
  #
  #   avg_reward = np.mean(rewards)
  #   sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
  #
  #   if num_episodes > 1:
  #       msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
  #       self.logger.info(msg)
  #
  #   return avg_reward
