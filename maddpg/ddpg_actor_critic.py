import numpy as np
import os

import gym
import numpy as np
import tensorflow as tf
from config import config
from utils.general import get_logger, export_plot
from utils.network import build_mlp

#TODO: random process N for action exploration

class DDPGActorCritic(object):
  """
        Class for implementing a single actor-critic DDPG with Q learning
        This is for a single agent used in MADDPG setting, therefore we need:
        an approximation policy network for each other agent

  """

  def __init__(self, agent_idx, env, configuration, logger=None):
      self.agent_idx = agent_idx # the index of this agent
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
### shared placeholders
  def add_placeholders_op(self):
    self.state_placeholder = tf.placeholder(tf.float32, shape=(None, self.env.n, self.observation_dim))
    self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
    if self.config.discrete:
      self.action_placeholder = tf.placeholder(tf.int32, shape=(None))
    else:
      self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))
    self.reward_placeholder = tf.placeholder(tf.float32, shape=(None))


### actor network
  def add_actor_network_placeholders_op(self):
    self.q_value_placeholder_for_policy_gradient = tf.placeholder(tf.float32, shape=(None))
    self.action_logits_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))

  def build_policy_network_op(self, scope = "policy_network"):
    """
    Builds the policy network.
    """
    mu_scope = "mu"
    target_mu_scope = "target_mu"
    with tf.variable_scope(scope):
      self.mu = build_mlp(self.observation_placeholder, self.action_dim, mu_scope, n_layers=self.config.n_layers, size=self.config.layer_size)
      self.target_mu = build_mlp(self.observation_placeholder, self.action_dim, target_mu_scope, n_layers=self.config.n_layers, size=self.config.layer_size)


  def add_actor_gradients_op(self):
    """
    http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    :return: None
    """
    action_gradient = tf.gradients(self.q_value_placeholder_for_policy_gradient, self.action_placeholder)
    batch_actor_gradients = tf.gradients(self.action_logits_placeholder, )
    self.actor_gradients = # TODO

  def add_optimizer_op(self):
    """
    Apply self.actor_gradients
    :return: None
    """
    self.train_op =  # TODO


### actor networks for simulating other agents
  def build_policy_approx_networks(self):
    """
    Build one network per other agent to estimate what the other agents would do
    :return: None
    """
    policy_approximates = []
    # TODO
    self.policy_approximates = policy_approximates

  def add_update_policy_approx_network_op(self):
    """
    Add operation to update a single policy approximation network.
    When running, we will repeat this op for each agent approx network.
    See section 4.2 Inferring Policies of Other Agents for loss function and other info
    :return: None
    """
    self.update_policy_approx_network_op = # TODO


### critic network
  def add_critic_network_placeholders_op(self):
    #TODO: add a placeholder for all agent's action stacked, shape = (None, num_agents, action_dim)
    self.actions_n_placeholder = #TODO
    # add placeholders for update_critic_network_op
    self.y_placeholder = #TODO
    self.q_baseline_placeholder = #TODO

  def add_critic_network_op(self, scope="critic_network"):
    """
    Build critic network. Assign it to self.q, self.target_q
    :param scope: variable scope used for parameters in this network
    :return: None
    """
    #TODO: need to fix
    q_scope = "q"
    target_q_scope = "target_q"
    with tf.variable_scope(scope):
      if self.discrete:
        self.q = build_mlp(self.observation_placeholder, self.action_dim, scope=q_scope)
        self.target_q = build_mlp(self.observation_placeholder, self.action_dim, scope=target_q_scope)
      else:
        input = tf.concat([self.observation_placeholder, self.action_placeholder], axis=1)
        self.q = build_mlp(input, 1, scope=q_scope)
        self.target_q = build_mlp(input, 1, scope=target_q_scope)

  def add_update_critic_network_op(self):
    loss = tf.losses.mean_squared_error(self.y_placeholder, self.q_baseline_placeholder)
    self.update_critic_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


### update target networks
  def add_update_target_op(self, scope, target_scope):
      """
      update_target_op will be called periodically
      to update target network weights based on the constantly-changing network
      target_theta <- tau * theta + (1-tau) * target_theta


      Args:
          scope: (string) name of the scope of variables for the most up-to-date network
          target_scope: (string) name of the scope of variables
                      for the target network
      """
      #TODO below needs to be updated
      op_list = list()

      q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, q_scope)
      target_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_q_scope)
      q_vars_by_name_suffix = dict([(q_var.name[q_var.name.find('/'):], q_var) for q_var in q_vars])
      target_q_vars_by_name_suffix = dict([(var.name[var.name.find('/'):], var) for var in target_q_vars])
      for q_var_name_suffix, q_var in q_vars_by_name_suffix.iteritems():
          # q_var = tf.Print(q_var, [q_var], "current_q", summarize=20)
          target_q_var = target_q_vars_by_name_suffix[q_var_name_suffix]
          # target_q_var = tf.Print(target_q_var, [target_q_var], "target before", summarize=20)
          target_q_var = tf.assign(target_q_var, q_var)
          # target_q_var = tf.Print(target_q_var, [target_q_var], "target after", summarize=20)
          op_list.append(target_q_var)

      self.update_target_op = tf.group(*op_list)


  def build(self):
      """
      Build model by adding all necessary variables

      You don't have to change anything here - we are just calling
      all the operations you already defined to build the tensorflow graph.
      """

      # add shared placeholders
      self.add_placeholders_op()
      # create actor net
      self.add_actor_network_placeholders_op()
      self.build_policy_network_op()
      self.add_actor_gradients_op()
      self.add_optimizer_op()
      # create actor approx nets
      self.build_policy_approx_networks()
      self.add_update_policy_approx_nets_op()
      # create critic net
      self.add_critic_network_placeholders_op()
      self.add_critic_network_op()
      self.add_update_critic_network_op()

      self.add_update_target_op()




  #################### Running the model ######################

  def update_critic_network(self, samples, estimated_actions):
    """
    Update the critic network
    Args:
        samples: a tuple (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask)
                obs_batch: np.array of shape (None, num_agent, observation_dim)
    TODO: we might want more granular input args than samples, to avoid duplicate numpy operations
    """
    #TODO fix and finish
    q_values = []
    batched_next_obs = np.array(samples[3])
    batched_r = np.array(samples[2])
    done_mask = samples[4]
    if self.discrete:
        target_q_values = self.sess.run(self.target_q, feed_dict={self.observation_placeholder: batched_next_obs})
        a_indices = np.array(list(enumerate(samples[1])))
        target_q_a_values = tf.gather_nd(target_q_values, a_indices)
        q_samp = batched_r + self.config.gamma * target_q_a_values * tf.cast(tf.logical_not(self.done_mask),
                                                                             dtype=tf.float32)
    for sample in samples:
        q_values.append()


  def update_actor_network(self, samples):
    """

    :param samples:
    TODO: we might want more granular input args than samples, to avoid duplicate numpy operations
    :return:
    """
    #TODO

  def get_sampled_action(self, observation):
    """
    Run self.sample_action op

    :param observation: single observation to run self.sampled_action with
    :return: action
    """
    batch = np.expand_dims(observation, 0)
    action_logits = self.sess.run(self.mu, feed_dict={self.observation_placeholder: batch})[0]
    action = action_logits
    if self.config.discrete:
      action = tf.argmax(action_logits, axis=1)
      # action = tf.squeeze(tf.multinomial(action_logits, 1), axis=1)
    return action


  def train_for_batch_samples(self, samples):
    """
        Train for a batch of samples

            Args:
              samples: a tuple (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask)
                    obs_batch: np.array of shape (None, num_agent, observation_dim)

    """

    observations, true_actions, rewards, _, _ = samples
    # update this agent's policy approx networks for other agents
    observations_by_agent = np.swapaxes(observations, 0, 1) # shape (num_agent, batch_size, observation_dim)
    true_actions_by_agent = np.swapaxes(true_actions, 0, 1)
    for i in self.env.n:
      obs = observations_by_agent[i]
      act = true_actions_by_agent[i]
      #TODO: may need to re-think this after implementing update_policy_approx_networks_op
      self.sess.run(self.update_policy_approx_networks_op, feed_dict={self.observation_placeholder : obs,
                                                                    self.action_placeholder : act})

    # get an estimated action from each agent approx network
    # Specifically, for the current agent, get the action from the target policy network
    # for all other agents, get the action from the approx network
    #TODO
    est_actions = #TODO # shape = (None, num_agent, action_dim)

    # update centralized Q network
    self.update_critic_network(samples, est_actions)

    # update actor network
    self.update_actor_network(samples)

    # update target networks
    self.sess.run(self.update_target_op, feed_dict={})
