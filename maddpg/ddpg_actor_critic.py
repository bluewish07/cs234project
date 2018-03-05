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

        IMPORTANT: all the actions we use in this network are vectors of length action_dim.
                   for discrete case, actions are one-hot vectors

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

      # top level scopes
      self.policy_approx_networks_scope = "policy_approx_networks"
      self.actor_network_scope = "actor_network"
      self.critic_network_scope = "critic_network"





  ############### Building the model graph ####################
### shared placeholders
  def add_placeholders_op(self):
    self.state_placeholder = tf.placeholder(tf.float32, shape=(None, self.env.n, self.observation_dim))
    self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
    self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))
    self.action_logits_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))
    self.reward_placeholder = tf.placeholder(tf.float32, shape=(None))


### actor network
  def add_actor_network_placeholders_op(self):
    with tf.variable_scope(self.actor_network_scope):
      self.q_value_placeholder_for_policy_gradient = tf.placeholder(tf.float32, shape=(None))

  def build_policy_network_op(self, scope=None):
    """
    Builds the policy network.
    """
    if scope is None:
      scope = self.actor_network_scope
    self.mu_scope = "mu"
    self.target_mu_scope = "target_mu"
    with tf.variable_scope(scope):
      self.mu = build_mlp(self.observation_placeholder, self.action_dim, self.mu_scope, n_layers=self.config.n_layers, size=self.config.layer_size)
      self.target_mu = build_mlp(self.observation_placeholder, self.action_dim, self.target_mu_scope, n_layers=self.config.n_layers, size=self.config.layer_size)


  def add_actor_gradients_op(self, scope=None):
    """
    http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    :return: None
    """
    if scope is None:
      scope = self.actor_network_scope
    action_gradient = tf.gradients(self.q_value_placeholder_for_policy_gradient, self.action_logits_placeholder)
    combined_scope = scope + "/" + self.mu_scope
    self.mu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, combined_scope)
    batch_actor_gradients = tf.gradients(self.action_logits_placeholder, self.mu_vars, -1 * action_gradient)
    self.actor_gradients = tf.reduce_mean(batch_actor_gradients, axis=0)

  def add_optimizer_op(self):
    """
    Apply self.actor_gradients
    :return: None
    """
    self.train_actor_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.mu_vars))


### actor networks for simulating other agents
  def build_policy_approx_networks(self):
    """
    Build one network per other agent to estimate what the other agents would do
    :return: None
    """
    policy_approximates = []
    log_stds = []
    with tf.variable_scope(self.policy_approx_networks_scope):
      for i in range(self.env.n):
        if i == self.agent_idx:
          policy_approximates.append(None)
          log_stds.append(None)
          continue
        scope = "agent_" + str(i)
        approx = build_mlp(self.observation_placeholder, self.action_dim, scope, self.config.n_layers,
                             self.config.layer_size)
        policy_approximates.append(approx)
        with tf.variable_scope(scope):
          log_std = tf.get_variable("log_std", shape=[self.action_dim], dtype=tf.float32)
          log_stds.append(log_std)

    self.policy_approximates = policy_approximates
    self.policy_approx_log_stds = log_stds

  def add_update_policy_approx_networks_op(self):
    """
    Add operation to update all policy approximation networks.
    See section 4.2 Inferring Policies of Other Agents for loss function and other info
    :return: None
    """

    update_ops = []
    for i in range(self.env.n):
        if i == self.agent_idx:
          update_ops.append(None)
          continue
        log_std = self.policy_approx_log_stds[i]
        #TODO later: we can try other distributions as well?
        dist = tf.contrib.distributions.MultivariateNormalDiag(self.action_logits_placeholder, tf.exp(log_std))
        logprob = dist.log_prob(self.action_placeholder)
        entropy = dist.entropy()
        loss = -tf.reduce_mean(logprob + self.config.policy_approx_lambda * entropy)
        update = tf.train.AdamOptimizer(self.lr).minimize(loss)
        update_ops.append(update)

    self.update_policy_approx_networks_op = update_ops


### critic network
  def add_critic_network_placeholders_op(self):
    with tf.variable_scope(self.critic_network_scope):
        self.actions_n_placeholder = tf.placeholder(tf.float32, shape=(None, self.env.n, self.action_dim))
        # add placeholders for update_critic_network_op
        self.y_placeholder = tf.placeholder(tf.float32, shape=(None))
        self.q_baseline_placeholder = tf.placeholder(tf.float32, shape=(None))

  def add_critic_network_op(self, scope=None):
    """
    Build critic network. Assign it to self.q, self.target_q.
    :param scope: variable scope used for parameters in this network
    :return: None
    """
    if scope is None:
      scope = self.critic_network_scope
    self.q_scope = "q"
    self.target_q_scope = "target_q"
    #TODO: need to fix below
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
  def get_assign_ops(self, scope, target_scope):
    op_list = list()

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_scope)
    vars_by_name_suffix = dict([(var.name[var.name.rfind('/'):], var) for var in vars])
    target_vars_by_name_suffix = dict([(var.name[var.name.rfind('/'):], var) for var in target_vars])
    for var_name_suffix, var in vars_by_name_suffix.iteritems():
      target_var = target_vars_by_name_suffix[var_name_suffix]
      new_target_var = self.config.tau * var + (1 - self.config.tau) * target_var
      assign_op = tf.assign(target_var, new_target_var)
      op_list.append(assign_op)

    return op_list

  def add_update_target_op(self):
      """
      update_target_op will be called periodically
      to update target network weights based on the constantly-changing network
      target_theta <- tau * theta + (1-tau) * target_theta

      """
      op_list = self.get_assign_ops(self.q_scope, self.target_q_scope) + self.get_assign_ops(self.mu_scope, self.target_mu_scope)

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
      self.add_policy_approx_networks_placeholders_op()
      self.build_policy_approx_networks()
      self.add_update_policy_approx_networks_op()
      # create critic net
      self.add_critic_network_placeholders_op()
      self.add_critic_network_op()
      self.add_update_critic_network_op()

      self.add_update_target_op()


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
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)


  #################### Running the model ######################
  def update_policy_approx_networks(self, observations_by_agent, actions_by_agent):
    """

    :param observations_by_agent: shape (num_agent, batch_size, observation_dim)
    :param actions_by_agent: shape (num_agent, batch_size, action_dim)
    :return:
    """
    for i in self.env.n:
      if i == self.agent_idx: continue
      obs = observations_by_agent[i]
      act = actions_by_agent[i]
      approx_network = self.policy_approximates[i]
      action_logits = self.sess.run(approx_network, feed_dict={self.observation_placeholder: obs})
      update_approx_network = self.update_policy_approx_networks_op[i]
      self.sess.run(update_approx_network, feed_dict={self.action_placeholder : act,
                                                      self.action_logits_placeholder : action_logits})

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


  def update_actor_network(self, observation, actions_n, state):
    """

    :param
        observation: batched observations for current agent, shape=(None, observation_dim)
        actions_n: batched n-agent actions
                            shape=(None, num_agent,) if discrete
                            shape=(None, num_agent, action_dim) otherwise
        state: shape=(None, num_agent, observation_dim)
    TODO: we might want more granular input args than samples, to avoid duplicate numpy operations
    :return:
    """
    q_values = self.sess.run(self.q, feed_dict={self.state_placeholder : state,
                                                self.actions_n_placeholder : actions_n})
    _, action_logits = self.get_action_and_logits(observation)
    self.sess.run(self.train_actor_op, feed_dict={self.observation_placeholder : observation,
                                                  self.action_logits_placeholder : action_logits,
                                                  self.q_value_placeholder_for_policy_gradient : q_values})



  def get_action_and_logits(self, observations):
    """
    Run
    :param observations: batched observations
    :return: action: shape=(None, action_dim). if discrete, a single action is one-hot vector
             action_logits: shape=(None, action_dim)
                            for discrete case, this is direct output of mu
                            for none-discrete case, this is the same as action
    """
    action_logits = self.sess.run(self.mu, feed_dict={self.observation_placeholder: observations})
    actions = action_logits
    if self.config.discrete:
      action_indices = tf.argmax(action_logits, axis=1)
      # action_indices = tf.squeeze(tf.multinomial(action_logits, 1), axis=1)
      actions = tf.one_hot(action_indices, self.action_dim)
    return actions, action_logits

  def get_sampled_action(self, observation):
    """
    Get a single action for a single observation, used for stepping the environment

    :param observation: single observation to run self.sampled_action with
    :return: action: if discrete, output is a single number
                     otherwise, shape=(action_dim)
    """
    batch = np.expand_dims(observation, 0)
    actions, action_logits = self.get_action_and_logits(batch)
    action = actions[0]
    if self.config.random_process_exploration:
      with tf.variable_scope("agent_" + str(self.agent_idx)):
        log_std = tf.get_variable("random_process_log_std", shape=[self.action_dim], dtype=tf.float32)
        dist = tf.contrib.distributions.MultivariateNormalDiag(action_logits, tf.exp(log_std))
        action = dist.sample()[0]
    return action


  def train_for_batch_samples(self, samples):
    """
        Train for a batch of samples

            Args:
              samples: a tuple (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask)
                    obs_batch: np.array of shape (None, num_agent, observation_dim)

    """

    state, true_actions, rewards, _, _ = samples
    # update this agent's policy approx networks for other agents
    observations_by_agent = np.swapaxes(state, 0, 1) # shape (num_agent, batch_size, observation_dim)
    true_actions_by_agent = np.swapaxes(true_actions, 0, 1)
    observation_for_current_agent = observations_by_agent[self.agent_idx]
    self.update_policy_approx_networks(observations_by_agent, true_actions_by_agent)

    # get an estimated action from each agent approx network
    # Specifically, for the current agent, get the action from the target policy network
    # for all other agents, get the action from the approx network
    est_actions_by_agent = []
    for i in range(self.env.n):
      observations_i = observations_by_agent[i]
      actions_i = None
      if i == self.agent_idx:
        actions_i = self.sess.run(self.target_mu, feed_dict={self.observation_placeholder : observations_i})
      else:
        actions_i = self.sess.run(self.policy_approximates[i], feed_dict={self.observation_placeholder : observations_i})
      est_actions_by_agent.append(actions_i)
    est_actions = np.swapaxes(est_actions_by_agent, 0, 1)  # shape = (None, num_agent, action_dim)

    # update centralized Q network
    self.update_critic_network(samples, est_actions)

    # update actor network
    self.update_actor_network(observation_for_current_agent, true_actions, state)

    # update target networks
    self.sess.run(self.update_target_op, feed_dict={})
