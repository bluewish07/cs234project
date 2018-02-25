import numpy as np

from actor_critic import ActionCritic

#TODO: random process N for action exploration

class DDPGActorCritic(ActionCritic):
  """
        Class for implementing a single actor-critic DDPG with Q learning
        This is for a single agent used in MADDPG setting, therefore we need:
        an approximation policy network for each other agent

  """

  ############### Building the model graph ####################

  def build_policy_approx_networks(self):
    """
    Build one network per other agent to estimate what the other agents would do
    :return: None
    """
    policy_approximates = []
    # TODO
    self.policy_approximates = policy_approximates

  def add_update_policy_approx_nets_op(self):
    """
    Add operation to update the policy approximation networks
    See section 4.2 Inferring Policies of Other Agents for loss function and other info
    :return: None
    """
    self.update_policy_approx_networks_op = # TODO

  def add_actor_gradients_op(self):
    """
    http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    :return: None
    """
    self.actor_gradients = # TODO

  def add_optimizer_op(self):
    """
    Apply self.actor_gradients
    :return: None
    """
    self.train_op =  # TODO

  def build(self):
      """
      Build model by adding all necessary variables

      You don't have to change anything here - we are just calling
      all the operations you already defined to build the tensorflow graph.
      """

      # add placeholders for the main network
      self.add_placeholders_op()
      # create actor net
      self.build_policy_network_op()
      # create critic net, which also covers its loss
      self.add_baseline_op()
      # add optmizer for the actor network
      self.add_actor_gradients_op()
      self.add_optimizer_op()

  #################### Running the model ######################



  def train_for_batch_samples(self, samples):
    """
        Train for a batch of samples

            Args:
              samples: a tuple of lists ([obs_t], [a_t], [r_t], [obs_t+1], done_mask)

    """
    observations, true_actions, rewards, _, _ = samples
    # compute Q-val estimates (discounted future returns) for each time step
    returns = self.get_returns(samples)
    advantages = self.calculate_advantage(returns, observations, true_actions)

    # run training operations
    # update this agent's policy approx networks for other agents
    self.sess.run(self.update_policy_approx_networks_op, feed_dict={self.observation_placeholder : observations,
                                                                    self.action_placeholder : true_actions})
    # update centralized Q network
    self.update_baseline(returns, observations, true_actions)
    # update actor network
    approx_actions = self.sess.run(self.policy_approximates, feed_dict={self.observation_placeholder : observations,
                                                                        self.action_placeholder: true_actions})
    self.sess.run(self.train_op, feed_dict={
      self.observation_placeholder: observations,
      self.action_placeholder: approx_actions,
      self.advantage_placeholder: advantages})

    self.env.render()