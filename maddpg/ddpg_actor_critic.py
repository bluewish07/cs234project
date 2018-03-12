import os
import math

import gym
import numpy as np
import tensorflow as tf
from config import config
from utils.general import get_logger, export_plot
from utils.network import build_mlp, entropy
from utils.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise


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

        # Noise to simulate the random process
        # TODO: consider making this an input parameter so we can tweak it?
        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(env.action_space[0].n),
            sigma=0.3,
            theta=0.15,
            dt=1e-2,
            x0=None)

        self.t = 0


        ############### Building the model graph ####################
    ### shared placeholders
    def add_placeholders_op(self):
        self.state_placeholder = tf.placeholder(tf.float32, shape=(None, self.env.n, self.observation_dim))
        self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.action_logits_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.reward_placeholder = tf.placeholder(tf.float32, shape=(None))



    ### actor networks for simulating other agents
    def build_policy_approx_networks(self):
        """
        Build one network per other agent to estimate what the other agents would do
        :return: None
        """
        policy_approximates = []
        logprobs = []
        log_stds = []
        with tf.variable_scope(self.policy_approx_networks_scope):
            for i in range(self.env.n):
                if i == self.agent_idx:
                    policy_approximates.append(None)
                    log_stds.append(None)
                    logprobs.append(None)
                    continue
                scope = "agent_" + str(i)
                approx = build_mlp(self.observation_placeholder, self.action_dim, scope, self.config.n_layers,
                                                       self.config.layer_size, output_activation=tf.nn.softmax)
                # approx = build_mlp(self.observation_placeholder, self.action_dim, scope, self.config.n_layers,
                #                    self.config.layer_size, output_activation=None)
                # logprobs.append(tf.nn.softmax(approx, axis=-1))
                policy_approximates.append(approx)
                with tf.variable_scope(scope):
                    log_std = tf.get_variable("log_std", shape=[self.action_dim], dtype=tf.float32)
                    log_stds.append(log_std)

        logprobs = policy_approximates
        self.policy_approximates = policy_approximates
        self.policy_approximates_log_probs = logprobs
        self.policy_approx_log_stds = log_stds

    def add_update_policy_approx_networks_op(self):
        """
        Add operation to update all policy approximation networks.
        See section 4.2 Inferring Policies of Other Agents for loss function and other info
        :return: None
        """
        self.policy_approx_networks_losses = [None] * self.env.n
        self.policy_approx_grad_norms = [None] * self.env.n

        update_ops = []
        for i in range(self.env.n):
            if i == self.agent_idx:
                update_ops.append(None)
                continue
            log_std = self.policy_approx_log_stds[i]
            #TODO later: we can try other distributions as well?
            prob = self.policy_approximates_log_probs[i]
            logits = self.policy_approximates[i]
            action_logits = tf.nn.softmax(self.action_placeholder)
            logprob = -tf.nn.softmax_cross_entropy_with_logits(labels=action_logits, logits=logits)
            dist = tf.contrib.distributions.Categorical(probs=prob)
            logits_entropy = dist.entropy()
            #logits_entropy = entropy(logits)
            loss = -tf.reduce_mean(logprob + self.config.policy_approx_lambda * logits_entropy)
            self.policy_approx_networks_losses[i] = tf.Print(loss, [loss], message="agent_"+str(i)+" loss:")
            optimizer = tf.train.AdamOptimizer(self.lr)
            combined_scope = self.agent_scope + "/" + self.policy_approx_networks_scope + "/" + "agent_" + str(i)
            vars_in_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, combined_scope)
            grads_vars = optimizer.compute_gradients(loss, var_list=vars_in_scope)
            grad_norm = tf.global_norm([grad for (grad, _) in grads_vars])
            self.policy_approx_grad_norms[i] = tf.Print(grad_norm, [grad_norm], message="agent_"+str(i)+" grad norm:")
            update = optimizer.apply_gradients(grads_vars)

            # update = tf.train.AdamOptimizer(self.lr).minimize(loss)
            update_ops.append(update)

        self.update_policy_approx_networks_op = update_ops


    ### critic network
    def add_critic_network_placeholders_op(self):
        with tf.variable_scope(self.critic_network_scope):
            self.actions_n_placeholder = tf.placeholder(tf.float32, shape=(None, self.env.n, self.action_dim))
            self.q_next_placeholder = tf.placeholder(tf.float32, shape=(None))
            self.done_mask_placeholder = tf.placeholder(tf.bool, shape=[None])

    def add_critic_network_op(self):
        """
        Build critic network. Assign it to self.q, self.target_q.
        :param scope: variable scope used for parameters in this network
        :return: None
        """
        self.q_scope = "q"
        self.target_q_scope = "target_q"
        with tf.variable_scope(self.critic_network_scope):
            input = tf.concat([tf.layers.flatten(self.state_placeholder), tf.layers.flatten(self.actions_n_placeholder)],
                              axis=1)
            self.q = build_mlp(input, 1, self.q_scope, self.config.n_layers, self.config.layer_size)
            self.target_q = build_mlp(input, 1, self.target_q_scope, self.config.n_layers, self.config.layer_size)
            if self.config.debug_logging:
                self.q = tf.Print(self.q, [self.q], message="q", summarize=20)
                self.target_q = tf.Print(self.target_q, [self.target_q], message="target_q", summarize=20)

    def add_update_critic_network_op(self):
        future_q = self.q_next_placeholder * tf.cast(tf.logical_not(self.done_mask_placeholder), dtype=tf.float32)
        # if self.config.debug_logging: future_q = tf.Print(future_q, [future_q, tf.shape(future_q)], message="future q")
        y = self.reward_placeholder + self.config.gamma * future_q
        if self.config.debug_logging: y = tf.Print(y, [y, tf.shape(y)], message="y")
        loss = tf.losses.mean_squared_error(y, tf.squeeze(self.q, axis=1))
        self.critic_network_loss = tf.Print(loss, [loss], message="critic network loss: ")
        optimizer = tf.train.AdamOptimizer(self.lr)
        combined_scope = self.agent_scope + "/" + self.critic_network_scope + "/" + self.q_scope
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, combined_scope)
        grads_vars = optimizer.compute_gradients(loss, var_list=q_vars)
        grad_norm = tf.global_norm([grad for (grad, _) in grads_vars])
        self.critic_network_grad_norm = tf.Print(grad_norm, [grad_norm], message="critic network grad_norm: ")
        if self.config.grad_clip:
            variables = [v for g,v in grads_vars]
            clipped = [tf.clip_by_norm(g, self.config.clip_val) for g,v in grads_vars]
            grad_norm = tf.global_norm(clipped)
            self.critic_network_grad_norm = tf.Print(grad_norm, [grad_norm], message="critic network grad_norm: ")
            grads_vars = zip(clipped, variables)
        self.update_critic_op = optimizer.apply_gradients(grads_vars)


    def build_policy_network_op(self):
        """
            Builds the policy network.
        """
        self.mu_scope = "mu"
        self.target_mu_scope = "target_mu"
        with tf.variable_scope(self.actor_network_scope):
            # self.mu = build_mlp(self.observation_placeholder, self.action_dim, self.mu_scope,
            #                     n_layers=self.config.n_layers, size=self.config.layer_size, output_activation=tf.nn.softmax)

            self.mu = build_mlp(self.observation_placeholder, self.action_dim, self.mu_scope,
                                n_layers=self.config.n_layers, size=self.config.layer_size,
                                output_activation=None, use_batch_normalization=self.config.use_batch_normalization)
            if self.config.debug_logging: self.mu = tf.Print(self.mu, [self.mu], message="mu", summarize=20)

            self.mu_noise = tf.nn.softmax(self.mu - tf.log(-tf.log(tf.random_uniform(tf.shape(self.mu)))), axis=-1)
            if self.config.debug_logging: self.mu_noise = tf.Print(self.mu_noise, [self.mu_noise], summarize=10,
                                                                   message="action logits")



            # self.target_mu = build_mlp(self.observation_placeholder, self.action_dim, self.target_mu_scope,
            #                            n_layers=self.config.n_layers, size=self.config.layer_size, output_activation=tf.nn.softmax)
            self.target_mu = build_mlp(self.observation_placeholder, self.action_dim, self.target_mu_scope,
                                       n_layers=self.config.n_layers, size=self.config.layer_size,
                                       output_activation=None, use_batch_normalization=self.config.use_batch_normalization)
            self.target_mu_noise = tf.nn.softmax(self.target_mu - tf.log(-tf.log(tf.random_uniform(tf.shape(self.target_mu)))), axis=-1)


        # if self.config.random_process_exploration:
        #     # log_std = tf.get_variable("random_process_log_std", shape=[self.action_dim], dtype=tf.float32)
        #     log_std = tf.fill([self.action_dim], math.log(self.config.sampling_std))
        #     std = tf.exp(log_std)
        #     #std = tf.Print(std, [std], message="std dev: ")
        #     dist = tf.contrib.distributions.MultivariateNormalDiag(self.mu, std)
        #     self.sample_action_op = dist.sample()
        self.sample_action_op = self.mu_noise


    def add_actor_loss_op(self):
        slice_1 = tf.slice(self.actions_n_placeholder, [0, 0, 0], [self.config.batch_size, self.agent_idx, self.action_dim])
        slice_2 = tf.slice(self.actions_n_placeholder, [0, self.agent_idx+1, 0], [self.config.batch_size, self.env.n - self.agent_idx - 1, self.action_dim])
        action_logits = tf.expand_dims(self.mu_noise, axis=1)
        actions_n = tf.concat([slice_1, action_logits, slice_2], axis=1)
        input = tf.concat([tf.layers.flatten(self.state_placeholder), tf.layers.flatten(actions_n)],
                          axis=1)

        combined_q_scope = self.critic_network_scope + "/" + self.q_scope
        self.q_reuse = build_mlp(input, 1, combined_q_scope, self.config.n_layers, self.config.layer_size)

        # self.q_copy_scope = "q_copy"
        # with tf.variable_scope(self.actor_network_scope):
        #     self.q_copy = build_mlp(input, 1, self.q_copy_scope, self.config.n_layers, self.config.layer_size)


    def add_optimizer_op(self):
        """

            :return: None
        """
        # combined_q_scope = self.agent_scope + "/" + self.critic_network_scope + "/" + self.q_scope
        # q_copy_scope = self.agent_scope + "/" + self.actor_network_scope + "/" + self.q_copy_scope
        # copy_q_ops = self.get_copy_ops(combined_q_scope, q_copy_scope)
        # self.copy_q_op = tf.group(*copy_q_ops)

        optimizer = tf.train.AdamOptimizer(self.lr)
        combined_scope = self.agent_scope + "/" + self.actor_network_scope + "/" + self.mu_scope
        self.mu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, combined_scope)
        # if self.config.debug_logging: self.q_copy = tf.Print(self.q_copy, [self.q_copy], message="q copy")

        self.objective = -tf.reduce_mean(self.q_reuse)
        # self.objective = -tf.reduce_mean(self.q_copy)
        if self.config.debug_logging: self.objective = tf.Print(self.objective, [self.objective], message="policy network objective: ")
        self.policy_network_objective = self.objective
        grads_vars = optimizer.compute_gradients(self.objective, self.mu_vars)
        policy_network_grad_norm = tf.global_norm([grad for (grad, _) in grads_vars])
        self.policy_network_grad_norm = tf.Print(policy_network_grad_norm, [policy_network_grad_norm], message="policy network grad norm: ")
        if self.config.grad_clip:
            variables = [v for g,v in grads_vars]
            clipped = []
            if self.config.debug_logging:
                for g, _ in grads_vars:
                    g = tf.clip_by_norm(g, self.config.clip_val)
                    g = tf.Print(g, [tf.reduce_max(g), tf.shape(g)], message="policy gradient")
                    clipped += [g]
            else:
                clipped = [tf.clip_by_norm(g, self.config.clip_val) for g,v in grads_vars]
            grads_vars = zip(clipped, variables)

        self.train_actor_op = optimizer.apply_gradients(grads_vars)

        ### update target networks
    def get_copy_ops(self, scope, target_scope):
        op_list = list()

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_scope)
        for idx, var in enumerate(vars):
            target_var = target_vars[idx]
            assign_op = tf.assign(target_var, var)
            op_list.append(assign_op)
        return op_list

    def get_assign_ops(self, scope, target_scope):
        op_list = list()

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        vars = sorted(vars, key=lambda var: var.name)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_scope)
        target_vars = sorted(target_vars, key=lambda var: var.name)
        for idx, var in enumerate(vars):
            target_var = target_vars[idx]
            new_target_var = self.config.tau * var + (1.0 - self.config.tau) * target_var
            assign_op = tf.assign(target_var, new_target_var)
            op_list.append(assign_op)

        return op_list

    def add_update_target_op(self):
        """
        update_target_op will be called periodically
        to update target network weights based on the constantly-changing network
        target_theta <- tau * theta + (1-tau) * target_theta

        """
        combined_q_scope = self.agent_scope + "/" + self.critic_network_scope + "/" + self.q_scope
        combined_target_q_scope = self.agent_scope + "/" + self.critic_network_scope + "/" + self.target_q_scope
        combined_mu_scope = self.agent_scope + "/" + self.actor_network_scope + "/" + self.mu_scope
        combined_target_mu_scope = self.agent_scope + "/" + self.actor_network_scope + "/" + self.target_mu_scope
        op_list = self.get_assign_ops(combined_q_scope, combined_target_q_scope) + self.get_assign_ops(combined_mu_scope, combined_target_mu_scope)

        self.update_target_op = tf.group(*op_list)


    def build(self, agent_scope):
        """
        Build model by adding all necessary variables

        You don't have to change anything here - we are just calling
        all the operations you already defined to build the tensorflow graph.
        """
        self.agent_scope = agent_scope # top level scope for this agent, will be needed when getting variables

        # add shared placeholders
        self.add_placeholders_op()
        # create actor approx nets
        self.build_policy_approx_networks()
        self.add_update_policy_approx_networks_op()
        # create critic net
        self.add_critic_network_placeholders_op()
        self.add_critic_network_op()
        self.add_update_critic_network_op()
        # create actor net
        self.build_policy_network_op()
        self.add_actor_loss_op()
        self.add_optimizer_op() # depends on self.q

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

        #self.add_summary()


    #################### Running the model ######################
    def update_policy_approx_networks(self, observations_by_agent, actions_by_agent):
        """

        :param observations_by_agent: shape (num_agent, batch_size, observation_dim)
        :param actions_by_agent: shape (num_agent, batch_size, action_dim)
        :return:
        """
        for i in range(self.env.n):
            if i == self.agent_idx: continue
            obs = observations_by_agent[i]
            act = actions_by_agent[i]
            update_approx_network = self.update_policy_approx_networks_op[i]
            loss = self.policy_approx_networks_losses[i]
            grad_norm = self.policy_approx_grad_norms[i]
            ops_to_run = [update_approx_network]
            if self.config.debug_logging: ops_to_run += [loss, grad_norm]
            self.sess.run(ops_to_run,
                          feed_dict={self.action_placeholder : act,
                                self.observation_placeholder : obs})

    def update_critic_network(self, state, true_actions, observations_by_agent, next_state, next_observations_by_agent, rewards, done_mask, agents_list=None):
        """
        Update the critic network
        Args:
            samples: a tuple (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask)
                    obs_batch: np.array of shape (None, num_agent, observation_dim)
        TODO: we might want more granular input args than samples, to avoid duplicate numpy operations
        """
        if self.config.debug_logging:
            print("state: ")
            print(state)

        # get the next estimated action from each agent approx network given next observation
        # Specifically, for the current agent, get the next action from the target policy network
        # for all other agents, get the action from the approx network
        est_next_actions_by_agent = []
        for i in range(self.env.n):
            next_observations_i = next_observations_by_agent[i]
            next_actions_i = None
            if i == self.agent_idx:
                next_actions_i = self.sess.run(self.target_mu_noise, feed_dict={self.observation_placeholder: next_observations_i})
            else:
                if not self.config.use_true_actions: # use approximate policy networks instead of the true action another agent would take
                  next_actions_i = self.sess.run(self.policy_approximates_log_probs[i],
                                               feed_dict={self.observation_placeholder: next_observations_i})
                else:
                  # NV TODO: Should we take the mean, or should we use get_sampled_action() instead
                  other = agents_list[i]
                  next_actions_i = other.sess.run(other.target_mu_noise, feed_dict={other.observation_placeholder: next_observations_i})
                  
            est_next_actions_by_agent.append(next_actions_i)
        est_next_actions = np.swapaxes(est_next_actions_by_agent, 0, 1)  # shape = (None, num_agent, action_dim)
            
        q_next = self.sess.run(self.target_q, feed_dict={self.state_placeholder : next_state,
                                                         self.actions_n_placeholder : est_next_actions})
        q_next = np.squeeze(q_next, axis=1)

        # Then get an estimated action from each agent approx network given current observation
        # Specifically, for the current agent, get the action from the policy network
        # for all other agents, get the action from the approx network
        # est_actions_by_agent = []
        # for i in range(self.env.n):
        #     observations_i = observations_by_agent[i]
        #     actions_i = None
        #     if i == self.agent_idx:
        #         actions_i = self.sess.run(self.mu_noise, feed_dict={self.observation_placeholder: observations_i})
        #     else:
        #         actions_i = self.sess.run(self.policy_approximates_log_probs[i],
        #                                   feed_dict={self.observation_placeholder: observations_i})
        #     est_actions_by_agent.append(actions_i)
        # est_actions = np.swapaxes(est_actions_by_agent, 0, 1)  # shape = (None, num_agent, action_dim)

        ops_to_run = [self.update_critic_op]
        if self.config.use_batch_normalization:
            scope = self.agent_scope + "/" + self.critic_network_scope
            ops_to_run += [tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)]
        if self.config.debug_logging: ops_to_run += [self.critic_network_loss, self.critic_network_grad_norm]
        self.sess.run(ops_to_run, feed_dict={self.state_placeholder : state,
                                                        self.actions_n_placeholder: true_actions,
                                                        self.q_next_placeholder : q_next,
                                                        self.reward_placeholder : rewards,
                                                        self.done_mask_placeholder : done_mask})



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

        # self.sess.run(self.copy_q_op, feed_dict={})

        ops_to_run = [self.train_actor_op]
        if self.config.use_batch_normalization:
            scope = self.agent_scope + "/" + self.actor_network_scope
            ops_to_run += [tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)]
        if self.config.debug_logging: ops_to_run += [self.objective, self.policy_network_grad_norm]
        self.sess.run(ops_to_run, feed_dict={self.state_placeholder : state,
                                                     self.actions_n_placeholder : actions_n,
                                                     self.observation_placeholder : observation})



    def get_action_and_logits(self, observations):
        """
        Run
        :param observations: batched observations
        :return: action: shape=(None, action_dim). if discrete, a single action is one-hot vector
                 action_logits: shape=(None, action_dim)
                                for discrete case, this is direct output of mu
                                for none-discrete case, this is the same as action
        """
        action_logits = self.sess.run(self.mu_noise, feed_dict={self.observation_placeholder: observations})
        actions = action_logits
        if self.config.discrete:
            action_indices = tf.argmax(action_logits, axis=1)
            actions = tf.one_hot(action_indices, self.action_dim)
        return actions, action_logits     
    
    def get_sampled_action(self, observation, is_evaluation=False):
        """
        Get a single action for a single observation, used for stepping the environment

        :param observation: single observation to run self.sampled_action with
        :return: action: if discrete, output is a single number
                         otherwise, shape=(action_dim)
        """
        batch = np.expand_dims(observation, 0)
        action, logits = None, None

        if self.config.random_process_exploration == 2 and not is_evaluation: # dist sampling
            logits_batched, action_batched = self.sess.run([self.mu_noise, self.sample_action_op], feed_dict={self.observation_placeholder: batch})
            logits = logits_batched[0]
            #action = action_batched[0]
            action = logits
        else:
            actions, action_logits = self.get_action_and_logits(batch)
            action = actions[0]
            if self.config.random_process_exploration == 1 and not is_evaluation: # ornstein-uhlenbeck
                action = action + self.noise()    
            if is_evaluation:
                return action
            logits = action_logits[0]

        # self.logger.info("action logits: ")
        # self.logger.info(logits)
        # self.logger.info("sampled action: ")
        # self.logger.info(action)
        return action


    def train_for_batch_samples(self, samples, agents_list=None):
        """
            Train for a batch of samples

                Args:
                  samples: a tuple (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask)
                        obs_batch: np.array of shape (None, num_agent, observation_dim)
        """
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.agent_scope)


        state, true_actions, rewards, next_state, done_mask = samples
        # update this agent's policy approx networks for other agents
        observations_by_agent = np.swapaxes(state, 0, 1) # shape (num_agent, batch_size, observation_dim)
        true_actions_by_agent = np.swapaxes(true_actions, 0, 1)
        rewards_by_agent = np.swapaxes(rewards, 0, 1)
        next_observations_by_agent = np.swapaxes(next_state, 0, 1)
        observation_for_current_agent = observations_by_agent[self.agent_idx]
        reward_for_current_agent = rewards_by_agent[self.agent_idx]
        self.update_policy_approx_networks(observations_by_agent, true_actions_by_agent)

        # update centralized Q network
        self.update_critic_network(state, true_actions, observations_by_agent, next_state, next_observations_by_agent, reward_for_current_agent, done_mask, agents_list)

        # update actor network
        self.update_actor_network(observation_for_current_agent, true_actions, state)

        # update target networks
        self.sess.run(self.update_target_op, feed_dict={})

        #self.record_summary(self.t)
        self.t += 1


    ############### summary and logging ###############

    def record_summary(self, t):
        """
            Add summary to tfboard

            You don't have to change or use anything here.
            """
        policy_approx_networks_loss = self.sess.run(self.policy_approx_networks_loss)
        fd = {
            self.policy_approx_networks_loss_placeholder: policy_approx_networks_loss,
            self.policy_network_objective_placeholder: self.policy_network_objective,
            self.critic_network_loss_placeholder: self.critic_network_loss,
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
        self.policy_approx_networks_loss_placeholder = tf.placeholder(tf.float32, shape=(), name="policy_approx_networks_loss")
        self.policy_network_objective_placeholder = tf.placeholder(tf.float32, shape=(), name="policy_network_objective")
        self.critic_network_loss_placeholder = tf.placeholder(tf.float32, shape=(), name="critic_network_loss")

        #self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # extra summaries from python -> placeholders
        tf.summary.scalar("policy approx networks loss", self.policy_approx_networks_loss_placeholder)
        tf.summary.scalar("policy network objective", self.policy_network_objective_placeholder)
        tf.summary.scalar("critic network loss", self.critic_network_loss_placeholder)
        #tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path+"agent"+str(self.agent_idx)+"/", self.sess.graph)
