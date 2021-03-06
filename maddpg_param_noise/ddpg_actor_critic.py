import os
import math

import gym
import numpy as np
import tensorflow as tf
from config import config
from network import build_mlp, entropy
from AdaptiveParamNoiseSpec import AdaptiveParamNoiseSpec
from utils.general import get_logger, export_plot
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

        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.6, desired_action_stddev=0.2)

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
        policy_approximate_logits = []
        policy_approximate_actions = []
        with tf.variable_scope(self.policy_approx_networks_scope):
            for i in range(self.env.n):
                if i == self.agent_idx:
                    policy_approximate_logits.append(None)
                    policy_approximate_actions.append(None)
                    continue
                scope = "agent_" + str(i)
                logits = build_mlp(self.observation_placeholder, self.action_dim, scope, self.config.n_layers,
                                   self.config.layer_size, output_activation=None)
                policy_approximate_logits.append(logits)
                policy_approximate_actions.append(tf.nn.softmax(logits, axis=-1))

        self.policy_approximate_logits = policy_approximate_logits
        self.policy_approximate_actions = policy_approximate_actions


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

            approx_logits = self.policy_approximate_logits[i]
            action_labels = self.action_placeholder
            if self.config.random_process_exploration > 0:
                # needed for o-u and dist. sampling because they may not produce a valid prob. dist. for the action
                action_labels = tf.nn.softmax(self.action_placeholder)

            logprob = -tf.nn.softmax_cross_entropy_with_logits(labels=action_labels, logits=approx_logits)
            dist = tf.contrib.distributions.Categorical(logits=approx_logits)
            logits_entropy = dist.entropy()
            loss = -tf.reduce_mean(logprob + self.config.policy_approx_lambda * logits_entropy)
            self.policy_approx_networks_losses[i] = tf.Print(loss, [loss], message="agent_"+str(i)+" loss:")
            optimizer = tf.train.AdamOptimizer(self.lr)
            combined_scope = self.agent_scope + "/" + self.policy_approx_networks_scope + "/" + "agent_" + str(i)
            vars_in_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, combined_scope)
            grads_vars = optimizer.compute_gradients(loss, var_list=vars_in_scope)
            grad_norm = tf.global_norm([grad for (grad, _) in grads_vars])
            self.policy_approx_grad_norms[i] = tf.Print(grad_norm, [grad_norm], message="agent_"+str(i)+" grad norm:")
            if self.config.grad_clip:
                variables = [v for g, v in grads_vars]
                clipped = [tf.clip_by_norm(g, self.config.clip_val) for g, v in grads_vars]
                grads_vars = zip(clipped, variables)
            update = optimizer.apply_gradients(grads_vars)

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
            self.mu = build_mlp(self.observation_placeholder, self.action_dim, self.mu_scope,
                                n_layers=self.config.n_layers, size=self.config.layer_size,
                                output_activation=None, use_batch_normalization=self.config.use_batch_normalization)
            if self.config.debug_logging: self.mu = tf.Print(self.mu, [self.mu], message="mu", summarize=20)
            self.target_mu = build_mlp(self.observation_placeholder, self.action_dim, self.target_mu_scope,
                                       n_layers=self.config.n_layers, size=self.config.layer_size,
                                       output_activation=None,
                                       use_batch_normalization=self.config.use_batch_normalization)

            self.mu_normalized = tf.nn.softmax(self.mu, axis=-1)
            self.target_mu_normalized = tf.nn.softmax(self.target_mu, axis=-1)

            if self.config.param_noise:
                self.setup_param_noise(self.observation_placeholder)
                self.mu_noise = tf.nn.softmax(self.mu - tf.log(-tf.log(tf.random_uniform(tf.shape(self.mu)))), axis=-1)
                if self.config.debug_logging: self.mu_noise = tf.Print(self.mu_noise, [self.mu_noise], summarize=10,
                                                                   message="action logits")
                self.target_mu_noise = tf.nn.softmax(self.target_mu - tf.log(-tf.log(tf.random_uniform(tf.shape(self.target_mu)))), axis=-1)

            elif self.config.random_process_exploration == 0:
                self.mu_noise = tf.nn.softmax(self.mu - tf.log(-tf.log(tf.random_uniform(tf.shape(self.mu)))), axis=-1)
                if self.config.debug_logging: self.mu_noise = tf.Print(self.mu_noise, [self.mu_noise], summarize=10,
                                                                   message="action logits")
                self.target_mu_noise = tf.nn.softmax(self.target_mu - tf.log(-tf.log(tf.random_uniform(tf.shape(self.target_mu)))), axis=-1)
            elif self.config.random_process_exploration == 1:
                self.mu_noise = self.mu_normalized
                self.target_mu_noise = self.target_mu_normalized
            elif self.config.random_process_exploration == 2:
                log_std = tf.get_variable("random_process_log_std", shape=[self.action_dim], dtype=tf.float32)
                std = tf.exp(log_std)
                dist = tf.contrib.distributions.MultivariateNormalDiag(self.mu_normalized, std)
                self.mu_noise = dist.sample()
                self.target_mu_noise = self.target_mu_normalized


    def add_actor_loss_op(self):
        slice_1 = tf.slice(self.actions_n_placeholder, [0, 0, 0], [self.config.batch_size, self.agent_idx, self.action_dim])
        slice_2 = tf.slice(self.actions_n_placeholder, [0, self.agent_idx+1, 0], [self.config.batch_size, self.env.n - self.agent_idx - 1, self.action_dim])
        action_logits = tf.expand_dims(self.mu_noise, axis=1)
        actions_n = tf.concat([slice_1, action_logits, slice_2], axis=1)
        input = tf.concat([tf.layers.flatten(self.state_placeholder), tf.layers.flatten(actions_n)],
                          axis=1)

        combined_q_scope = self.critic_network_scope + "/" + self.q_scope
        self.q_reuse = build_mlp(input, 1, combined_q_scope, self.config.n_layers, self.config.layer_size)


    def add_optimizer_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        combined_scope = self.agent_scope + "/" + self.actor_network_scope + "/" + self.mu_scope
        self.mu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, combined_scope)

        self.objective = -tf.reduce_mean(self.q_reuse)
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

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = tf.identity(self.mu) 
        #param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor
        self.logger.info('setting up param noise')
        self.perturb_policy_ops = self.get_perturbed_actor_updates(self.mu, param_noise_actor, self.param_noise_stddev)
 
        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = tf.identity(self.mu) 
        #adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_mu = adaptive_param_noise_actor 
        self.perturb_adaptive_policy_ops = self.get_perturbed_actor_updates(self.mu, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.mu - adaptive_mu)))

    def adapt_param_noise(self, batch):
        if self.config.param_noise is None:
            return 0.

        # agent_batch = batch[:][self.agent_idx+1][:]
        state, true_actions, rewards, next_state, done_mask = batch
        observations_by_agent = np.swapaxes(state, 0, 1) # shape (num_agent, batch_size, observation_dim)
        observations_by_current_agent = observations_by_agent[self.agent_idx] 

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
                                     self.param_noise_stddev: self.param_noise.current_stddev,
                                     })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
                                                       self.observation_placeholder: observations_by_current_agent,
                                                       self.param_noise_stddev: self.param_noise.current_stddev,
                                                        })

        mean_distance = np.mean(distance)
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def get_perturbed_actor_updates(self, actor, perturbed_actor, param_noise_stddev):
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor.name)
        perturbed_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=perturbed_actor.name)
        actor_perturbable_vars = self.perturbable_vars(actor)
        perturbed_actor_perturbable_vars = self.perturbable_vars(perturbed_actor)
        assert len(actor_vars) == len(perturbed_actor_vars)
        assert len(actor_perturbable_vars) == len(perturbed_actor_perturbable_vars)

        updates = []
        for var, perturbed_var in zip(actor_vars, perturbed_actor_vars):
            if var in actor_perturbable_vars:
                #self.logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
                updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
            else:
                #self.logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
                updates.append(tf.assign(perturbed_var, var))
        assert len(updates) == len(actor_vars)
        return tf.group(*updates)

    def perturbable_vars(self, model):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.name)
        return [var for var in trainable_vars if 'LayerNorm' not in var.name]

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
            if self.config.debug_logging or self.config.approx_debugging: ops_to_run += [loss, grad_norm]
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
                    next_actions_i = self.sess.run(self.policy_approximate_actions[i],
                                               feed_dict={self.observation_placeholder: next_observations_i})
                    # if self.config.approx_debugging:
                    #     other = agents_list[i]
                    #     true_next = other.sess.run(other.target_mu_noise,
                    #                      feed_dict={other.observation_placeholder: next_observations_i})
                    #     print("prediction:")
                    #     print(next_actions_i)
                    #     print("real")
                    #     print(true_next)
                else:
                    # NV TODO: Should we take the mean, or should we use get_sampled_action() instead
                    other = agents_list[i]
                    next_actions_i = other.sess.run(other.target_mu_noise, feed_dict={other.observation_placeholder: next_observations_i})
                  
            est_next_actions_by_agent.append(next_actions_i)
        est_next_actions = np.swapaxes(est_next_actions_by_agent, 0, 1)  # shape = (None, num_agent, action_dim)
            
        q_next = self.sess.run(self.target_q, feed_dict={self.state_placeholder : next_state,
                                                         self.actions_n_placeholder : est_next_actions})
        q_next = np.squeeze(q_next, axis=1)


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



    def get_sampled_action(self, observation, is_evaluation=False):
        """
        Get a single action for a single observation, used for stepping the environment

        :param observation: single observation to run self.sampled_action with
        :return: action: if discrete, output is a single number
                         otherwise, shape=(action_dim)
        print('obs shape - ',tf.shape(observation))
        """
        if self.config.normalize_obs:
            observation = np.clip(observation, -5.0, 5.0) # observation range from -5.0 to 5.0
            print('normalized obs shape - ',tf.shape(observation))

        batch = np.expand_dims(observation, 0)
        action = None

        if is_evaluation:
            action_batched = None
            if self.config.run_evaluation_with_noise:
                action_batched = self.sess.run(self.mu_noise, feed_dict={self.observation_placeholder: batch})
            else:
                action_batched = self.sess.run(self.mu_normalized, feed_dict={self.observation_placeholder: batch})
            action = action_batched[0]
            return action


        action_batched = self.sess.run(self.mu_noise, feed_dict={self.observation_placeholder: batch})
        action = action_batched[0]
        if not self.config.param_noise and self.config.random_process_exploration == 1: # ornstein-uhlenbeck
            action = action + self.noise()

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
