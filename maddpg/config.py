import tensorflow as tf

class config():
    # https://github.com/openai/maddpg
    env_name = "simple_spread"
    algo_name = "MADDPG"
    record           = False
    render = False # True

    # output config
    output_path  = "results/" + env_name + "/" + algo_name + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1

    
    # model and training config
    discrete = False # if True, we use a single number to represent an action; else we use a vector of length action_dim
    num_batches = 600 # number of batches trained on
    batch_size_in_episodes = 40 # number of samples used to compute each policy update
    max_ep_len = 25 # maximum episode length
    batch_size = batch_size_in_episodes * max_ep_len
    train_freq = 100 # do a training step after every train_freq samples added to replay buffer
    eval_freq = 25 # every eval_freq batches, evaluate
    learning_rate = 0.01
    gamma              = .95 # the discount factor
    policy_approx_lambda = .001
    tau = 0.01

    use_baseline = True 
    normalize_advantage=True
    replay_buffer_size = 1000000
    # parameters for the policy and baseline models
    n_layers = 2
    layer_size = 128
    activation=tf.nn.relu 
    
    # added configs
    use_true_actions = True
    random_process_exploration = 2  # 0 = no exploration, 1 = ornstein uhlenbeck 2 = sampling from dist
    sampling_std = .5 
    grad_clip = True # if true, clip the gradient using clip_val
    clip_val = .5
    
    action_clip = True #if true, clip actions taken to be between -2 and 2 (used in maddpg.sample_n)


    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
