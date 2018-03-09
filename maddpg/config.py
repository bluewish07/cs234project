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
    num_batches = 60 # number of batches trained on
    batch_size_in_episodes = 1024 # number of samples used to compute each policy update
    max_ep_len = 25 # maximum episode length
    batch_size = batch_size_in_episodes * max_ep_len
    train_freq = 100 # do a training step after every train_freq samples added to replay buffer
    learning_rate = 0.01
    gamma              = .95 # the discount factor
    policy_approx_lambda = .001
    tau = 0.01
    random_process_exploration = True
    use_baseline = True 
    normalize_advantage=True
    replay_buffer_size = 1000000
    # parameters for the policy and baseline models
    n_layers = 2
    layer_size = 128
    activation=tf.nn.relu 


    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
