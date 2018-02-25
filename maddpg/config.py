import tensorflow as tf

class config():
    env_name = "simple_spread"
    algo_name = "MADDPG"
    record           = False

    # output config
    output_path  = "results/" + env_name + "/" + algo_name + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1

    
    # model and training config
    num_batches = 200 # number of batches trained on 
    batch_size = 1000 # number of samples used to compute each policy update
    max_ep_len = 1000 # maximum episode length
    train_freq = 100 # do a training step after every train_freq samples added to replay buffer
    learning_rate = 3e-2
    gamma              = 1.0 # the discount factor
    use_baseline = True 
    normalize_advantage=True
    replay_buffer_size = 1000000
    # parameters for the policy and baseline models
    n_layers = 1 
    layer_size = 16 
    activation=tf.nn.relu 


    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
