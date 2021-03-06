import tensorflow as tf

class config():

    env_name = "simple_spread"
    algo_name = "PG"
    record    = False
    render = False 

    # output config
    output_path  = "results/" + env_name + "/" + algo_name + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1

    eval_freq = 250  # every eval_freq batches, evaluate
    eval_episodes = 40  # number of episodes to do an evaluation run on

    
    # model and training config
    num_batches = 60000 # number of batches trained on
    # batch_size = 1024 # unused
    batch_size_in_episodes = 50 # number of episodes used to compute each policy update
    max_ep_len = 25 # maximum episode length
    learning_rate = 1e-2
    gamma         = .95 # the discount factor
    use_baseline = True 
    normalize_advantage=True 
    # parameters for the policy and baseline models
    n_layers = 2
    layer_size = 128
