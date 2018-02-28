import os
import sys
from pg.multi_agent_pg import MultiAgentPG
from pg.config import config
from make_env import  make_env
import numpy as np
import numpy.random as random
import tensorflow as tf

seed = 234

if __name__ == '__main__':
  env = make_env(config.env_name)
  
  env.seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)
  
  # train model
  model = MultiAgentPG(env)

  model.run()