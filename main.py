import os
import sys
from pg.pg import PG
import make_env

if __name__ == '__main__':
    env_name = "simple_spread"

    env = make_env(env_name)
    # train model
    model = PG(env)
    model.run()