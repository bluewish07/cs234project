# Multi-Agent Reinforcement Learning for Cooperative Navigation

## Contributors
Diana Huang, Shalini Keshavamurthy, and Nitin Viswanathan

## Overview
This repository contains the code used in our CS234 Winter 2018 class project, where we implemented both MADDPG and policy gradient approaches to train agents in a cooperative navigation scenario.

## Code structure

- `pg/config.py, pg/pg.py`: contains code for the configuration and implementation of running multiple independently learning agents using a single-agent policy gradient approach

- `maddpg/config.py, maddpg/maddpg.py, maddpg/ddpg_actor_critic.py`: contains code for the configuration and implementation of running multiple agents learning using MADDPG

- `.main.py`: Modify this to determine which algorithm you run (vanilla PG vs. MADDPG)

- `./utils/`: various utility functions we wrote to run the multi-agent scenarios
    
## Getting started:
- Known dependencies: tensorflow, OpenAI gym, numpy, OpenAI multi-agent-envs(https://github.com/openai/multiagent-particle-envs)

- ensure OpenAI multi-agent-envs is cloned to the root directory of this project

- simply run `python main.py` to begin running with our default settings.

- if you want to change the algorithm between PG and MADDPG, edit main.py

- if you want to change experiment settings, edit the appropriate config.py file
