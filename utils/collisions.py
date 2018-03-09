import numpy as np

def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False

def count_agent_collisions(env):
    """
    Returns a count of all collisions between agents.
    """
    count = 0
    for i in range(env.n):
        for j in range(env.n):
            if (j > i): # so we don't double-count
              if is_collision(env.agents[i], env.agents[j]):
                count += 1
    return count