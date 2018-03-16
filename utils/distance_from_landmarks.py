import numpy as np

def get_distance_from_landmarks(env):
  world = env.world
  dists = []
  for a in world.agents:
    dist = []
    for l in world.landmarks:
      dist.append( np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) )
    dists.append(np.min(dist))  
  return np.mean(dists)

