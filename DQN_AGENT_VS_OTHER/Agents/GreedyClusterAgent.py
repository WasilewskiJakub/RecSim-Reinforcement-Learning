import numpy as np
from recsim import agent

class GreedyClusterAgent(agent.AbstractEpisodicRecommenderAgent):
  """Simple agent sorting all documents of a topic according to quality."""

  def __init__(self, observation_space, action_space):
    self.n = len(action_space)
    super(GreedyClusterAgent, self).__init__(action_space)
  
  def step(self, reward, observation):
    del reward
    my_docs = []
    my_doc_scores = []
    favourite_genre = np.argmax(observation['user']['genre_preferences'])
     
    for i, doc in enumerate(observation['doc'].values()):

      quality = doc['quality']

      cluster_id = doc['cluster_id']
      
      score = quality + (1 if cluster_id == favourite_genre else 0)
      my_docs.append(i)
      my_doc_scores.append(score)

    if not bool(my_docs):
      return []
    sorted_indices = np.argsort(my_doc_scores)[::-1]

    return list(np.array(my_docs)[sorted_indices])[0:self.n]