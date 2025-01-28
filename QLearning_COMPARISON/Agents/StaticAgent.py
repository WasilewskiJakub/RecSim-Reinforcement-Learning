from recsim import agent
from recsim.agent import AbstractEpisodicRecommenderAgent

class StaticAgent(AbstractEpisodicRecommenderAgent):
  def __init__(self, observation_space, action_space):
    if len(observation_space['doc'].spaces) < len(action_space.nvec):
      raise RuntimeError('Slate size larger than size of the corpus.')
    super(StaticAgent, self).__init__(action_space)

  def step(self, reward, observation):
    return list(range(self._slate_size))
  
def create_agent_static(sess, environment, eval_mode, summary_writer=None):
  return StaticAgent(environment.observation_space, environment.action_space)
