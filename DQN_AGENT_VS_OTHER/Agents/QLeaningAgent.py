import numpy as np
from recsim.agent import AbstractEpisodicRecommenderAgent
from collections import defaultdict
import copy
from enum import Enum
from MusicEnv.Documents import MusicDocument

class QLearningType(Enum):
    SIMPLE = 1
    MID = 2
    COMPLEX = 3


class QLearningAgent(AbstractEpisodicRecommenderAgent):
    class ClusterSet:
        def __init__(self, clusters):
            self.clusters = sorted(set(clusters))

        def __eq__(self, other):
            if not isinstance(other, QLearningAgent.ClusterSet):
                return False
            return self.clusters == other.clusters

        def __hash__(self):
            return hash(tuple(self.clusters))

        def __repr__(self):
            return f"ClusterSet({self.clusters})"

    def __init__(self, observation_space, action_space, type:QLearningType = QLearningType.SIMPLE, learning_rate=0.1, discount_factor=0.8, exploration_rate=0.4):
        if len(observation_space['doc'].spaces) < len(action_space.nvec):
            raise RuntimeError('Slate size larger than size of the corpus.')
        self.slate_size = len(action_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self._type = type

        self.q_values = defaultdict(lambda: defaultdict(float))
        
        # Test 2:
        self.genre_bins = np.linspace(0, 1, 8)
        self.production_bins = np.linspace(0, 1, 8)
        self.quality_bins = np.linspace(0, 1, 8)
        self.tempo_bins = np.arange(MusicDocument.MIN_TEMPO, MusicDocument.MAX_TEMPO + 1, 10)  # Tempo: 40-200 co 20 BPM
        self.quality_tolerance_bins = np.linspace(0, 1, 8)
            



        super(QLearningAgent, self).__init__(action_space)

    def _discretize_state(self, observation):
        """Discretize the state features into bins."""
        if self._type == QLearningType.SIMPLE:
            return (
                np.argmax(observation['genre_preferences'])
                )
        elif self._type == QLearningType.MID:
            return(
                np.argmax(observation['genre_preferences'])
                ,np.digitize(observation['preferred_tempo'], self.tempo_bins)
            )
        else:
            return (
                tuple(np.digitize(observation['genre_preferences'], self.genre_bins))
                ,np.argmax(observation['genre_preferences'])
                ,np.digitize(observation['preferred_tempo'], self.tempo_bins)
                ,np.digitize(observation['quality_tolerance'], self.quality_tolerance_bins)
                )
    
    def _discretize_action(self, document):
        if self._type == QLearningType.SIMPLE:
            return(
                np.digitize(document['quality'], self.quality_bins)
            )
        elif self._type == QLearningType.MID:
            return(
                np.digitize(document['quality'], self.quality_bins)
                ,np.digitize(document['production_value'], self.production_bins)
                ,document['cluster_id']
            )
        return (
            np.digitize(document['production_value'], self.production_bins)
            ,np.digitize(document['quality'], self.quality_bins)
            ,np.digitize(document['tempo'], self.tempo_bins)
            ,document['cluster_id']
        )

    def step(self, reward, observation):
        current_state = self._discretize_state(observation['user'])
        
        if reward is not None and observation['response'] is not None:
            self._update_q_values(reward, current_state, observation)

        documents = observation['doc']
        doc_ids = list(documents.keys())
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        if np.random.rand() < self.exploration_rate:
            slate = np.random.choice(doc_ids, self.slate_size, replace=False).tolist()
        else:
            possible_actions = {doc : self._discretize_action(documents[doc]) for doc in documents}
            sorted_actions = sorted(
                possible_actions.items(),
                key=lambda item: self.q_values[current_state][item[1]],
                reverse=True
            )

            slate = [item[0] for item in sorted_actions[:self.slate_size]]

        
        slate_indices = [doc_id_to_index[doc_id] for doc_id in slate]
        return list(map(int, slate_indices))

    def _update_q_values(self, reward, current_state, observation):
        """Update Q-values based on reward and next state."""

        response = observation['response']
        prev_state = self._discretize_state(response[0])
        
        next_max_q = max([self.q_values[current_state][self._discretize_action(observation['doc'][doc])] for doc in observation['doc']], default=0)
        
        prev_action = {self._discretize_action(res) : res['user_score'] if res['click'] else 0 for res in response}

        x = 2
        for p_act in prev_action.keys():
            old_q = self.q_values[prev_state][p_act]
            new_val = (1 - self.learning_rate) * old_q + self.learning_rate * (prev_action[p_act] + self.discount_factor * next_max_q)
            self.q_values[prev_state][p_act] = new_val