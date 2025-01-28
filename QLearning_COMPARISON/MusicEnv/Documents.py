from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import spaces
import numpy as np

from recsim.environments.interest_exploration import(
    IEDocument,
    IETopicDocumentSampler
)

# 1. Wesoły
# 2. Smutny
# 3. Energetyczny
# 4. Romantyczny
# 5. Spokojny
# 6. Motywujący
# 7. Zrelaksowany
# 8. Mroczny
# 9. Epicki
# 10. Nostalgiczny
# 11. Zadziorny
# 12. Mistyczny



class MusicDocument(IEDocument):

    MOOD_COUNT = 12
    """Class representing a music document in the environment."""

    def __init__(self, doc_id, cluster_id, quality, discovery_potential, virality_factor,
                 tempo, mood_intensity, production_value, fatigue_factor, duration):
        """
        Initialize a music document.

        Args:
            doc_id (int): Unique identifier for the document.
            genre_id (int): The genre cluster the document belongs to.
            quality (float): Artistic quality of the music (0-1).
            discovery_potential (float): Novelty or uniqueness of the document (0-1).
            virality_factor (float): Likelihood of becoming popular (0-1).
            tempo (float): Tempo of the music in beats per minute (e.g., 60-200).
            mood_intensity (float): Emotional intensity of the music (0-1).
            production_value (float): Technical quality of production (0-1).
            fatigue_factor (float): Long-term fatigue factor of the document (0-1).
        """
        
        
        super(MusicDocument, self).__init__(doc_id, cluster_id = cluster_id, quality=quality)
        self.discovery_potential = discovery_potential
        self.virality_factor = virality_factor
        self.tempo = tempo
        self.mood_intensity = mood_intensity
        self.production_value = production_value
        self.fatigue_factor = fatigue_factor
        self.duration = duration

    def create_observation(self):
        base_observation = super(MusicDocument, self).create_observation()

        base_observation.update({
            'discovery_potential': np.array(self.discovery_potential),
            'virality_factor': np.array(self.virality_factor),
            'tempo': np.array(self.tempo),
            'mood_intensity': np.array(self.mood_intensity),
            'production_value': np.array(self.production_value),
            'duration': np.array(self.duration)
        })
        return base_observation

    @classmethod
    def observation_space(cls):
        combined_space = spaces.Dict({
            'cluster_id': spaces.Discrete(IEDocument.NUM_CLUSTERS)
            ,'quality': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'discovery_potential': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'virality_factor': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'tempo': spaces.Box(low=40.0, high=200.0, shape=tuple(), dtype=np.float32)
            ,'mood_intensity': spaces.Box(low=0.0, high=1.0, shape=(MusicDocument.MOOD_COUNT,), dtype=np.float32)
            ,'production_value': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'duration': spaces.Box(low=2.2, high=7.30, shape=tuple(), dtype=np.float32)
        })
        return combined_space
    
class MusicDocumentSampler(IETopicDocumentSampler):
    """Sampler for generating music documents."""

    def __init__(self
                ,topic_distribution=(.2, .8)
                ,topic_quality_mean=(.8, .2)
                ,topic_quality_stddev=(.1, .1)
                ,doc_ctor = MusicDocument
                ,**kwargs):
        """Initialize the document sampler."""

        super(MusicDocumentSampler, self).__init__(
            topic_distribution=topic_distribution,
            topic_quality_mean=topic_quality_mean,
            topic_quality_stddev=topic_quality_stddev,
            doc_ctor = doc_ctor,
            **kwargs
        )

    def sample_document(self):
        """Generates a new music document with random attributes."""
        doc_features = {}
        doc_features['doc_id'] = self._doc_count
        self._doc_count += 1

        # Sampling genre/cluster
        doc_features['cluster_id'] = self._rng.choice(self._number_of_topics)
        doc_features['quality'] = self._rng.uniform(0.0, 1.0)

        doc_features['discovery_potential'] = self._rng.uniform(0.0, 1.0)
        doc_features['virality_factor'] = self._rng.uniform(0.0, 1.0)
        doc_features['tempo'] = self._rng.uniform(40.0, 200.0)
        doc_features['mood_intensity'] = self._rng.randint(0, 2, size = MusicDocument.MOOD_COUNT)
        doc_features['production_value'] = self._rng.uniform(0.6, 1)
        doc_features['fatigue_factor'] = self._rng.uniform(0.0, 1.0)
        doc_features['duration'] = self._rng.uniform(2.20, 7.30)

        return self._doc_ctor(**doc_features)