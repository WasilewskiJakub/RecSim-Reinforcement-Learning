from recsim import user
from recsim import choice_model
import numpy as np
from gym import spaces

from MusicEnv.Documents import MusicDocument

from recsim.environments.interest_exploration import(
    IEUserModel,
    IEDocument
)

class MusicListenerState(user.AbstractUserState):
    """Class representing the state of a music listener."""

    def __init__(self 
                 ,genre_preferences 
                 ,preferred_tempo 
                 ,sensitivity_mood 
                 ,quality_tolerance
                 ,exploration_tendency
                 ,remaining_time):
        """
        Initialize a new music listener state.

        Args:
            genre_preferences (np.array): wektor z wartosciami [0, 1.0], i-ty element wektora mówi jak bardzo uzytkownik lubi genre o id i.
            preferred_tempo (float): tempo utworu (BPM) jakie uzytkownik lubi najbardziej wartosci [40-200]
            sensitivity_mood (float): Czulosc na pole mood_intensity z klasy reeeeprzentującej dokument (myzyke)
            quality_tolerance (float): Minimalna jakosc jaką użytkownik jest w stanie zaakcptowac w utworze.
            remaining_time (float): maksymalna ilosc czasu jaka uzytkownik moze spedzic na sluchaniu muzyki.
            exploration_tendency - sklonnosc uzytkownika do poznawania utworow z nowych epok.
            genre_fatigue (np.array): wektor zmeczenia dana genre, i-ty element mowi jak bardzo sluchanie genr o id: i nudzi sluchacza.
        """
        self.genre_preferences = genre_preferences
        self.preferred_tempo = preferred_tempo
        self.sensitivity_mood = sensitivity_mood
        self.quality_tolerance = quality_tolerance
        self.remaining_time = remaining_time
        self.exploration_tendency = exploration_tendency
        self.genre_fatigue = np.zeros_like(genre_preferences)

    def score_document(self, doc_obs):

        favourite_genre = np.argmax(self.genre_preferences)
        fa_genre_extra = 1.0 + self.exploration_tendency

        if favourite_genre == doc_obs['cluster_id']:
            fa_genre_extra = 2.5

        likes = ((1 + self.genre_preferences[doc_obs['cluster_id']]) * (1 + doc_obs['quality'])) * fa_genre_extra

        tempo_bonus= 1 if abs(self.preferred_tempo - doc_obs['tempo']) <= 10 else 0
        mood_bonus = 1 + np.sum(self.sensitivity_mood == doc_obs['mood_intensity'])/MusicDocument.MOOD_COUNT
        quality_bonus = 1.0 if abs(1 - doc_obs['production_value']) <= self.quality_tolerance else 0.0
        score = likes + doc_obs['production_value'] + tempo_bonus + quality_bonus + mood_bonus if favourite_genre == doc_obs['cluster_id'] else doc_obs['production_value'] + mood_bonus
        return score

    def update_state(self, doc, response):
        """Update the user's state based on the document consumed."""
        
        self.update_preferences(doc, response.clicked)
    
    def update_fatigue(self, doc, clicked):
        cluster_id = doc.cluster_id
        preference = self.genre_preferences[cluster_id]
        if clicked:
            fatigue_increase = 0.05 * (1.0 - preference)
        else:
            fatigue_increase = 0.1 * (1.0 - preference) + 0.05
        self.genre_fatigue[cluster_id] = min(1.0, self.genre_fatigue[cluster_id] + fatigue_increase)

        for other_cluster_id in range(len(self.genre_fatigue)):
            if other_cluster_id != cluster_id:
                refresh_factor = 0.01 * self.exploration_tendency * (1.0 - self.genre_fatigue[other_cluster_id])
                self.genre_fatigue[other_cluster_id] = max(0.0, self.genre_fatigue[other_cluster_id] - refresh_factor)

    def time_consumption_update(self, doc, clicked):
        genre_preference = self.genre_preferences[doc.cluster_id]
        
        if clicked:
            reduction_percentage = 0.05 - 0.04 * genre_preference
        else:
            reduction_percentage = 0.25 + 0.05 * (1 - genre_preference)
        
        time_penalty = doc.duration * reduction_percentage
        self.remaining_time -= time_penalty
        self.remaining_time = max(0.0, self.remaining_time)
    
    def update_preferences(self, doc, clicked):
        cluster_id = doc.cluster_id
        preference = self.genre_preferences[cluster_id]

        if clicked:
            boredom_reduction = 0.02 * (1.0 - preference)
            refresh_factor_base = 0.02
        else:
            boredom_reduction = 0.05 * (1.0 - preference)
            refresh_factor_base = 0.01

        self.genre_preferences[cluster_id] = max(0.0, preference - boredom_reduction)

        for other_cluster_id in range(len(self.genre_preferences)):
            if other_cluster_id != cluster_id:
                refresh_factor = refresh_factor_base * preference
                self.genre_preferences[other_cluster_id] = min(1.0, self.genre_preferences[other_cluster_id] + refresh_factor)


    def create_observation(self):
        """Return an observable representation of the user's state."""
        return {
            'genre_preferences': self.genre_preferences
            ,'preferred_tempo': self.preferred_tempo
            ,'quality_tolerance':self.quality_tolerance
        }

    @classmethod
    def observation_space(cls):
        """Define the observation space for the user state."""
        return spaces.Dict({
            'genre_preferences': spaces.Box(low=0.0, high=1.0, shape=(IEDocument.NUM_CLUSTERS,), dtype=np.float32)
            ,'preferred_tempo': spaces.Box(low=40.0, high=200.0, shape=tuple(), dtype=np.float32)
            ,'quality_tolerance': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
        })
    
class MusicListenerSampler(user.AbstractUserSampler):
    """Sampler for generating music listeners."""

    def __init__(self
                ,seed=0
                ,user_ctor=MusicListenerState
                ,**kwargs):
        super(MusicListenerSampler, self).__init__(user_ctor, **kwargs)
        self._rng = np.random.default_rng(seed)

    def sample_user(self):
        """Generate a new music listener with randomized preferences."""
        genre_preferences = self._rng.uniform(0.0, 1.0, IEDocument.NUM_CLUSTERS)
        preferred_tempo = self._rng.uniform(40, 200)  # BPM
        sensitivity_mood = self._rng.integers(0, 2, size = MusicDocument.MOOD_COUNT)
        quality_tolerance = self._rng.uniform(0.01, 0.3)  # Minimum quality
        remaining_time = self._rng.uniform(60.0, 600.0)  # Listening time in minutes
        exploration_tendency = np.clip(self._rng.normal(loc=0.3, scale=0.25), 0.02, 0.80)

        features = {}

        features['genre_preferences'] = genre_preferences
        features['preferred_tempo'] = preferred_tempo
        features['sensitivity_mood'] = sensitivity_mood
        features['quality_tolerance'] = quality_tolerance
        features['remaining_time'] = remaining_time
        features['exploration_tendency'] = exploration_tendency
        
        return self._user_ctor(**features)

class MusicResponse(user.AbstractResponse):
    """Class to represent a user's response to a music document."""

    def __init__(self
                ,clicked=False
                ,quality=0.0
                ,cluster_id=0
                ,listen_time = 0.0
                ,genre_preferences = []
                ,preferred_tempo = 0.0
                ,discovery_potential = 0.0
                ,virality_factor = 0.0
                ,tempo = 0.0
                ,production_value = 0.0
                ,duration = 0.0
                ,user_score = 0.0
                ,mood_intensity = []
                ,quality_tolerance = 0.0):
        
        # Other:
        self.clicked = clicked
        self.listen_time = listen_time
        self.user_score = user_score
        
        #Document:
        self.quality = quality
        self.cluster_id = cluster_id
        self.discovery_potential = discovery_potential
        self.virality_factor = virality_factor
        self.tempo = tempo
        self.production_value = production_value
        self.duration = duration
        self.mood_intensity = mood_intensity

        # Listener:
        self.genre_preferences = genre_preferences
        self.preferred_tempo = preferred_tempo
        self.quality_tolerance = quality_tolerance


    def create_observation(self):
        return {
            'click': int(self.clicked)
            ,'listen_time': np.array(self.listen_time)

            ,'quality': np.array(self.quality)
            ,'cluster_id': int(self.cluster_id)
            ,'discovery_potential': np.array(self.discovery_potential)
            ,'virality_factor':np.array(self.virality_factor)
            ,'tempo':np.array(self.tempo)
            ,'production_value':np.array(self.production_value)
            ,'duration':np.array(self.duration)

            ,'genre_preferences':np.array(self.genre_preferences)
            ,'preferred_tempo': np.array(self.preferred_tempo)
            ,'quality_tolerance':np.array(self.quality_tolerance)

            ,'user_score' : np.array(self.user_score)
        }

    @classmethod
    def response_space(cls):
        return spaces.Dict({
            'click': spaces.Discrete(2)
            ,'listen_time': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'user_score': spaces.Box(low=0.0, high=np.inf, shape=tuple(), dtype=np.float32)
            
            ,'quality': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'cluster_id': spaces.Discrete(IEDocument.NUM_CLUSTERS)
            ,'discovery_potential': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'virality_factor': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'tempo': spaces.Box(low=40.0, high=200.0, shape=tuple(), dtype=np.float32)
            ,'production_value': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
            ,'duration': spaces.Box(low=2.20, high=7.30, shape=tuple(), dtype=np.float32)
            ,'mood_intensity': spaces.Box(low=0.0, high=1.0, shape=(MusicDocument.MOOD_COUNT,), dtype=np.float32)

            ,'genre_preferences': spaces.Box(low=0.0, high=1.0, shape=(IEDocument.NUM_CLUSTERS,), dtype=np.float32)
            ,'preferred_tempo': spaces.Box(low=40.0, high=200.0, shape=tuple(), dtype=np.float32)
            ,'quality_tolerance': spaces.Box(low=0.0, high=1.0, shape=tuple(), dtype=np.float32)
        })


class MusicListenerModel(user.AbstractUserModel):
    """Model for simulating music listeners' interactions."""

    def __init__(self 
                 ,slate_size 
                 ,choice_model_ctor=choice_model.MultinomialLogitChoiceModel
                 ,user_state_ctor=MusicListenerState
                 ,response_model_ctor = MusicResponse
                 ,seed=0):
        """
        Initialize the music listener model.

        Args:
            slate_size (int): Number of items in a recommendation slate.
            choice_model_ctor (callable): Constructor for the choice model.
            user_state_ctor (callable): Constructor for the user state.
            response_model_ctor (callable): Constructor for the response model.
            seed (int): Seed for random number generator.
        """
        sampler = MusicListenerSampler(seed=seed, user_ctor = user_state_ctor)
        super(MusicListenerModel, self).__init__(response_model_ctor, sampler, slate_size)
        if choice_model_ctor is None:
            raise ValueError("A choice model constructor must be provided.")
        self.choice_model = choice_model_ctor({'no_click_mass': 5,'min_normalizer': 0.0})


    def is_terminal(self):
        """Check if the session is over."""
        return self._user_state.remaining_time <= 0

    def update_state(self, slate_documents, responses):
        """Update the user's state based on responses to the slate."""
        for doc, response in zip(slate_documents, responses):
            self._user_state.update_state(doc, response)

    def calculate_listentime(self, i, selected_index, documents, response):
        if selected_index == i:
            response.clicked = True
            if self._user_state.genre_preferences[documents[i].cluster_id] > 0.7:
                listen_time_percentage = np.random.uniform(0.7, 1.0)
            else:
                listen_time_percentage = np.random.uniform(0.6, 0.8)
        else:
            response.clicked = False
            if self._user_state.genre_preferences[documents[i].cluster_id] > 0.7:
                listen_time_percentage = np.random.uniform(0.3, 0.6)
            else:
                listen_time_percentage = np.random.uniform(0.1, 0.3)

        response.listen_time = listen_time_percentage

    def simulate_response(self, documents):
        """Simulate user responses to a slate of documents."""
        responses = [self._response_model_ctor() for _ in documents]

        doc_observations = [doc.create_observation() for doc in documents]
        self.choice_model.score_documents(self._user_state, doc_observations)
        selected_index = self.choice_model.choose_item()

        for i, response in enumerate(responses):
            self.calculate_listentime(i, selected_index, documents, response)
            response.user_score = self._user_state.score_document(doc_observations[i])

            response.quality = documents[i].quality
            response.cluster_id = documents[i].cluster_id
            response.discovery_potential = documents[i].discovery_potential
            response.virality_factor = documents[i].virality_factor
            response.tempo = documents[i].tempo
            response.production_value = documents[i].production_value
            response.duration = documents[i].duration

            response.genre_preferences = self._user_state.genre_preferences
            response.preferred_tempo = self._user_state.preferred_tempo
            response.quality_tolerance = self._user_state.quality_tolerance

        if selected_index is not None:
            self._generate_clicked(documents[selected_index], responses[selected_index])

        return responses

    def _generate_clicked(self, doc, response):
        response.clicked = True


def clicked_watchtime_reward(responses):
  """Calculates the total clicked watchtime from a list of responses.

  Args:
    responses: A list of IEvResponse objects

  Returns:
    reward: A float representing the total watch time from the responses
  """
  reward = 0.0
  for response in responses:
    if response.clicked:
        reward += response.user_score
  return reward