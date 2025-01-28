import numpy as np
from recsim.agent import AbstractEpisodicRecommenderAgent
from collections import defaultdict, deque
import random
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K

from MusicEnv.Documents import MusicDocument
from recsim.environments.interest_exploration import IEDocument


class DqnAgent(AbstractEpisodicRecommenderAgent):
    
    def _normalize_tempo(self, tempo):
        return (tempo - MusicDocument.MIN_TEMPO) / (MusicDocument.MAX_TEMPO - MusicDocument.MIN_TEMPO)
    
    def _extractUserFeatures(self, userF):
        one_hot_preferences = np.zeros(IEDocument.NUM_CLUSTERS)
        one_hot_preferences[np.argmax(userF['genre_preferences'])] = 1
        
        tmp = np.concatenate((
            np.array(one_hot_preferences) # 6
            ,np.array([np.argmax(userF['genre_preferences'])]) # 6
            ,np.array([self._normalize_tempo(userF['preferred_tempo'])]) # 1
            ,np.array([userF['quality_tolerance']]) # 1
        ))
        #  14
        return tmp
    
    def _extractDocumentFeatures(self, document):
        quality = document['quality']
        production_value = document['production_value']
        tempo = self._normalize_tempo(document['tempo'])
        
        cluster_id = document['cluster_id']
        one_hot_cluster = np.zeros(IEDocument.NUM_CLUSTERS)
        one_hot_cluster[cluster_id] = 1
        
        tmp = np.concatenate((
            np.array(one_hot_cluster) # 6
            ,np.array([quality]) # 1
            ,np.array([production_value]) # 1
            ,np.array([tempo]) # 1
            ,document['mood_intensity'] # 12
        ))
        return tmp
    
    def _get_input_from_curent_observation(self, observation):
        documents = observation['doc']
        user = observation['user']
        userF =  np.array(self._extractUserFeatures(user))
        input_matrix = []
        for doc in documents:
            observ_state = np.concatenate((self._extractDocumentFeatures(documents[doc]), userF))
            input_matrix.append(observ_state)
        return np.array(input_matrix, dtype=np.float32)
    
    def _get_training_set_from_prev_state(self, observation):
        def calculateScore(response):
            return response['click'] * response['user_score']
        
        response = observation['response']
        userF= self._extractUserFeatures(response[0])
        input_matrix = []
        target_vector = []
        for res in response:
            observ_state = np.concatenate((self._extractDocumentFeatures(res), userF))
            input_matrix.append(observ_state)
            target_vector.append(calculateScore(res))

        return np.array(input_matrix, dtype=np.float32), np.array(target_vector, dtype=np.float32)
    
    def getInuptSize(self):
        docs = (
            IEDocument.NUM_CLUSTERS
            + 1
            + 1
            + 1
            + MusicDocument.MOOD_COUNT
        )
        user = (
            IEDocument.NUM_CLUSTERS
            # + IEDocument.NUM_CLUSTERS
            + 1
            + 1
            + 1
        )
        return docs + user
    
    def _build_network(self,lr):
        model = Sequential([
            Dense(128, activation='relu', input_dim=(self.getInuptSize())),  # Dopasuj do wymiaru stanu
            Dense(32, activation='relu'),
            Dense(64, activation='sigmoid'),
            Dense(1, activation='linear')  # Jedna Q-wartość na każdą możliwą akcję
        ])
        # model = Sequential([
        #     Dense(128, activation='relu', input_dim=(self.getInuptSize())),  # Dopasuj do wymiaru stanu
        #     Dense(128, activation='relu'),
        #     Dense(1, activation='linear')  # Jedna Q-wartość na każdą możliwą akcję
        # ])
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        return model
    
    # MAIN COMPONENS IMPLEMENTATION: 

    def __init__(self, observation_space, action_space, learning_rate=0.02, discount_factor=0.9, epsilon_end = 0.01):
        super(DqnAgent, self).__init__(action_space)
        if len(observation_space['doc'].spaces) < len(action_space.nvec):
            raise RuntimeError('Slate size larger than size of the corpus.')
        self.slate_size = len(action_space)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.2
        self.epsilon_decay = 0.9
        self.current_epsilon = self.epsilon_start
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.session = tf.Session()
            K.set_session(self.session)
            
            self.discount_factor = discount_factor
            self.replay_buffer = deque(maxlen=300000)
            self.train = True

            self.q_network = self._build_network(learning_rate)
            self.td_network = self._build_network(learning_rate)

            self.session.run(tf.global_variables_initializer())
            self.td_network.set_weights(self.q_network.get_weights())
            
            self.c = 0
            self.c_step = 50

        self.eps_counter = 0
        self.eps_step = 10



    def step(self, reward, observation):
        self._get_input_from_curent_observation(observation)

        if self.train and reward is not None and observation['response'] is not None:
            self._update_q_values(observation)
        
        documents = observation['doc']
        doc_ids = list(documents.keys())
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        if self.train and np.random.rand() < self.current_epsilon or observation['response'] is None:
        # if False:    
            slate = np.random.choice(doc_ids, self.slate_size, replace=False).tolist()
        else:
            with self.graph.as_default():
                with self.session.as_default():
                    X_current = self._get_input_from_curent_observation(observation)
                    q_values = self.q_network.predict(X_current).flatten()
            top_n_indices = np.argsort(q_values)[-self.slate_size:][::-1]
            return top_n_indices.tolist()

        slate_indices = [doc_id_to_index[doc_id] for doc_id in slate]
        return list(map(int, slate_indices))
   

    def _update_q_values(self, observation):
        with self.graph.as_default():
            with self.session.as_default():
                self.c += 1
                if self.c % self.c_step == 0:
                    self.td_network.set_weights(self.q_network.get_weights())
                
                X_train, Y_train = self._get_training_set_from_prev_state(observation)
                X_current = self._get_input_from_curent_observation(observation)

                next_max_q = max(self.td_network.predict(X_current))
                y = [score + self.discount_factor * (next_max_q * 0.0 if score == 0 else 1.0) for score in Y_train]
                
                self.replay_buffer.extend(zip(X_train, np.array(y).reshape(-1,1)))


                batch_size = 8

                # batch_size = 32
                if len(self.replay_buffer) >= batch_size:
                    batch = random.sample(self.replay_buffer, batch_size)
                    x, y = zip(*batch)
                    x = np.array(x)
                    y = np.array(y)
                    self.q_network.train_on_batch(x, y)

    def end_episode(self, reward, observation=None):
            print("[LOG] buffer lenght:", len(self.replay_buffer))

            self.eps_counter += 1
            mooood = self.eps_counter % self.eps_step
            if  mooood != 0:
                return
            
            self.current_epsilon = max(self.epsilon_end, 
                                self.current_epsilon * self.epsilon_decay)
            print(f"[LOG]: epsilon {self.current_epsilon}")

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()