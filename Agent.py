import numpy as np
import random, os, json

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

import skimage as skimage
from skimage import transform, color, exposure, io
from skimage.transform import rotate
from matplotlib import pyplot as plt
from django.urls.conf import path


class DeepQLearningAgent:
    
    def __init__(self, 
                 state_size, 
                 action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1     # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.batch_size = 32
        self.filename = './data/data.h5'
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        print(self.epsilon)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load_data(self):
        if os.path.exists(self.filename):
            self.model.load_weights(self.filename, True)


    def save_data(self):
        self.model.save_weights(self.filename, True)


class DQNAgent:
    
    def __init__(self, 
                 action_size=2, 
                 memory_size=50000, 
                 bitch_size=32,
                 learning_rate=0.1,
                 gamma=0.95,
                 epsilon=1.0,
                 model_file='model.h5',
                 model_json='model.json'):
        self.state_shape = (84, 84, 4)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.bitch_size = bitch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_file = model_file
        self.model_json = model_json
        self.frames = deque(maxlen=4)
        self.loss = 0
        self.model = self.__build_model()
    
    def __build_model(self):
        model = Sequential()
#         model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=self.state_shape))
#         model.add(Activation('relu'))
#         model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
#         model.add(Activation('relu'))
#         model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
#         model.add(Activation('relu'))
#         model.add(Flatten())
#         model.add(Dense(512))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', input_shape=self.state_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def get_random_action(self):
        return random.randint(0, self.action_size - 1), ['random']

    def get_optimize_action(self, state):
        actions = self.model.predict(state)[0]
        action = np.argmax(actions)
        return action, actions
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.get_random_action()
        return self.get_optimize_action(state)
    
    def get_state_by_frame(self, frame):
        image = self.image_processing(frame)
        #self.show_image(image)
        self.frames.append(image)
        while len(self.frames) < 4:
            self.frames.append(image)
        state = np.stack((self.frames[0], self.frames[1], self.frames[2], self.frames[3]), axis=2)
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        return state
    
    def get_state_by_path(self, path):
        frame = self.read_image(path)
        return self.get_state_by_frame(frame)
        
    def store_transition(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        
    def train(self):
        if len(self.memory) < self.bitch_size:
            return
        minibatch = random.sample(self.memory, self.bitch_size)
#         for state, action, reward, next_state, terminal in minibatch:
#             new_reward = reward
#             if terminal == False:
#                 max_action_value = np.max(self.model.predict(next_state)[0])
#                 new_reward = reward + self.gamma * max_action_value
#             actions = self.model.predict(state)
#             actions[0][action] = new_reward
#             self.model.fit(state, actions, epochs=1, verbose=0)
        state, action, reward, next_state, terminal = zip(*minibatch)
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)
        targets = self.model.predict(state)
        actions = self.model.predict(next_state)
        targets[range(self.bitch_size), action] = reward + self.gamma * np.max(actions, axis=1) * np.invert(terminal)
        self.loss = self.model.train_on_batch(state, targets)
        if self.epsilon > 0.1:
            self.epsilon -= 0.0000001
            
    def load_data(self):
        if os.path.exists(self.model_file):
            self.model.load_weights(self.model_file)

    def save_data(self):
        self.model.save_weights(self.model_file, overwrite=True)
        with open(self.model_json, "w") as file:
            json.dump(self.model.to_json(), file)

    def image_processing(self, image):
        image = color.rgb2gray(image)
        image = transform.resize(image, (self.state_shape[0], self.state_shape[1]))
        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = image / 255
        return image
    
    def show_image(self, image):
        io.imshow(image)
        plt.show()
    
    def read_image(self, path):
        return io.imread(path)


#main function for testing.
if __name__ == '__main__':
    image = io.imread('screenshot.jpg')
    io.imshow(image)
    plt.show()
    image = color.rgb2gray(image)
    io.imshow(image)
    plt.show()
    image = transform.resize(image, (80, 80))
    io.imshow(image)
    plt.show()
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    io.imshow (image)
    plt.show()
    image = image / 255
    io.imshow(image)
    plt.show()
    print(image.shape)
    
<<<<<<< HEAD
    state = np.stack((image, image, image, image), axis=2)
    print(state.shape)
    
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
    print(state.shape)
    
    #agent = DQNAgent()
    #print(agent.get_random_action())
#     agent = DeepQLearningAgent(1, 2)
#     agent.load_data()
#     for i in range(0, 50):
#         print ('Round:', i + 1)
#         position = 1
#         prev_state = []
#         curr_state = []
#         prev_action = -1
#         curr_action = -1
#         reward = 0
#         done = False
#         
#         while True:
#             #Renew state
#             prev_state = curr_state
#             curr_state = [position]
# 
#             #Renew action
#             prev_action = curr_action
#             curr_action = agent.get_action(curr_state)
#             if curr_action == 0:
#                 position = max(1, position - 1)
#             elif curr_action == 1:
#                 position = position + 1
#                 
#             #Study
#             if prev_action != -1:
#                 reward = 0
#                 if curr_state[0] == 6:
#                     reward = 1
#                     done = True
#                 agent.remember(prev_state, prev_action, reward, curr_state, done)
#                 agent.replay(agent.batch_size)
#                 
#             print(curr_state)
#             if done:
#                 done = False
#                 break
#     
#     agent.save_data()
#     print('END')
=======
    agent = DeepQLearningAgent(1, 2)
    #agent.load_data()
    for i in range(0, 50):
        print ('Round:', i + 1)
        position = 1
        prev_state = np.array([position])
        curr_state = np.array([position])
        prev_action = -1
        curr_action = -1
        reward = 0
        done = False
 
        while True:
            #Renew state
            prev_state = curr_state
            curr_state = np.array([position])

            #Renew action
            prev_action = curr_action
            curr_action = agent.get_action(curr_state)
            if curr_action == 0:
                position = max(1, position - 1)
            elif curr_action == 1:
                position = position + 1
            
            #Study
            if prev_action != -1:
                reward = -1
                if curr_state[0] == 6:
                    reward = 1
                    done = True
                agent.remember(prev_state, prev_action, reward, curr_state, done)
                agent.replay(agent.batch_size)
                
            print(curr_state, agent.epsilon)
            if done:
                done = False
                break
    
    #agent.save_data()
    print('END')
>>>>>>> f21f18890147640dbe340bccee9e56553413d15f

