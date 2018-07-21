import numpy as np
import random, os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


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
        self.learning_rate = 0.01
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



#main function for testing.
if __name__ == '__main__':
    
    agent = DeepQLearningAgent(1, 2)
    agent.load_data()
    for i in range(0, 50):
        print ('Round:', i + 1)
        position = 1
        prev_state = []
        curr_state = []
        prev_action = -1
        curr_action = -1
        reward = 0
        done = False
        
        while True:
            #Renew state
            prev_state = curr_state
            curr_state = [position]

            #Renew action
            prev_action = curr_action
            curr_action = agent.get_action(curr_state)
            if curr_action == 0:
                position = max(1, position - 1)
            elif curr_action == 1:
                position = position + 1
                
            #Study
            if prev_action != -1:
                reward = 0
                if curr_state[0] == 6:
                    reward = 1
                    done = True
                agent.remember(prev_state, prev_action, reward, curr_state, done)
                agent.replay(agent.batch_size)
                
            print(curr_state)
            if done:
                done = False
                break
    
    agent.save_data()
    print('END')

