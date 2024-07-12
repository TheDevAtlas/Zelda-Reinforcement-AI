import multiprocessing
from multiprocessing import Pool
from pyboy import PyBoy
import tensorflow as tf
import numpy as np
import random

# train over the course of a week constantly #

# Define the neural network model
def create_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025), loss='mse')
    return model

# Define the agent
class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.model = create_model(input_shape, num_actions)
        self.target_model = create_model(input_shape, num_actions)
        self.update_target_model()
        self.memory = []
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.batch_size = 32
        self.num_actions = num_actions

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to run the game
def run_game(instance_id):
    pyboy = PyBoy('loz.gbc', window_type="headless")
    agent = DQNAgent(input_shape=(160, 144, 1), num_actions=8)
    pyboy.set_emulation_speed(0)
    
    done = False
    state = pyboy.get_screen_buffer()
    state = np.reshape(state, [1, 160, 144, 1])
    for _ in range(2 * 60 * 60):  # Run for 2 game hours at 60 fps
        action = agent.act(state)
        pyboy.send_input(WindowEvent.PRESS_ARROW_UP)  # Example action
        pyboy.tick()
        next_state = pyboy.get_screen_buffer()
        next_state = np.reshape(next_state, [1, 160, 144, 1])
        reward = 1  # Define your reward function
        done = False  # Define your done condition
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay()
    pyboy.stop()
    return agent

# Run multiple instances in parallel
def run_parallel_games(num_instances):
    with Pool(processes=num_instances) as pool:
        agents = pool.map(run_game, range(num_instances))
    return agents

if __name__ == "__main__":
    num_instances = 16
    agents = run_parallel_games(num_instances)
    for i, agent in enumerate(agents):
        print(f"Instance {i}: Epsilon = {agent.epsilon}")

