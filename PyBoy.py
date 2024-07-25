# https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)
# https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)/RAM_map

# DATE + Training Fitness #
# 7/18/2024 - 1 : 802679
# 7/18/2024 - 2 : 930830
# 7/18/2024 - 3 : 935034
# 7/18/2024 - FIXED FITNESS ERROR
# 7/18/2024 - 4 : 40
# 7/18/2024 - 5 : 40 -> 20
# 7/18/2024 - FIXED OVERWRITE ISSUE - KEEPS BEST
# 7/18/2024 - 6 : 20 -> 35
# 7/18/2024 - set of 5 training : 

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import multiprocessing
import os
import pickle
import time

actions = ['', 'a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class GenericPyBoyEnv(gym.Env):

    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0:
            pass
        else:
            self.pyboy.button(actions[action])

        self.pyboy.tick(30,True)

        done = self.pyboy.game_wrapper.game_over

        self._calculate_fitness()
        reward = self._fitness# - self._previous_fitness

        observation = self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        inventoryValues = 0

        # Inventory memory locations
        inventory_start = 0xDB02
        inventory_end = 0xDB0B

        # Loop through the inventory memory range and add the values to inventoryValues
        for address in range(inventory_start, inventory_end + 1):
            if self.pyboy.memory[address] > 0:
                inventoryValues += 7

        # Items held memory locations
        held_start = 0xDB00
        held_end = 0xDB01

        for address in range(held_start, held_end + 1):
            if self.pyboy.memory[address] > 0:
                inventoryValues += 10  # Assign a higher value for held items

        # World map status memory locations
        visit_start = 0xD800
        visit_end = 0xD8FF

        for address in range(visit_start, visit_end + 1):
            byte_value = self.pyboy.memory[address]
            if byte_value & 0x80:  # visited
                inventoryValues += 15
            if byte_value & 0x20:  # owl talked
                inventoryValues += 10
            if byte_value & 0x10:  # changed from initial status
                inventoryValues += 5

        # Other significant memory locations and values
        if self.pyboy.memory[0xDB0C] == 0x01:  # Flippers
            inventoryValues += 20
        if self.pyboy.memory[0xDB0D] == 0x01:  # Potion
            inventoryValues += 20
        if self.pyboy.memory[0xDB43] > 0:  # Power bracelet level
            inventoryValues += 10 * self.pyboy.memory[0xDB43]
        if self.pyboy.memory[0xDB44] > 0:  # Shield level
            inventoryValues += 10 * self.pyboy.memory[0xDB44]
        if self.pyboy.memory[0xDB49] > 0:  # Ocarina songs
            inventoryValues += 5 * bin(self.pyboy.memory[0xDB49]).count('1')
        if self.pyboy.memory[0xDB4D] > 0:  # Number of bombs
            inventoryValues += 2 * self.pyboy.memory[0xDB4D]
        if self.pyboy.memory[0xDB45] > 0:  # Number of arrows
            inventoryValues += 2 * self.pyboy.memory[0xDB45]
        if self.pyboy.memory[0xDB5D] > 0 or self.pyboy.memory[0xDB5E] > 0:  # Number of rupees
            inventoryValues += self.pyboy.memory[0xDB5D] + self.pyboy.memory[0xDB5E]

        # Update the fitness value
        self._fitness = inventoryValues

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0

        observation = self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

def save_model(index, fitness):
    model_filename = f'model_{index}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(fitness, f)
    print(f"Model {index} saved with fitness {fitness}")

def load_best_model():
    best_fitness = -np.inf
    best_model_filename = None
    for filename in os.listdir('.'):
        if filename.startswith('model_') and filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                fitness = pickle.load(f)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_model_filename = filename
    if best_model_filename:
        with open(best_model_filename, 'rb') as f:
            best_fitness = pickle.load(f)
        print(f"Best model loaded: {best_model_filename} with fitness {best_fitness}")
    else:
        print("No previous model found. Starting fresh.")
    return best_fitness

def run_bot(index):
    # Initialize PyBoy with the specific game ROM path
    pyboy = PyBoy('loz.gbc')#, window="null")
    env = GenericPyBoyEnv(pyboy, debug=False)
    observation, info = env.reset()

    best_fitness = load_best_model()

    start_time = time.time()
    one_hour = 60 * 60  # 1 hour in seconds

    while time.time() - start_time < one_hour:
        action = env.action_space.sample()  # Replace with your action selection logic
        observation, reward, done, truncated, info = env.step(action)
        # if done:
        #     break

    if env._fitness > best_fitness:
        save_model(index, env._fitness)
    else:
        print(f"Model {index} fitness {env._fitness} did not improve over best fitness {best_fitness}")

    env.close()

if __name__ == "__main__":
    for epoch in range(11): # sets of training
        processes = []
        for i in range(4):  # Number of bots you want to run
            p = multiprocessing.Process(target=run_bot, args=(i,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
