# https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)
# https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)/RAM_map



import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import multiprocessing

actions = ['','a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class GenericPyBoyEnv(gym.Env):

    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        self._fitness=0
        self._previous_fitness=0
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

        # Consider disabling renderer when not needed to improve speed:
        #self.pyboy.tick(1, False)
        self.pyboy.tick(1)

        done = self.pyboy.game_wrapper.game_over

        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

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
        self._fitness += inventoryValues


    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness=0
        self._previous_fitness=0

        observation=self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

def run_bot(index):
    # Initialize PyBoy with the specific game ROM path
    pyboy = PyBoy('loz.gbc')
    env = GenericPyBoyEnv(pyboy, debug=False)
    observation, info = env.reset()
    
    for _ in range(60000):  # Run for a fixed number of steps or until done
        action = env.action_space.sample()  # Replace with your action selection logic
        observation, reward, done, truncated, info = env.step(action)
        #pyboy.tick()
        #if done:
            #break
        

    env.close()

if __name__ == "__main__":
    processes = []
    for i in range(6):  # Number of bots you want to run
        p = multiprocessing.Process(target=run_bot, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()