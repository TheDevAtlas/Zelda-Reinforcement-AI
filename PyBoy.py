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
        # self.pyboy.tick(1, False)
        self.pyboy.tick(1)

        done = self.pyboy.game_wrapper.game_over

        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

        observation=self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness

        # NOTE: Only some game wrappers will provide a score
        # If not, you'll have to investigate how to score the game yourself
        # self._fitness=self.pyboy.game_wrapper.score

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
    
    for _ in range(100000):  # Run for a fixed number of steps or until done
        action = env.action_space.sample()  # Replace with your action selection logic
        observation, reward, done, truncated, info = env.step(action)
        pyboy.tick()
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