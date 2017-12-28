# Random play with monitor function


# Import the gym module
import gym
from gym.wrappers import Monitor

# Create a breakout environment
env = gym.make('Breakout-v0')
#env = gym.make('BreakoutDeterministic-v4')
env = Monitor(env, 'Video')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

is_done = False
while not is_done:
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    # Render
    env.render()        
