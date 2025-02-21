import gym
import panda_test_env

print(panda_test_env.__version__)
env = gym.make("panda-test-v0")
env.reset()


for i in range(1000):
    env.render()
    action = env.action_space.sample()
    env.step(action)
    if i%100 == 0:
        print("action:", action) 

env.close()
