import gym
import panda_test_env

print(panda_test_env.__version__)
env = gym.make("panda-test-v0")


done = False
error = 0.01
fingers = 1
# info = [0.7, 0, 0.1]
info = {"object_position": [0.7, 0, 0.1]}

k_p = 10  # propotional gain
k_d = 1  # derivative gain
dt = 1.0 / 240.0  # the default timestep in pybullet is 240 Hz
t = 0

for i_episode in range(5):
    observation = env.reset()
    fingers = 1
    for t in range(100):
        env.render()
        print(observation)
        pos = info["object_position"]
        dx = pos[0] - observation[0]
        dy = pos[1] - observation[1]
        dz = pos[2] - observation[2]
        target_z = pos[2]
        if abs(dx) < error and abs(dy) < error and abs(dz) < error:
            fingers = 0
        if (observation[3] + observation[4]) < error + 0.02 and fingers == 0:
            target_z = 0.5
        dz = target_z - observation[2]

        pd_x = k_p * dx + k_d * dx / dt
        pd_y = k_p * dy + k_d * dy / dt
        pd_z = k_p * dz + k_d * dz / dt

        action = [pd_x, pd_y, pd_z, fingers]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
