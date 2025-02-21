from gym.envs.registration import register

__version__ = "0.0.4"
register(id="panda-test-v0", entry_point="panda_test_env.envs:PandaEnv")
