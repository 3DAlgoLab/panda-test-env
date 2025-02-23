# from gym.envs.registration import register
from gymnasium import register

__version__ = "0.0.5.dev"
__ID__ = "panda-test-v0"
register(id=__ID__, entry_point="panda_test_env.envs:PandaEnv")
