# Imports from Sample Factory library
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.enjoy import enjoy
from sample_factory.huggingface.huggingface_utils import load_from_hf

# Imports specific for gym environment from sample factory examples
from sf_examples.train_gym_env import parse_custom_args, make_gym_env_func
import torch
import panda_test_env
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Register Lunar Lander environment
register_env(panda_test_env.__ID__, make_gym_env_func)


if __name__ == "__main__":
    # Initialize basic arguments for running the experiment. These parameters are required to run any experiment
    # The parameters can also be specified in the command line
    experiment_name = "panda_robot_exp"
    argv = [
        "--algo=APPO",
        f"--env={panda_test_env.__ID__}",
        f"--experiment={experiment_name}",
    ]

    cfg = parse_custom_args(argv=argv, evaluation=False)

    # The following parameters can be changed from the default
    cfg.reward_scale = 0.05
    cfg.train_for_env_steps = 1_000_000
    cfg.gae_lambda = 0.99
    cfg.num_workers = 20
    cfg.num_envs_per_worker = 6
    cfg.seed = 0

    # Experiments can also be run using CPU only
    # For best performance, it is recommended to turn on GPU Hardware acceleration in Colab under "Runtime" > "Change Runtime Type"
    cfg.device = "gpu" if torch.cuda.is_available() else "cpu"

    run_rl(cfg)
