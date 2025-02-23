from typing import Optional
import argparse
import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl


from panda_test_env.envs import PandaEnv

def make_custom_env(
    full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None
):
    # see the section below explaining arguments
    return PandaEnv(full_env_name, cfg, env_config, render_mode=render_mode)


def register_custom_env_envs():
    # register the env in sample-factory's global env registry
    # after this, you can use the env in the command line using --env=custom_env_name
    register_env("panda_test_env", make_custom_env)


def add_custom_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
    # You can extend the command line arguments here
    p.add_argument("--custom_argument", default="value", type=str, help="")


def custom_env_override_defaults(_env, parser):
    # Modify the default arguments when using this env.
    # These can still be changed from the command line. See configuration guide for more details.
    parser.set_defaults(
        encoder_conv_architecture="convnet_atari",
        obs_scale=255.0,
        gamma=0.99,
        learning_rate=0.00025,
        lr_schedule="linear_decay",
        adam_eps=1e-5,
    )


def parse_args(argv=None, evaluation=False):
    # parse the command line arguments to build
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_custom_env_args(partial_cfg.env, parser, evaluation=evaluation)
    custom_env_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_custom_env_envs()
    cfg = parse_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
