import setuptools
from pathlib import Path
import panda_test_env

setuptools.setup(
    name="panda_test_env",
    author="Jaeyoon Jeong",
    version=panda_test_env.__version__,
    description="A OpenAI Gym env for testing pybullet simulator",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include="panda_test_env*"),
    install_requires=["gym", "pybullet"],
    python_requires=">=3.9",
)
