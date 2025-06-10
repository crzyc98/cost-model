from setuptools import find_packages, setup

setup(
    name="cost-model",
    version="0.1.0",
    packages=find_packages(include=["cost_model*"]),
    python_requires=">=3.9",
)
