"""Minimal setup.py — makes rts_engine pip-installable for Colab."""

from setuptools import find_packages, setup

setup(
    name="rts-engine",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    description="Pure-Python Red Alert 2 text-state RTS engine for GRPO training",
)
