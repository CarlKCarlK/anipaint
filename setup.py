from setuptools import setup

setup(
    name="anipaint",
    version="0.1.0",
    description="A Python package for painting mattes",
    install_requires=["pysnptools", "pillow"],
    python_requires="~=3.8",
    license="proprietary",
    packages=["anipaint", "anipaint/tests"],  # basically, everything with a __init__.py
)
