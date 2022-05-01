# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
required_packages = []
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages.extend([ln.strip() for ln in file.readlines()])

test_packages = [
    "pytest==7.1.2",
    "pytest-cov==3.0.0",
]

dev_packages = [
    "black==21.4b2",
    "isort==5.10.1",
    "flake8==4.0.1",
    "pre-commit==2.18.1",
]

setup(
    name="Charlotte",
    version="1.0.0",
    description="Autonomous Driving",
    author="Aaron Grainer",
    author_email="aaronlimfz@gmail.com",
    url="https://github.com/AaronGrainer/charlotte",
    keywords=["autonomous-driving", "computer-vision"],
    classifiers=[
        "Topic :: Autonomous Driving",
        "Topic :: Computer Vision",
    ],
    # python_requires="==3.8.12",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages,
    },
)
