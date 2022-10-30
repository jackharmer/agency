<!-- LOGO -->

<h1 align="center">
  Agency
  <br>
</h1>

<div align="center">

![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A playground for reinforcement learning using pytorch.

This repo is for quickly testing out new ideas in RL, with an emphasis on simplicity.

</div>


<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a>
</p>

<br>

## ⭐️ Features
### Algorithms:
* SAC (N-step)
* AWAC (N-step)
* PPO (N-step)
### Distributions:
|      | Continuous | Categorical | Gumbel |
| :--- | ---------- | ----------- | ------ |
| SAC  | X          | X           | X      |
| AWAC | X          | -           | X      |
| PPO  | X          | X           | X      |
### Architectures:
|      | MLP | Convolutional |
| :--- | --- | ------------- |
| SAC  | X   | X             |
| AWAC | X   | X             |
| PPO  | X   | X             |
### Environments:
* Gym
* Unity mlagents
### N-step Discounting (without for loops!)
* Implemented using matrix multiplications, without the need for for loops, to significantly improve performance.
(see core/tools/gamma_matrix.py)


## ⚙️ Installation
Either run the following, or go to the docker section below:
```bash
conda env create -f conda_env.yaml
conda activate agency
python -m pip install -e .
```

Launch the tests:

```bash
python -m pytest tests/functional
```

Launch the training tests (tuned on a system with a 6 core CPU and Nvidia 1080 GPU):
```bash
python -m pytest tests/training
```

## 🌀 Try some examples:
#### Train Lunar Lander in 1 min!:
```bash
python examples/gym/train_sac_mlp_continuous_lunarlander.py
```
#### Train Pong in 2 mins!:
```bash
python examples/gym/train_sac_vision_categorical_pong.py
```

## 📖 Learn the framework.
The best place to start is to take a look in examples/tutorials.

train_sac_with_helper.py demonstrates how to setup a basic MLP network and use it train lunar lander using Soft Actor Critic.

It can be launched using the following command (Training takes around 1 minute on a Nvidia 1080 GPU):

```bash
python examples/tutorials/train_sac_with_helper.py
```

The examples folder contains many more files that show how to train using different algorithms, distributions and network architectures.


### Hyper parameters sweeps
See train_sac_mlp_continuous_lunarlander.py for an example of how to randomize hyper params:

```bash
python examples/gym/train_sac_mlp_continuous_lunarlander.py --sweep --n 10
```

## ⚙️ Docker
1. Install docker and nvidia-docker.
2. Build the container:
```bash
./docker_build.sh
```
3. Run the tests using code from the current working directory:
```bash
./docker_run.sh python -m pytest tests
```
4. Launch a training run
```bash
./docker_run.sh python examples/gym/train_sac_mlp_continuous_lunarlander.py
```

## 🔔 Tips and Tricks
### Setting up WSLg
Install cuda:
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

At the time of writing, this consists of:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

### Run an individual test from a file with ::test_name
```bash
python -m pytest tests/medium/test_discrete_vision_minigrid.py::test_name
```

### View terminal output when running a test with -s
```bash
python -m pytest -s tests/medium
```

## ❤️ Areas in need of some love:
* Unity env data collection.
    * add support for action branching.
    * add support for multiple brains.