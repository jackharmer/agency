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
* SAC (N-Step)
* AWAC (N-Step)
### Distributions:
* Continuous
* Categorical
* Gumbel
### Architectures:
* MLP
* Convolutional
### Environments:
* Gym
* Unity mlagents


## ⚡️ Installation
```bash
conda env create -f ./conda_env.yaml
conda activate agency
python -m pip install -e .
```

Launch the tests:

```bash
python -m pytest tests/short
```

Longer test can be found in tests/medium.


## 📖 Getting started
The best place to start is to take a look in examples/tutorials.

train_sac_with_helper.py demonstrates how to setup a basic MLP network and use it train lunar lander using Soft Actor Critic.

It can be launched using the following command (Training takes around 10-15 mins on a consumer GPU (1080)):

```bash
python examples/tutorials/train_sac_with_helper.py
```

The examples folder contains many more files that show how to train using different algorithms, distributions and network architectures.


### Hyper parameters sweeps
See train_sac_mlp_continuous_lunarlander.py for an example of how to randomize hyper params:

```bash
python examples/gym/train_sac_mlp_continuous_lunarlander.py --sweep --n 10
```


## ⚙️ Training Status
| Algo. |Arch.  | Action space| Example envs that train        |
|:------|-------|-------------|-------------------------------|
| SAC   | MLP   | Categorical | CartPole, Identity            |
| SAC   | MLP   | Gumbel      | Identity                      |
| SAC   | MLP   | Continuous  | LunarLander, 3dBall, Identity |
| SAC   | Conv. | Continuous  | CarRacing                     |
| SAC   | Conv. | Gumbel      | GridWorld, basic              |
| SAC   | Conv. | Categorical | Minigrid, Pong                |
| AWAC  | MLP   | Categorical |  --Not supported--            |
| AWAC  | MLP   | Gumbel      | Identity                     |
| AWAC  | MLP   | Continuous  | LunarLander, 3dBall         |
| AWAC  | Conv. | Continuous  | -                           |
| AWAC  | Conv. | Gumbel      | Pong                        |
| AWAC  | Conv. | Categorical | --Not supported--           |



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
* Gym env data collection
    * Improve performance.
    * Add vec env rendering.
* Unity env data collection.
    * add support for action branching.
    * add support for multiple brains.