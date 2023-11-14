# kagu-torch

[![PyPI](https://img.shields.io/pypi/v/kagu-torch)](https://pypi.org/project/kagu-torch/)
[![Publish to PyPI](https://github.com/ramyamounir/kagu-torch/actions/workflows/pypi_publish.yaml/badge.svg)](https://github.com/ramyamounir/kagu-torch/actions/workflows/pypi_publish.yaml)

Official implementation for IJCV paper [Towards Automated Ethogramming: Cognitively-Inspired Event Segmentation for Wildlife Monitoring](https://ramymounir.com/publications/AutomatedEthogramming/)

![Overview of Kagu](https://github.com/ramyamounir/kagu-torch/blob/main/assets/overview.png)


---

## Overview

### Documentation

Checkout the [documentation](https://ramymounir.com/docs/kagu/) of code to learn more details.

### Installation

```bash
pip install kagu-torch # with pip from PyPI
pip install git+'https://github.com/ramyamounir/kagu-torch' # with GitHub
```


### Training

Use the provided python [training script](https://github.com/ramyamounir/kagu-torch/blob/main/train.py) to train or multiple gpus. Bash scripts with CLI arguments are provided in the [helper_scripts](https://github.com/ramyamounir/kagu-torch/tree/main/helper_scripts)

> We use the [DDPW library](https://ddpw.projects.sujal.tv/) to enable scaling up our training to SLURM with one line of code.


---

Citing our paper
----------------
If you find our approaches useful in your research, please consider citing:
```
@article{mounir2023towards,
  title={Towards Automated Ethogramming: Cognitively-Inspired Event Segmentation for Streaming Wildlife Video Monitoring},
  author={Mounir, Ramy and Shahabaz, Ahmed and Gula, Roman and Theuerkauf, J{\"o}rn and Sarkar, Sudeep},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2023},
  publisher={Springer}
}
```


