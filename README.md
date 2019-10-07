# Grace Hopper 2019 Workshop Material
# DS717: Prototype to Production: How to Scale your Deep Learning Model

## Introduction
This repository provides the contents of a workshop given at Grace Hopper 2019.  These examples focus on scaling performance while keeping convergence consistent on a sample Deep Learning Model, [NCF](https://arxiv.org/abs/1708.05031) using a 1x V100 16G GPU.

Refer to [Slides](https://github.com/swethmandava/scaleDL_ghc19/GHC-19.pdf) for a brief overview.

## Quick Start Guide

To run the jupyter notebook, perform the following steps using the default parameters of the NCF model.

1. Clone the repository.

```bash
git clone https://github.com/swethmandava/scaleDL_ghc19.git
```

2. Build the PyTorch NGC container.

```bash
bash scripts/docker/build.sh
```

3. Download and preprocess the dataset.

This repository provides scripts to download, verify and extract the [ML-20m dataset](https://grouplens.org/datasets/movielens/20m/).

To download, verify, and extract the required datasets, run:

```bash
bash scripts/data/e2e_dataset.sh
```

The script launches a Docker container with the current directory mounted and downloads the datasets to a `data/` folder on the host.


5. Start an interactive session in the NGC container to run the hands on workshop.

After you build the container image and download the data, you can start an interactive CLI session as follows:

```bash
bash scripts/docker/launch.sh
```

In your web browser, open the jupyter notebook by following the instructions given in your terminal. For example, go to `127.0.0.1:8888` and use the given token. Select `ncf.ipynb` and run each cell.

## Release notes

- This repository is meant to be a learning tool to understand various computational and convergence tricks to scale your deep learning model. Refer to [NCF PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) to achieve state of the art accuracy and performance.

- This repository is not maintained. For most up-to-date Deep Learning models achieving best performance and convergence, check out [NVIDIA's Deep Learning Examples](NVIDIA/DeepLearningExamples).

## References

- Micikevicius, S. Narang, J. Alben, G. F. Diamos,E. Elsen, D. Garcia, B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, and H. Wu. Mixed precision training. CoRR, abs/1710.03740, 2017
- Yang You, Igor Gitman, Boris Ginsburg. Large Batch Training of Convolutional Networks. arXiv:1708.03888
- Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le. Don't Decay the Learning Rate, Increase the Batch Size. arXiv:1711.00489
- Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, angqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet n 1 hour. arXiv preprint arXiv:1706.02677, 2017.
- [LARS Implementation](https://github.com/noahgolmant/pytorch-lars)