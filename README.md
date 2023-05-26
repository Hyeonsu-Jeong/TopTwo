# TopTwo
This repository contains code and the real world dataset for our ICML paper: "Recovering Top-Two Answers and Confusion Probability in Multi-Choice Crowdsourcing".

## Introduction

## Datasets

We provide 5 publicly accessible datasets(**Web**, **Dog**, **Rte**, **Trec**, **Bluebird**) and **Color** dataset that we created. The datasets are contained in ./Datasets folders. We provide data.txt and truth.txt files for each dataset. Each line of the crowd_data.txt file consists of three numbers corresponding to (task, worker, answer), and each line of the ground_truth file consists of two numbers corresponding to (task, ground_truth). In the **Color** dataset, we also provide the most confusing answer in the color_conf.txt.

## How to run the code
We provide three matlab codes in this repository : **RealExperiment.m**, **SyntheticExperiment.m**, **DrawDistribution_pair.m**, and **DrawDistribution_full.m**.

For the experiment on the real world dataset, you can change the variable "dataset" at the top of **RealExperiment.m** to obtain the prediction error of each dataset. 

For the synthetic experiment,  You can change the variables in **SyntheticExperiment.m** file to obtain the prediction error curve of our algorithms and the state-of-the-art methods in the various scenarios. 

You can obtain the graph in the main text by running **DrawDistribution_pair.m**, and **DrawDistribution_full.m**.

## CIFAR10H

We provide simple python codes for evaluate the neural network training using hard/top2/full label. 

### Prerequisites
- Python 3.6
- PyTorch 1.12.1
- CUDA 11.6

### Training Examples
- training ResNet with hard label:
```
python main.py --lr 0.1 --type full --model resnet
```
- training vgg with top2 label:
```
python main.py --lr 0.1 --type top2 --model vgg
```

## Citation