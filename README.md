# TopTwo
This repository contains code and the real world dataset for our NeurIPS 2022 paper: "Recovering Top Two Most Plausible Answers and Confusion Probability in Multi-Class Crowdsourcing".

## Introduction

## Datasets

We provide 6 publicly accessible datasets(**Adult2**, **Web**, **Dog**, **Flag**, **Food**, **Plot**) and **Color** dataset that we created. The datasets are contained in ./dataset folders. We provide crowd_data.txt and ground_truth.txt files for each dataset. Each line of the crowd_data.txt file consists of three numbers corresponding to (worker, task, answer). Each line of the ground_truth.txt file consists of two numbers corresponding to (task, ground_truth). In the **Color** dataset, we also provide the most confusing answer in the most_confusing_answer.txt.

## How to run the code
We provide three matlab codes in this repository : **RealExperiment.m**, **SyntheticExperiment.m**, and **DrawDistribution.m**.

For the experiment on the real world dataset, you can change the variable "dataset" at the top of **RealExperiment.m** to obtain the prediction error of each dataset. You can also get the distribution of the real world dataset using **DrawDistribution.m**.

For the synthetic experiment,  You can change the variables in **SyntheticExperiment.m** file to obtain the prediction error curve of our algorithms and the state-of-the-art methods in the various scenarios. 

## Citation
