# TOML-BERT
Task-Oriented Multi-level Learning based on BERT (TOML-BERT): a dual-level pretraining framework that considers both structural patterns and domain knowledge for molecular property prediction. 

![image](https://github.com/yanjing-duan/TOML-BERT/assets/123799114/6e591beb-6caa-4f91-83bb-f26c705a1649)


## Overview
* node-level pretraining: contains the codes for the masked atom prediction pre-training task.
* graph-level pretraining (regression) and graph-level pretraining (classification): contains the codes for pseudo-label prediction pre-training tasks.
* fine-tuning (regression) and fine-tuning (classification): contain the code for fine-tuning on specified tasks.
* dataset: contain the code to build datasets for pre-traing and fine-tuning.
* utils: contain the code to convert molecules to graphs.

## Implementation steps
1. Run the node-level pretraining.py file to pretrain the TOML-BERT model with large unlabeled data.
2. Run the graph-level pretraining (regression).py or graph-level pretraining (classification).py file to pretrain the model with large pseudo-labeled data.
3. Run the fine-tuning (regression).py or fine-tuning (classification).py file to fine-tune the model on experimental molecular prediction datasets.

## Requirements
* tensorflow-gpu==2.3.0
* rdkit==2022.03.2
* numpy==1.18.5
* pandas==1.5.3
* scikit-learn==1.0.2
* openbabel==3.1.1
