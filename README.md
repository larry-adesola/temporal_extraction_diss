Temporal Relation Classification

This repository contains code for training and evaluating temporal relation classification models, 
including variants with POS embeddings, time prediction, and support for multiple transformer 
backbones like RoBERTa, DeBERTa, and XLM-RoBERTa. The models are designed to classify temporal 
relations such as BEFORE, AFTER, EQUAL, and VAGUE between pairs of events in text.

Make sure you have the following dependencies installed:

python3 needed

pip install torch transformers scikit-learn matplotlib

python -m spacy download pt_core_news_sm

Example of train and run model:

cd into code and

python main.py --epochs 10 --model 1

List of args:

usage: main.py [-h] [--trainset_loc TRAINSET_LOC] [--testset_loc TESTSET_LOC]
               [--batch_size BATCH_SIZE]
               [--update_batch_size UPDATE_BATCH_SIZE] [--epochs EPOCHS]
               [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
               [--dropout DROPOUT] [--warmup_proportion WARMUP_PROPORTION]
               [--num_workers NUM_WORKERS] [--seed SEED] [--beta1 BETA1]
               [--beta2 BETA2] [--model MODEL]

Train a temporal relation classification model.

options:
  -h, --help            show this help message and exit
  --trainset_loc TRAINSET_LOC
                        Path to the training dataset file.
  --testset_loc TESTSET_LOC
                        Path to the testing dataset file.
  --batch_size BATCH_SIZE
                        Batch size for training.
  --update_batch_size UPDATE_BATCH_SIZE
                        Batch size for each model update.
  --epochs EPOCHS       Number of epochs for training.
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer.
  --weight_decay WEIGHT_DECAY
                        Weight decay for optimizer.
  --dropout DROPOUT     Dropout rate for the model.
  --warmup_proportion WARMUP_PROPORTION
                        Proportion of steps for warmup in the scheduler.
  --num_workers NUM_WORKERS
                        Number of workers for DataLoader.
  --seed SEED           Random seed for reproducibility.
  --beta1 BETA1         Beta 1 parameters (b1, b2) for optimizer.
  --beta2 BETA2         Beta 1 parameters (b1, b2) for optimizer.
  --model MODEL         Baseline - 0 Time prediction - 1 Weight FCT - 2 Weight
                        FCT No Time - 3 POS Embedding - 4 Deberta & POS - 5
                        XLM & POS - 6




