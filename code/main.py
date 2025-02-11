from models.model import TemporalRelationClassification
from models.model_time import TemporalRelationClassificationWithTime
from models.model_weight_fct import TemporalRelationClassificationWithWeightedFCT
from models.model_weight_no_time import TemporalRelationClassificationWithWeightNoTime
from models.model_pos_embedding import TemporalRelationClassificationWithPOSEmbedding
from models.model_deberta import TemporalRelationClassificationWithDebertaPOS
from models.models_xlm_pos import XLMTemporalRelationClassificationWithPOSEmbedding
from transformers import RobertaTokenizerFast, AdamW, RobertaConfig, get_linear_schedule_with_warmup, DebertaV2TokenizerFast, DebertaV2Config, XLMRobertaTokenizerFast, XLMRobertaConfig
import argparse
import os
from torch.utils.data import DataLoader
import torch
from eval_tools import evaluate_model
from train import train_model
from data_utils.tokenise import contextualise_data
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
  parser = argparse.ArgumentParser(description="Train a temporal relation classification model.")
  
  # Arguments for model training
  parser.add_argument("--trainset_loc", type=str, default="../trainset-temprel.xml",
                      help="Path to the training dataset file.")
  parser.add_argument("--testset_loc", type=str, default="../testset-temprel.xml",
                      help="Path to the testing dataset file.")
  parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training.")
  parser.add_argument("--update_batch_size", type=int, default=32,
                  help="Batch size for each model update.")
  parser.add_argument("--epochs", type=int, default=30,
                      help="Number of epochs for training.")
  parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for the optimizer.")
  parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay for optimizer.")
  parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate for the model.")
  parser.add_argument("--warmup_proportion", type=float, default=0.1,
                      help="Proportion of steps for warmup in the scheduler.")
  parser.add_argument("--num_workers", type=int, default=2,
                      help="Number of workers for DataLoader.")
  parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility.")
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="Beta 1 parameters (b1, b2) for optimizer.")
  parser.add_argument("--beta2", type=float, default=0.999,
                      help="Beta 1 parameters (b1, b2) for optimizer.")
  parser.add_argument("--model", type=int, default="4",
                      help="Baseline - 0 Time prediction - 1 Weight FCT - 2 Weight FCT No Time - 3 POS Embedding - 4 Deberta & POS - 5 XLM & POS - 6")
  
  return parser.parse_args()

def main(input_args=None):

  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  # Parse arguments
  args = parse_args()
  
  #set_seed(args.seed)

  # Access arguments
  trainsetLoc = args.trainset_loc
  testsetLoc = args.testset_loc
  train_batch_size = args.batch_size
  test_batch_size = args.batch_size // 2
  num_epochs = args.epochs
  learning_rate = args.learning_rate
  num_workers = args.num_workers
  
  # Tokenizer and datasets
  if args.model == 5:
    tokeniser = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")
  elif args.model == 6:
    tokeniser = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
  else:
    tokeniser = RobertaTokenizerFast.from_pretrained("roberta-large")

  train_dataset, dev_dataset, test_dataset = contextualise_data(tokeniser, trainsetLoc, testsetLoc, args.model)
  
  train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
  test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
  dev_dataloader = DataLoader(dev_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

  
  # Model and optimizer
  config = RobertaConfig.from_pretrained("roberta-large", num_labels=4, hidden_dropout_prob=args.dropout)
  
  if args.model == 1:
    print("Time Prediction In Use")
    model = TemporalRelationClassificationWithTime.from_pretrained(
      "roberta-large", config=config,
      dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
    )
  elif args.model == 2:
    print("Weighted Cross Entropy Loss")
    model = TemporalRelationClassificationWithWeightedFCT.from_pretrained(
      "roberta-large", config=config,
      dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
    )
  elif args.model == 3:
    print("Weighted Cross Entropy Loss No Time Prediction")
    model = TemporalRelationClassificationWithWeightNoTime.from_pretrained(
      "roberta-large", config=config,
      dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
    )
  elif args.model == 4:
    print("POS Tagging")
    model = TemporalRelationClassificationWithPOSEmbedding.from_pretrained(
      "roberta-large", config=config,
      dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
    )
  elif args.model == 5:
    print("DEBERTA & POS Tagging")
    config = DebertaV2Config.from_pretrained("microsoft/deberta-v3-large", num_labels=4, hidden_dropout_prob=args.dropout)
    model = TemporalRelationClassificationWithDebertaPOS.from_pretrained(
      "microsoft/deberta-v3-large", config=config,
      dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
    )
  elif args.model == 6:
    print("XLM & POS Tagging")
    config = XLMRobertaConfig.from_pretrained("xlm-roberta-large", num_labels=4, hidden_dropout_prob=args.dropout)
    model = XLMTemporalRelationClassificationWithPOSEmbedding.from_pretrained(
      "xlm-roberta-large", config=config,
      dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
    )
  else:
    model = TemporalRelationClassification.from_pretrained(
      "roberta-large", config=config,
      dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
    )
  
  # Training
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_model(args, model, train_dataloader, dev_dataloader, device, args.model>=4)


  print(f"Test Stats")
  evaluate_model(model, test_dataloader, device, args.model>=4)


if __name__ == "__main__":
  main()


  

