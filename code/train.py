from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import RobertaTokenizerFast, AdamW, RobertaConfig, get_linear_schedule_with_warmup, DebertaV2TokenizerFast, DebertaV2Config, XLMRobertaTokenizerFast, XLMRobertaConfig
from typing import List
from data import temprel_set
from models.model import TemporalRelationClassification
from models.model_time import TemporalRelationClassificationWithTime
from models.model_weight_fct import TemporalRelationClassificationWithWeightedFCT
from models.model_weight_no_time import TemporalRelationClassificationWithWeightNoTime
from models.model_pos_embedding import TemporalRelationClassificationWithPOSEmbedding
from models.model_deberta import TemporalRelationClassificationWithDebertaPOS
from models.models_xlm_pos import XLMTemporalRelationClassificationWithPOSEmbedding
import numpy as np
import argparse
import random
from math import ceil
import os





def parse_args():
    parser = argparse.ArgumentParser(description="Train a temporal relation classification model.")
    
    # Arguments for model training
    parser.add_argument("--trainset_loc", type=str, default="../train_pt.xml",
                        help="Path to the training dataset file.")
    parser.add_argument("--testset_loc", type=str, default="../test_pt.xml",
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
    parser.add_argument("--model", type=int, default="6",
                        help="Baseline - 0 Time prediction - 1 Weight FCT - 2 Weight FCT No Time - 3 POS Embedding - 4 Deberta & POS - 5 XLM & POS - 6")
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def calc_f1(predicted_labels, all_labels):
    confusion = np.zeros((4, 4))
    for i in range(len(predicted_labels)):
        confusion[all_labels[i]][predicted_labels[i]] += 1

    acc = 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion)
    true_positive = 0
    for i in range(4-1):
        true_positive += confusion[i][i]
    prec = true_positive/(np.sum(confusion)-np.sum(confusion,axis=0)[-1])
    rec = true_positive/(np.sum(confusion)-np.sum(confusion[-1][:]))
    f1 = 2*prec*rec / (rec+prec)

    return acc, prec, rec, f1, confusion

def evaluate_model(model, dataloader, device, pos_enabled):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
          if pos_enabled:
            input_ids, attention_mask, event_ix, labels, pos_id = (item.to(device) for item in batch)
            logits = model(input_ids, attention_mask, event_ix, pos_ids=pos_id)
          else:
            input_ids, attention_mask, event_ix, labels = (item.to(device) for item in batch)
            logits = model(input_ids, attention_mask, event_ix)

          predictions = torch.argmax(logits, dim=1)

          all_labels.extend(labels.cpu().numpy())
          all_predictions.extend(predictions.cpu().numpy())

    print(classification_report(all_labels, all_predictions, target_names=["BEFORE", "AFTER", "EQUAL", "VAGUE"]))

    acc, prec, rec, f1, confusion = calc_f1(all_predictions, all_labels)
    
    print(f"Acc={acc}, Precision={prec}, Recall={rec}, F1={f1}")
    print(f"Confusion={confusion}")



def contextualise_data(tokeniser, trainsetLoc, testsetLoc, model):

    traindevset = temprel_set(trainsetLoc)
    traindev_tensorset = traindevset.to_tensor(tokenizer=tokeniser, pos_enabled=model >= 4)
    if model == 6:
      total_len = len(traindev_tensorset)
      split_point = int(0.9 * total_len)  # 90% for training, 10% for dev
      train_idx = list(range(split_point))
      dev_idx = list(range(split_point, total_len))
    else:
      train_idx = list(range(len(traindev_tensorset)-1852))
      dev_idx = list(range(len(traindev_tensorset)-1852, len(traindev_tensorset)))
    train_tensorset = Subset(traindev_tensorset, train_idx)
    dev_tensorset = Subset(traindev_tensorset, dev_idx) #Last 21 docs

    testset = temprel_set(testsetLoc)
    test_tensorset = testset.to_tensor(tokenizer=tokeniser, pos_enabled=model >= 4)
    return train_tensorset, dev_tensorset, test_tensorset


def train_model(args, model, train_dataloader, dev_dataloader, device, pos_enabled):
    
    num_training_steps_per_epoch = ceil(len(train_dataloader.dataset)/float(args.update_batch_size))
    num_training_steps = args.epochs * num_training_steps_per_epoch


    params_to_optimise = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params_to_optimise if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in params_to_optimise if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]


    #optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(args.beta1, args.beta2))

    # Calculate the number of warmup steps
    num_warmup_steps = ceil(num_training_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    update_per_batch = args.update_batch_size // args.batch_size
    model.to(device)


    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        print(f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(train_dataloader):
          
          if pos_enabled:
            input_ids, attention_mask, event_ix, labels, pos_id = (item.to(device) for item in batch)
            loss, logits = model(input_ids, attention_mask, event_ix, labels, pos_id)
          
          else:
            input_ids, attention_mask, event_ix, labels = (item.to(device) for item in batch)
            loss, logits = model(input_ids, attention_mask, event_ix, labels)


          loss /= update_per_batch
          loss.backward()
          
          if (i+1) % update_per_batch == 0 or (i+1) == len(train_dataloader):
              # global_loss.backward()

              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

              optimizer.step()
              scheduler.step()
              model.zero_grad()

          # optimizer.step()
          # scheduler.step()

          total_loss += loss.item()
          predictions = torch.argmax(logits, dim=1)
          correct += (predictions == labels).sum().item()
          total += labels.size(0)

        print(f"Train Loss: {total_loss / len(train_dataloader):.4f}, Train Accuracy: {correct / total:.4f}")

        # Evaluate on dev set
        evaluate_model(model, dev_dataloader, device, pos_enabled)


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
