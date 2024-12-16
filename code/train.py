from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import RobertaTokenizerFast, RobertaModel, AdamW, BertPreTrainedModel, RobertaConfig, get_scheduler
from typing import List, Optional
from data import temprel_set
from model import TemporalRelationClassification
import random
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a temporal relation classification model.")
    
    # Arguments for model training
    parser.add_argument("--trainset_loc", type=str, default="../trainset-temprel.xml",
                        help="Path to the training dataset file.")
    parser.add_argument("--testset_loc", type=str, default="../testset-temprel.xml",
                        help="Path to the testing dataset file.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
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
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, event_ix, labels = (item.to(device) for item in batch)
            logits = model(input_ids, attention_mask, event_ix)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    print(classification_report(all_labels, all_predictions, target_names=["BEFORE", "AFTER", "EQUAL", "VAGUE"]))

def contextualise_data(tokeniser, trainsetLoc, testsetLoc):

    traindevset = temprel_set(trainsetLoc)
    traindev_tensorset = traindevset.to_tensor(tokenizer=tokeniser)
    train_idx = list(range(len(traindev_tensorset)-1852))
    dev_idx = list(range(len(traindev_tensorset)-1852, len(traindev_tensorset)))
    train_tensorset = Subset(traindev_tensorset, train_idx)
    dev_tensorset = Subset(traindev_tensorset, dev_idx) #Last 21 docs

    testset = temprel_set(testsetLoc)
    test_tensorset = testset.to_tensor(tokenizer=tokeniser)
    return train_tensorset, dev_tensorset, test_tensorset


def train_model(model, train_dataloader, dev_dataloader, optimizer, scheduler, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in train_dataloader:
            input_ids, attention_mask, event_ix, labels = (item.to(device) for item in batch)

            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, event_ix, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {total_loss / len(train_dataloader):.4f}, Train Accuracy: {correct / total:.4f}")

        # Evaluate on dev set
        evaluate_model(model, dev_dataloader, device)


def main(input_args=None):
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
    tokeniser = RobertaTokenizerFast.from_pretrained("roberta-large")
    train_dataset, dev_dataset, test_dataset = contextualise_data(tokeniser, trainsetLoc, testsetLoc)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    dev_dataloader = DataLoader(dev_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    # Model and optimizer
    config = RobertaConfig.from_pretrained("roberta-large", num_labels=4, hidden_dropout_prob=args.dropout)
    model = TemporalRelationClassification.from_pretrained(
        "roberta-large", config=config,
        dataset={"label_mapping": {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}},
        alpha=1.0
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)

    # Calculate the number of warmup steps
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)

    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_dataloader, dev_dataloader, optimizer, scheduler, device, num_epochs=num_epochs)


    print(f"Test Stats")
    evaluate_model(model, test_dataloader, device)

if __name__ == "__main__":
    main()
