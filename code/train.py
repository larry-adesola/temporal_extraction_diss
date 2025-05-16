import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List
from math import ceil
from eval_tools import evaluate_model




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

    train_losses = []
    train_accuracies = []

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

      epoch_loss = total_loss / len(train_dataloader)
      epoch_acc = correct / total

      train_losses.append(epoch_loss)
      train_accuracies.append(epoch_acc)
      print(f"Train Loss: {total_loss / len(train_dataloader):.4f}, Train Accuracy: {correct / total:.4f}")

      # Evaluate on dev set
      evaluate_model(model, dev_dataloader, device, pos_enabled)
    return train_losses, train_accuracies
