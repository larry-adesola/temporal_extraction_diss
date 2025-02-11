import torch
import torch.nn as nn
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
from copy import deepcopy
#from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

count_BEFORE = 5483
count_AFTER  = 3819
count_EQUAL  = 359
count_VAGUE  = 1227

class RobertaClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    classifier_dropout = (
        config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )
    self.dropout = nn.Dropout(classifier_dropout)
    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

  def forward(self, features, **kwargs):
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x



class TemporalRelationClassificationWithWeightedFCT(BertPreTrainedModel):
  config_class = RobertaConfig
  base_model_prefix = "roberta"

  def __init__(self, config, dataset=None):
    super(TemporalRelationClassificationWithWeightedFCT, self).__init__(config)

    # Initialise RoBERTa backbone
    self.roberta = RobertaModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # Initialise classification head
    config_for_classification_head = deepcopy(config)
    config_for_classification_head.num_labels = len(dataset["label_mapping"])  
    config_for_classification_head.hidden_size *= 2
    config_for_classification_head.hidden_size += 2

    self.classifier = RobertaClassificationHead(config_for_classification_head)

    #time prediction classifier
    config_for_time_anchor = deepcopy(config)
    config_for_time_anchor.num_labels = 1

    self.time_anchor = RobertaClassificationHead(config_for_time_anchor)

    self.init_weights()

  def forward(self, input_ids, attention_mask, event_ix, labels=None):

    if labels is not None and not isinstance(labels, torch.Tensor):
      # Make it a tensor on the same device as input_ids
      labels = torch.tensor(labels, dtype=torch.long, device=input_ids.device)
    # Get contextual embeddings from RoBERTa
    outputs = self.roberta(input_ids, attention_mask=attention_mask)
    sequence_output = outputs[0]  # [batch_size, sequence_length, hidden_size]

    # Extract event embeddings
    event_1_ix, event_2_ix = event_ix.split(1, dim=-1)
    event_1 = torch.gather(sequence_output, dim=1,
                            index=event_1_ix.expand(sequence_output.size(0), sequence_output.size(2)).unsqueeze(dim=1))
    event_2 = torch.gather(sequence_output, dim=1,
                            index=event_2_ix.expand(sequence_output.size(0), sequence_output.size(2)).unsqueeze(dim=1))
    event_pair = torch.cat([event_1.squeeze(dim=1), event_2.squeeze(dim=1)], dim=1)

    event_1_time = torch.tanh(self.time_anchor(event_1).squeeze())
    event_2_time = torch.tanh(self.time_anchor(event_2).squeeze())

    augmented_repr = torch.cat([event_1_time.unsqueeze(dim=-1),
                                    event_2_time.unsqueeze(dim=-1)], dim=-1).view(-1 ,2)
    event_pair = torch.cat([event_pair, augmented_repr], dim=-1)

    # Pass through classification head
    logits = self.classifier(event_pair.unsqueeze(dim=1))
    logits = logits.squeeze(dim=1)
    if labels is not None:
      #to work out loss of time prediction in the cases of before, after, equal
      loss = 0.
      relative = event_1_time - event_2_time
      mask_before = (labels == 0).float()
      relative_sum_before = ((1 + relative) > 0).float() * (1 + relative)
      loss += torch.sum(relative_sum_before * mask_before)
      mask_after = (labels == 1).float()
      relative_sum_after = ((1 - relative) > 0).float() * (1 - relative)
      loss += torch.sum(relative_sum_after * mask_after)
      mask_equal = (labels == 2).float()
      loss += torch.sum(torch.abs(relative * mask_equal))
      loss /= sequence_output.size(0)
      


      label_counts = torch.tensor([count_BEFORE, count_AFTER, count_EQUAL, count_VAGUE], dtype=torch.float)

      weights = 1.0 / label_counts
      weights = weights / weights.sum()

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      weights = weights.to(device)


      loss_fct = nn.CrossEntropyLoss(weight=weights)

      loss += loss_fct(logits, labels)
      return loss, logits
    
    return logits









