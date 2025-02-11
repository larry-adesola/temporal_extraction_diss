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


class TemporalRelationClassificationWithWeightNoTime(BertPreTrainedModel):
  config_class = RobertaConfig
  base_model_prefix = "roberta"

  def __init__(self, config, dataset=None):
    super(TemporalRelationClassificationWithWeightNoTime, self).__init__(config)

    # Initialise RoBERTa backbone
    self.roberta = RobertaModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # Initialise classification head
    config_for_classification_head = deepcopy(config)
    config_for_classification_head.num_labels = len(dataset["label_mapping"])  
    config_for_classification_head.hidden_size *= 2
    self.classifier = RobertaClassificationHead(config_for_classification_head)

    self.init_weights()

  def forward(self, input_ids, attention_mask, event_ix, labels=None):
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

    # Pass through classification head
    logits = self.classifier(event_pair.unsqueeze(dim=1))
    logits = logits.squeeze(dim=1)

    # Compute loss if labels are provided
    if labels is not None:

      label_counts = torch.tensor([count_BEFORE, count_AFTER, count_EQUAL, count_VAGUE], dtype=torch.float)

      weights = 1.0 / label_counts
      weights = weights / weights.sum()


      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      weights = weights.to(device)

      # print(weights)
      # print(labels)
      # print(logits)


      loss_fct = nn.CrossEntropyLoss(weight=weights)


      loss = loss_fct(logits, labels)
      return loss, logits

    return logits
