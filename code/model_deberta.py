import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Config
from copy import deepcopy
#from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

class DebertaClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

  def forward(self, features, **kwargs):
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x



class TemporalRelationClassificationWithDebertaPOS(DebertaV2PreTrainedModel):
  config_class = DebertaV2Config
  base_model_prefix = "deberta"

  def __init__(self, config, dataset=None, pos_vocab_size=100, pos_dim=16):
    super(TemporalRelationClassificationWithDebertaPOS, self).__init__(config)

    # Initialize DeBERTa backbone
    self.deberta = DebertaV2Model(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # POS Embedding
    self.pos_embedding = nn.Embedding(pos_vocab_size, pos_dim)


    # Initialize classification head
    config_for_classification_head = deepcopy(config)
    config_for_classification_head.num_labels = len(dataset["label_mapping"])  

    config_for_classification_head.hidden_size *= 2
    config_for_classification_head.hidden_size += 2
    config_for_classification_head.hidden_size += (pos_dim * 2)

    self.classifier = DebertaClassificationHead(config_for_classification_head)

    #time prediction classifier
    config_for_time_anchor = deepcopy(config)
    config_for_time_anchor.num_labels = 1

    self.time_anchor = DebertaClassificationHead(config_for_time_anchor)

    self.init_weights()

  def forward(self, input_ids, attention_mask, event_ix, labels=None, pos_ids=None):

    if labels is not None and not isinstance(labels, torch.Tensor):
      # Make it a tensor on the same device as input_ids
      labels = torch.tensor(labels, dtype=torch.long, device=input_ids.device)
    # Get contextual embeddings from RoBERTa
    outputs = self.deberta(input_ids, attention_mask=attention_mask)
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

    if pos_ids is not None:
      pos_1 = torch.gather(pos_ids, 1, event_1_ix).squeeze(1)  
      pos_2 = torch.gather(pos_ids, 1, event_2_ix).squeeze(1)  
      
      pos_1_emb = self.pos_embedding(pos_1)  # [batch_size, pos_dim]
      pos_2_emb = self.pos_embedding(pos_2)  # [batch_size, pos_dim]
      
      # concat
      event_pair = torch.cat([event_pair, pos_1_emb, pos_2_emb], dim=-1)

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

      # Compute loss if labels are provided
  
      loss_fct = nn.CrossEntropyLoss()
      loss += loss_fct(logits, labels)
      return loss, logits
    return logits








