import torch
import torch.nn as nn
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
from copy import deepcopy
#from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from models.classify_head import RobertaClassificationHead



class TemporalRelationClassification(BertPreTrainedModel):
  config_class = RobertaConfig
  base_model_prefix = "roberta"

  def __init__(self, config, dataset=None):
    super(TemporalRelationClassification, self).__init__(config)

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


    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss, logits

    return logits
