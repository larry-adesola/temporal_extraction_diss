import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel, AdamW, BertPreTrainedModel, RobertaConfig, get_scheduler
from copy import deepcopy

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)  # Adjust input size here
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TemporalRelationClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, dataset=None, alpha=1.0):
        super(TemporalRelationClassification, self).__init__(config)

        # Initialize RoBERTa backbone
        self.alpha = alpha
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize classification head
        config_for_classification_head = deepcopy(config)
        config_for_classification_head.num_labels = len(dataset["label_mapping"])  # Assuming dataset provides label mapping
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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits

        return logits
