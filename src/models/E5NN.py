import glob
import os

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class E5NNConfig(PretrainedConfig):
    model_type = 'E5_NN'

    def __init__(self, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class E5NN(PreTrainedModel):
    config_class = E5NNConfig

    def __init__(self, config):
        super(E5NN, self).__init__(config)
        self.num_labels = config.num_labels
        self.e5 = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        self.linear = nn.Linear(1024, config.num_labels)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        e5_outputs = self.e5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        features = e5_outputs.pooler_output
        logits = self.linear(features)
        prob = torch.softmax(logits, dim=-1)
        loss = self.cross_entropy(prob, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
