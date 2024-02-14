from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class E5NNOutput(SequenceClassifierOutput):
    labels: Optional[torch.Tensor] = None


class E5NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e5 = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            features = self.e5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            ).pooler_output

        sep = features.shape[0] // 2
        features = features[:sep] + features[sep:]

        logits = self.linear_relu_stack(features)
        loss = None
        if labels is not None:
            prob = self.softmax(logits)
            loss = self.cross_entropy(prob, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
