from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class E5Output(ModelOutput):
    query_embeddings: Optional[torch.Tensor] = None
    doc_embeddings: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class E5Config(PretrainedConfig):
    model_type = 'E5'

    def __init__(self, num_labels=2, load_in_8bit=False, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.load_in_8bit = load_in_8bit


class E5(PreTrainedModel):
    config_class = E5Config

    def __init__(self, config):
        super(E5, self).__init__(config)
        self.num_labels = config.num_labels
        self.e5 = AutoModel.from_pretrained('intfloat/multilingual-e5-large',
                                            load_in_8bit=config.load_in_8bit
                                            )
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        e5_outputs = self.e5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embeddings = e5_outputs.pooler_output
        loss = self.cosine_loss(input1=embeddings[embeddings.shape[0]//2:],
                                input2=embeddings[:embeddings.shape[0]//2],
                                target=labels
                                )
        return E5Output(loss=loss,
                        query_embeddings=embeddings[:embeddings.shape[0]//2],
                        doc_embeddings=embeddings[embeddings.shape[0]//2:],
                        labels=labels
                        )
