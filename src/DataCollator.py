import torch
from transformers import DataCollatorWithPadding


class E5DataCollator(DataCollatorWithPadding):

    def __call__(self, examples):
        q_prefix, d_prefix = 'q_', 'd_'

        queries = [{k[len(q_prefix):]: v for k, v in example.items() if q_prefix in k}
                   for example in examples]

        docs = [{k[len(d_prefix):]: v for k, v in example.items() if d_prefix in k}
                for example in examples]

        batch_collated = self.tokenizer.pad(
            queries + docs,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        batch_collated['labels'] = torch.tensor(
            [example['label'] for example in examples])

        return batch_collated
