from transformers import DataCollatorWithPadding


class E5DataCollator(DataCollatorWithPadding):

    def __call__(self, examples):
        batch_dict = {
            'input_ids': [example['input_ids'] for example in examples],
            'attention_mask': [example['attention_mask'] for example in examples],
            'labels': [int(example['label']) for example in examples]
        }

        collated_batch_dict = self.tokenizer.pad(
            batch_dict,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        return collated_batch_dict
