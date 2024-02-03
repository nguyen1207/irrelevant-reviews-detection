from datasets import load_dataset


class E5DataLoader:
    def __init__(self, tokenizer, data_files):
        self.tokenizer = tokenizer
        dataset = load_dataset("csv", data_files=data_files, split='train')
        dataset = dataset.class_encode_column('label')
        dataset = dataset.train_test_split(
            test_size=0.2, stratify_by_column='label')
        self.train_dataset, self.eval_dataset = dataset['train'], dataset['test']
        self.train_dataset.set_transform(self._transform)
        self.eval_dataset.set_transform(self._transform)

    def _transform(self, examples):
        docs = [f'passage: {doc}' for doc in examples['description']]
        queries = [f'query: {query}' for query in examples['comment']]

        batch_dict = self.tokenizer(queries,
                                    text_pair=docs,
                                    max_length=512,
                                    truncation=True,
                                    )

        batch_dict['label'] = examples['label']

        return batch_dict
