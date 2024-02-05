from datasets import load_dataset


class E5DataLoader:
    def __init__(self, tokenizer, data_file):
        self.tokenizer = tokenizer
        dataset = load_dataset(
            "csv", data_files=data_file, split='train')
        dataset = dataset.class_encode_column('label')
        dataset = dataset.train_test_split(
            test_size=0.2, stratify_by_column='label')
        self.train_dataset, self.eval_dataset = dataset['train'], dataset['test']
        self.train_dataset.set_transform(self._transform)
        self.eval_dataset.set_transform(self._transform)

    def _transform(self, examples):
        docs = [f'passage: {doc}' for doc in examples['description']]
        queries = [f'query: {query}' for query in examples['comment']]

        assert len(docs) == len(queries)
        assert len(queries) == len(examples['label'])

        query_batch_dict = self.tokenizer(queries,
                                          max_length=512,
                                          truncation=True,
                                          )

        doc_batch_dict = self.tokenizer(docs,
                                        max_length=512,
                                        truncation=True,
                                        )

        merged_batch_dict = {f'q_{k}': v for k, v in query_batch_dict.items()}
        for k, v in doc_batch_dict.items():
            k = f'd_{k}'
            merged_batch_dict[k] = v

        merged_batch_dict['label'] = examples['label']

        return merged_batch_dict
