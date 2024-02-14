import argparse
import pickle
from os import name

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from DataLoader import E5DataLoader
from metrics import eval_clf
from utils import get_embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file", help='CSV Data file for finetuning', dest='data_file', required=True
    )

    args = parser.parse_args()
    data_file = args.data_file

    device = 'cuda' if torch.cuda.is_available() \
        else 'mps' if torch.backends.mps.is_available() \
        else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    dataloader = E5DataLoader(tokenizer, data_file=data_file)
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    model.to(device)

    query_embeddings_train, doc_embeddings_train = get_embeddings(
        model=model,
        dataset=dataloader.train_dataset
    )

    np_query_embeddings_train = query_embeddings_train.cpu().detach().numpy()
    np_doc_embeddings_train = doc_embeddings_train.cpu().detach().numpy()
    X_train = np_query_embeddings_train + np_doc_embeddings_train
    y_train = dataloader.train_dataset[:]['label']

    clf = RandomForestClassifier()
    clf.fit(
        X_train,
        y_train,
    )

    eval_clf(clf, model, dataloader.eval_dataset,
             cm_title='E5 with Random Forest classifier Confusion Matrix on eval dataset')

    with open('./saved_models/rf/rf.pkl', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()
