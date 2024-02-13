import argparse

import torch
from transformers import AutoTokenizer, TrainingArguments

from DataCollator import E5DataCollator
from DataLoader import E5DataLoader
from metrics import E5NN_compute_metrics
from models.E5NN import E5NN
from Trainer import E5Trainer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file", help='CSV Data file for finetuning', dest='data_file', required=True
    )
    parser.add_argument(
        "--epoch", help='Epoch (default: 2)', default=2, dest='epoch'
    )
    parser.add_argument(
        "--batch-size", help='Batch size (default: 2)', default=2, dest='batch_size'
    )

    args = parser.parse_args()
    epoch, batch_size, data_file = \
        int(args.epoch), int(args.batch_size), args.data_file

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    data_loader = E5DataLoader(tokenizer, data_file)
    train_data = data_loader.train_dataset
    eval_data = data_loader.eval_dataset

    model = E5NN()
    print(model)

    data_collator = E5DataCollator(
        tokenizer=tokenizer,
        max_length=512
    )

    training_args = TrainingArguments(
        output_dir='saved_models/e5nn',
        evaluation_strategy='steps',
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy='steps',
        save_steps=0.2,
        logging_steps=0.2,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=['labels']
    )

    trainer = E5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=E5NN_compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model('./saved_models/e5nn')


if __name__ == "__main__":
    main()
