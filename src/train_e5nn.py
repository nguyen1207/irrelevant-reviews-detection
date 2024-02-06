import argparse

import torch
from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, TrainingArguments

from DataCollator import E5DataCollator
from DataLoader import E5DataLoader
from metrics import E5NN_compute_metrics
from models.E5NN import E5NN, E5NNConfig
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
    parser.add_argument(
        "--lora-r", help='Rank of matrices (default: 8)', default=8, dest='lora_r'
    )
    parser.add_argument(
        "--lora-alpha", help='Lora alpha (default: 32)', default=32, dest='lora_alpha'
    )
    parser.add_argument(
        "--load-8-bit", help='8 bit quantization (default: False)', default=False, dest='load8bit'
    )

    args = parser.parse_args()
    epoch, batch_size, load8bit, \
        lora_r, lora_alpha, data_file = \
        int(args.epoch), int(args.batch_size), args.load8bit, \
        int(args.lora_r), int(args.lora_alpha), args.data_file

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    data_loader = E5DataLoader(tokenizer, data_file)
    train_data = data_loader.train_dataset
    eval_data = data_loader.eval_dataset

    device = 'cuda' if torch.cuda.is_available() \
        else 'mps' if torch.backends.mps.is_available() \
        else 'cpu'

    if load8bit:
        if device != 'cuda':
            print('CUDA GPU not found for quantization')
            exit(1)
        loftq_config = LoftQConfig(loftq_bits=8)

    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS if load8bit else None,
                             init_lora_weights="loftq" if load8bit else "gaussian",
                             loftq_config=loftq_config if load8bit else dict(),
                             target_modules=[
                                 'query',
                                 'key'
                             ],
                             inference_mode=False,
                             r=lora_r,
                             lora_alpha=lora_alpha,
                             lora_dropout=0.1
                             )

    config = E5NNConfig()
    model = E5NN(config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
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


if __name__ == "__main__":
    main()
