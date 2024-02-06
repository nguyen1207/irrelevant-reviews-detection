import evaluate
import torch


def E5NN_compute_metrics(eval_pred):
    logits = torch.tensor(eval_pred.predictions)
    labels = torch.tensor(eval_pred.label_ids, dtype=torch.int32)
    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    return metrics.compute(predictions=predictions, references=labels)
