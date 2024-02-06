import evaluate
import torch


def E5NN_compute_metrics(eval_pred):
    logits = torch.tensor(eval_pred.predictions)
    labels = torch.tensor(eval_pred.label_ids, dtype=torch.int32)
    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    return metrics.compute(predictions=predictions, references=labels)


def E5_compute_metrics(eval_pred):
    query_embeddings = torch.from_numpy(eval_pred.predictions[0])
    doc_embeddings = torch.from_numpy(eval_pred.predictions[1])
    assert len(query_embeddings) == len(doc_embeddings)

    similarity = torch.cosine_similarity(query_embeddings, doc_embeddings)
    threshold = 0.7
    predictions = torch.where(similarity > threshold, 1, 0)

    labels = torch.tensor(eval_pred.label_ids, dtype=int)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    return metrics.compute(predictions=predictions, references=labels)
