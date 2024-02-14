import evaluate
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

from utils import get_embeddings


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


def eval_clf(clf, embedding_model, dataset, cm_title):
    query_embeddings_eval, doc_embeddings_eval = get_embeddings(
        embedding_model, dataset)

    np_query_embeddings_eval = query_embeddings_eval.cpu().detach().numpy()
    np_doc_embeddings_eval = doc_embeddings_eval.cpu().detach().numpy()

    X_eval = np_query_embeddings_eval + np_doc_embeddings_eval
    y_eval = dataset[:]['label']
    y_pred = clf.predict(X_eval)

    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                  'Irrelevant', 'Relevant'])

    print(classification_report(y_eval, y_pred))
    disp.plot()
    disp.ax_.set_title(cm_title)
    plt.show()
