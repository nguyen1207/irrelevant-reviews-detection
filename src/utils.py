import torch
from tqdm import tqdm


def get_embeddings(model, dataset):
    device = model.device
    query_embeddings = torch.empty((0, 1024), device=device)
    doc_embeddings = torch.empty((0, 1024), device=device)

    with torch.no_grad():
        for record in tqdm(dataset):
            query_embedding = model(
                input_ids=torch.tensor(
                    record['q_input_ids'], device=device).unsqueeze(0),
                attention_mask=torch.tensor(
                    record['q_attention_mask'], device=device).unsqueeze(0)
            ).pooler_output

            doc_embedding = model(
                input_ids=torch.tensor(
                    record['d_input_ids'], device=device).unsqueeze(0),
                attention_mask=torch.tensor(
                    record['d_attention_mask'], device=device).unsqueeze(0)
            ).pooler_output

            query_embeddings = torch.vstack(
                (query_embeddings, query_embedding))
            doc_embeddings = torch.vstack((doc_embeddings, doc_embedding))

    return (query_embeddings, doc_embeddings)
