from transformers import Trainer


class E5Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(E5Trainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
