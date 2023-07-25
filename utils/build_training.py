from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate

class CustomTrainer(Trainer):
    """custom trainer based on original trainer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        # implement custom logic here
        custom_loss = ...
        return custom_loss