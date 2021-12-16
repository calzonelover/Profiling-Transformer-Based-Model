from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import datasets
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os

tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", model_max_length=416)

model = AutoModelForSequenceClassification.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", num_labels=5)

dataset = load_dataset("wongnai_reviews")
train_data = dataset['train'].select(range(200))
test_data = dataset['test'].select(range(20))

train_data = train_data.rename_column("review_body", "text")
train_data = train_data.rename_column("star_rating", "label")
test_data = test_data.rename_column("review_body", "text")
test_data = test_data.rename_column("star_rating", "label")


train_data = train_data.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched = True)
test_data = test_data.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched = True)

# print(train_data[0])
# exit()

# train_data.set_format('torch', columns=['input_ids', 'attention_mask','label'])
# test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# print(train_data[0])
# exit()


# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results-1gpu',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs-1gpu',          # directory for storing logs
    logging_steps=10,
    #evaluation_strategy='epoch'
)

# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)

trainer.train()
