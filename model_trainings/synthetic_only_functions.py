import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from datasets import load_dataset, concatenate_datasets

TRAINING_ARGS = TrainingArguments(
    output_dir='./results',
    eval_steps=1000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    save_steps=1000,
    save_total_limit=2,
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def show_confusion_matrix(conf_matrix, normalized=False, class_names=[0, 1]):
    if normalized:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="PuRd")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

def tokenize_function(examples):
    # Ensuring all are strings and replacing NaNs or Nones with empty strings
    texts = [str(text) if text is not None else "" for text in examples["text"]]
    result = tokenizer(
        texts,
        padding="max_length",
        truncation=True
    )
    return result

def compute_metrics(pred):
    logits, labels = pred.predictions, pred.label_ids
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def compute_test_metrics(pred, average):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def load_and_tokenize_training_set(filepath):
    vals_ds_bin = load_dataset('csv', data_files={'train': filepath}, split=[f'train[{k}%:{k+20}%]' for k in range(0, 100, 20)])
    trains_ds_bin = load_dataset('csv', data_files={'train': filepath}, split=[f'train[:{k}%]+train[{k+20}%:]' for k in range(0, 100, 20)])

    for index, val_ds in enumerate(vals_ds_bin):
        vals_ds_bin[index] = val_ds.map(tokenize_function, batched=True)

    for index, train_ds in enumerate(trains_ds_bin):
        trains_ds_bin[index] = train_ds.map(tokenize_function, batched=True)

    return vals_ds_bin, trains_ds_bin


def load_and_tokenize_synthetic_set(filepath):
    synth_ds_bin = load_dataset('csv', data_files={'train': filepath})

    # Accessing 'train' dataset and mapping the tokenize function
    synth_ds_bin['train'] = synth_ds_bin['train'].map(tokenize_function, batched=True)

    return synth_ds_bin


def load_predict_testset(data_files, model_path=None, trainer=None, args=TRAINING_ARGS):
    test = load_dataset('csv', data_files=data_files)
    test = test.map(tokenize_function, batched=True)

    if model_path is not None:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        trainer = Trainer(model=model, args=args)
        return trainer.predict(test["train"])
    else:
        return trainer.predict(test["train"])
