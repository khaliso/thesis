import warnings
from sklearn.exceptions import UndefinedMetricWarning
from transformers import AutoModelForSequenceClassification, Trainer

import wandb
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import (
    RobertaTokenizer,
    AlbertForSequenceClassification,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, concatenate_datasets
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

tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

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

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/Stormfront/original/training_big.csv")
synth_ds_bin = load_and_tokenize_synthetic_set("my_datasets/Stormfront/SF_synthetic_cleaned.csv")

wandb.init(project="SF-HateBert-composite")  # Initialize W&B before the loop

metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the BERT model here
for i in range(5):
    model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT", num_labels=2)

    # Append synth_ds_bin onto trains_ds_bin[i]
    combined_ds = concatenate_datasets([trains_ds_bin[i], synth_ds_bin['train']])
    combined_ds = combined_ds.shuffle(seed=42)

    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=combined_ds, eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
    wandb.watch(model)
    trainer.train()

    # Calculate F1 score
    predictions = trainer.predict(vals_ds_bin[i])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = vals_ds_bin[i]['labels']
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Log F1 score to WandB
    wandb.log({"F1 Score": f1})

    # Dump metrics into a pickle file
    metrics[i] = trainer.evaluate()
    print(metrics)
    metrics_df = pd.DataFrame.from_dict(metrics).transpose()
    metrics_df.to_csv("Logs/Stormfront_hatebert_comp_metrics.csv")

trainer.save_model("models/SF_hatebert_composite_staged")

TODO: Founta. WHich classifier? Stormfront also has weird values!
'''
tokenizer

Davidson
SemEval (PCL)
'''

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/Davidson/davidson_original_train.csv")
synth_ds_bin = load_and_tokenize_synthetic_set("my_datasets/Davidson/classified_davidson_cleaned.csv")


metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the model here
for i in range(5):
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Append synth_ds_bin onto trains_ds_bin[i]
    combined_ds = concatenate_datasets([trains_ds_bin[i], synth_ds_bin['train']])
    combined_ds = combined_ds.shuffle(seed=42)

    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=combined_ds, eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
    wandb.watch(model)
    trainer.train()

    # Calculate F1 score
    predictions = trainer.predict(vals_ds_bin[i])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = vals_ds_bin[i]['labels']
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Log F1 score to WandB
    wandb.log({"F1 Score": f1})

    # Dump metrics into a pickle file
    metrics[i] = {"F1 Score": f1}
    with open(f"Logs/Davidson_RoBERTa_composite_metrics_fold_{i}.pickle", "wb") as file:
        pickle.dump(metrics[i], file)

trainer.save_model("models/Davidson_RoBERTa_composite_staged")

'''
tokenizer
'''

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/Davidson/davidson_original_train.csv")
synth_ds_bin = load_and_tokenize_synthetic_set("my_datasets/Davidson/classified_davidson_cleaned.csv")


metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the BERT model here
for i in range(5):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    # Append synth_ds_bin onto trains_ds_bin[i]
    combined_ds = concatenate_datasets([trains_ds_bin[i], synth_ds_bin['train']])
    combined_ds = combined_ds.shuffle(seed=42)

    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=combined_ds, eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
    wandb.watch(model)
    trainer.train()

    # Calculate F1 score
    predictions = trainer.predict(vals_ds_bin[i])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = vals_ds_bin[i]['labels']
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Log F1 score to WandB
    wandb.log({"F1 Score": f1})

    # Dump metrics into a pickle file
    metrics[i] = {"F1 Score": f1}
    with open(f"Logs/Davidson_BERT_composite_metrics_fold_{i}.pickle", "wb") as file:
        pickle.dump(metrics[i], file)

trainer.save_model("models/Davidson_bert_composite_staged")

'''
Tokenizer

GermEval
HatEval 
'''

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/GermEval/GE_train_original.csv")
synth_ds_bin = load_and_tokenize_synthetic_set("my_datasets/GermEval/synthetic/composite_GE_dataset.csv")


wandb.init(project="GermEval-composite-mBERT")  # Initialize W&B before the loop

metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the BERT model here
for i in range(5):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

    # Append synth_ds_bin onto trains_ds_bin[i]
    combined_ds = concatenate_datasets([trains_ds_bin[i], synth_ds_bin['train']])
    combined_ds = combined_ds.shuffle(seed=42)

    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=combined_ds, eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
    wandb.watch(model)
    trainer.train()

    # Calculate F1 score
    predictions = trainer.predict(vals_ds_bin[i])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = vals_ds_bin[i]['labels']
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Log F1 score to WandB
    wandb.log({"F1 Score": f1})

    # Dump metrics into a pickle file
    metrics[i] = {"F1 Score": f1}
    with open(f"Logs/GermEval/mBERT_composite_metrics_fold_{i}.pickle", "wb") as file:
        pickle.dump(metrics[i], file)

trainer.save_model("models/GermEval_mbert_composite")

'''
Tokenizer
'''

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/Davidson/davidson_original_train.csv")
synth_ds_bin = load_and_tokenize_synthetic_set("my_datasets/Davidson/classified_davidson_cleaned.csv")


wandb.init(project="Davidson-composite-roberta-staged")  # Initialize W&B before the loop

metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the model here
for i in range(5):
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Append synth_ds_bin onto trains_ds_bin[i]
    combined_ds = concatenate_datasets([trains_ds_bin[i], synth_ds_bin['train']])
    combined_ds = combined_ds.shuffle(seed=42)

    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=combined_ds, eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
    wandb.watch(model)
    trainer.train()

    # Calculate F1 score
    predictions = trainer.predict(vals_ds_bin[i])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = vals_ds_bin[i]['labels']
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Log F1 score to WandB
    wandb.log({"F1 Score": f1})

    # Dump metrics into a pickle file
    metrics[i] = {"F1 Score": f1}
    with open(f"Logs/Davidson_RoBERTa_composite_metrics_fold_{i}.pickle", "wb") as file:
        pickle.dump(metrics[i], file)

trainer.save_model("models/Davidson_RoBERTa_composite_staged")