import warnings
from sklearn.exceptions import UndefinedMetricWarning
from transformers import AutoModelForSequenceClassification, Trainer

import wandb
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from mbert_functions import *

# Prepare undersampled dataset
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/GermEval/GE_train_original.csv")

wandb.init(project="mBERT-GermEval")

metrics = {}

# Depending on the model we want to train (and which tokenizer we've used) we change the BERT model here
for i in range(5):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=trains_ds_bin[i], eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
    trainer.train()

    # Calculate F1 score
    predictions = trainer.predict(vals_ds_bin[i])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Log F1 score to WandB
    wandb.log({"F1 Score": f1})

    # Dump metrics into a pickle file
    metrics[i] = trainer.evaluate()
    print(metrics)
    metrics_df = pd.DataFrame.from_dict(metrics).transpose()
    metrics_df.to_csv("Logs/mBERT-GermEval_metrics.csv")


    
trainer.save_model("models/mBERT-GermEval")