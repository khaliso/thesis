from synthetic_only_functions import *
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from transformers import AutoModelForSequenceClassification, Trainer
import wandb
import pickle
import numpy as np
from sklearn.metrics import f1_score

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/dontpatronizeme_v1.4/train_data_undersampled.csv")
synth_ds_bin = load_and_tokenize_synthetic_set("my_datasets/dontpatronizeme_v1.4/synthetic/synthetic_pcl_undersampled.csv")

wandb.init(project="Bert-synthetic-PCL-us")  # Initialize W&B before the loop

'''
CAREFUL TO GET THE RIGHT TOKENIZER!!!

GermEval:   bert-base-multilingual-cased
Founta:     bert-base-uncased
Davidson:   GroNLP/hateBERT
HatEval:    GroNLP/hateBERT
Stormfront: GroNLP/hateBERT

PCL:      roberta-base (and some other changes)
'''

metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the BERT model here
for i in range(5):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=synth_ds_bin['train'], eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
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
    metrics_df.to_csv("Logs/Bert-synthetic-PCL-us-metrics.csv")

trainer.save_model("models/Bert-synthetic-PCL-us")