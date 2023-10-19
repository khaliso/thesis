from roberta_functions import *
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from transformers import RobertaForSequenceClassification, Trainer
import wandb
import pickle
import numpy as np
from sklearn.metrics import f1_score

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/Stormfront/original/Stormfront_train_undersampled.csv")
synth_ds_bin = load_and_tokenize_synthetic_set("my_datasets/Stormfront/SF_synthetic_undersampled.csv")


wandb.init(project="SF-composite-roberta-staged")  # Initialize W&B before the loop

metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the model here
for i in range(4,5):
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
    metrics[i] = trainer.evaluate()
    print(metrics)
    metrics_df = pd.DataFrame.from_dict(metrics).transpose()
    metrics_df.to_csv("Logs/SF-composite-roberta-staged_metrics.csv")

trainer.save_model("models/SF-composite-roberta-staged")