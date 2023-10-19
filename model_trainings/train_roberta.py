from roberta_functions import *
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from transformers import RobertaForSequenceClassification, Trainer
import wandb
import pickle
import numpy as np
from sklearn.metrics import f1_score

# Prepare the datasets
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/dontpatronizeme_v1.4/train_data.csv")

wandb.init(project="RoBERTa_PCL-3")  # Initialize W&B before the loop

metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the model here
for i in range(4,5):
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    trainer = Trainer(model=model, args=TRAINING_ARGS, train_dataset=trains_ds_bin[i], eval_dataset=vals_ds_bin[i], compute_metrics=compute_metrics)
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
    metrics_df.to_csv("Logs/RoBERTa_PCL_3_metrics.csv")

trainer.save_model("models/RoBERTa_PCL_3")

