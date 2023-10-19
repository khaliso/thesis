import warnings
from sklearn.exceptions import UndefinedMetricWarning
import wandb
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from AlBERT_functions import *


# Prepare undersampled dataset
vals_ds_bin, trains_ds_bin = load_and_tokenize_training_set("my_datasets/HatEval/merged_train.csv")

wandb.init(project="HatEval-AlBert-base")  # Initialize W&B before the loop

metrics = {}

# Depending on the model we want to train (and which ones tokenizer we've used) we change the BERT model here
for i in range(4,5):
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)
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
    metrics_df.to_csv("Logs/HatEval_albert_base_metrics.csv")
    #with open(f"Logs/Founta/GE_Albert_metrics_fold_{i}.pickle", "wb") as file:
    #    pickle.dump(metrics[i], file)
        
trainer.save_model("models/HatEval_albert_base")

# Gotta downgrade protobuf for this to work
#pip install protobuf==3.20.x
