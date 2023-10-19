# What's Where?

## To run a fine-tune, there are a number of different options in 'model' trainings:
  - train_[model].py
  - train_composite_[model].py
  - synthetic_train.py

'train_[model].py' are the baseline trainings
'train_composite_[model].py' are the composite trainings including validation set wizardry
'synthetic_train.py' are the synthetic-only trainings, including validation set wizardry
    In this one, make sure to include the corresponding 'functions.py' and specify your tokenizer! 
    
The SMOTE trainings are conducted on a standard 'train_[model].py', just prep your dataset.

## My model is trained. How do I evaluate it?

Under 'Utility', run 'eval.py' after enter your model and testset filepaths. 
Make sure to link to the correct 'functions.py' file for the model you want to evaluate!
