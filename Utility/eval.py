from hatebert_functions import load_predict_testset, compute_test_metrics, TRAINING_ARGS

test_filepath = "your/file/path.csv"
model_path = "your/file/path" 


y_pred = load_predict_testset(test_filepath, model_path=model_path, args=TRAINING_ARGS)

print(compute_test_metrics(y_pred, 'macro'))

# 