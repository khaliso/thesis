# Load the classifier selected for this dataset: BERT undersampled, 'f1': 0.6891402961392428,
import csv
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Input CSV file
input_file = "my_datasets/Davidson/new_Davidson_non_hateful_synthetic_data.csv"
output_file = "my_datasets/Davidson/trunc_Davidson_non_hateful_synthetic_data.csv"

# Open the input CSV file for reading and the output CSV file for writing
with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # Iterate over each row in the input CSV file
    for row in reader:
        text = row[0].strip()

        # Truncate the text to a maximum of 512 tokens
        encoded_input = tokenizer.encode_plus(text, max_length=507, truncation=True, padding='max_length')
        truncated_text = tokenizer.decode(encoded_input['input_ids'], skip_special_tokens=True)

        # Write the truncated text to the output CSV file
        writer.writerow([truncated_text])


# Load the pre-trained model using pipeline
model = "experiment/pcl_roberta"  # Pre-trained model
pl = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Load and classify the dataset
dataset_path = 'my_datasets/Davidson/trunc_Davidson_non_hateful_synthetic_data.csv'  # hate and non-hate synthetic dataset here
output_file = 'my_datasets/Davidson/classified_trunc_Davidson_non_hateful_synthetic_data.csv'  # Specify the path for the output CSV file

with open(dataset_path, 'r') as f, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(f)
    writer = csv.writer(outfile)

    # Write header to the output CSV file
    writer.writerow(['Text', 'Label', 'Score'])

    for line in reader:
        text = line[0].strip()
        result = pl(text)
        label = result[0]['label']
        score = result[0]['score']

        # Write text, label, and score to the output CSV file
        writer.writerow([text, label, score])