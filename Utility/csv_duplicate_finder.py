import pandas as pd

def count_common_text_entries(file1, file2):
    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Get the 'text' entries in both files
    text_entries_file1 = set(df1['text'])
    text_entries_file2 = set(df2['text'])

    # Find the common 'text' entries
    common_text_entries = text_entries_file1.intersection(text_entries_file2)

    # Get the number of common 'text' entries
    num_common_text_entries = len(common_text_entries)

    return num_common_text_entries

# Specify the paths to the CSV files
file1_path = 'my_datasets/GermEval/GE_train_original.csv'
file2_path = 'my_datasets/GermEval/synthetic/composite_GE_dataset.csv'

# Call the function to count the common 'text' entries
common_text_entries_count = count_common_text_entries(file1_path, file2_path)

print(f"Number of 'text' entries contained in both files: {common_text_entries_count}")