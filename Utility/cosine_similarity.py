import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def compute_within_similarity(df, label):
    # Filter the data for the given label
    label_data = df[df['labels'] == label]['text']

    # Handle NaN values and non-string data
    label_data = label_data.dropna().astype(str).tolist()

    if not label_data:
        # Handle cases where there is no valid data for the label
        return 0.0  # You can choose to return 0 or another suitable value

    # Compute sentence embeddings
    embeddings = model.encode(label_data)

    # Convert embeddings to numpy arrays (if they aren't already)
    if isinstance(embeddings[0], torch.Tensor):
        embeddings = [embedding.cpu().numpy() for embedding in embeddings]

    # Compute cosine similarities
    similarities = []
    total_comparisons = len(embeddings) * (len(embeddings) - 1) // 2
    with tqdm(total=total_comparisons, desc=f"Within Similarity - Label {label}") as pbar:
        for i in range(len(embeddings) - 1):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))
                similarities.append(similarity[0][0])
                pbar.update(1)

    avg_similarity = sum(similarities) / len(similarities)
    result_key = f"Within Similarity - Label {label}"
    print(f"{result_key}: {avg_similarity}")
    return avg_similarity

def compute_between_similarity(df1, df2, label):
    label_data_1 = df1[df1['labels'] == label]['text']
    label_data_2 = df2[df2['labels'] == label]['text']

    # Handle NaN values and non-string data for label_data_1
    label_data_1 = label_data_1.dropna().astype(str).tolist()

    if not label_data_1:
        # Handle cases where there is no valid data for label_data_1
        return 0.0  # You can choose to return 0 or another suitable value

    # Handle NaN values and non-string data for label_data_2
    label_data_2 = label_data_2.dropna().astype(str).tolist()

    if not label_data_2:
        # Handle cases where there is no valid data for label_data_2
        return 0.0  # You can choose to return 0 or another suitable value

    embeddings_1 = model.encode(label_data_1)
    embeddings_2 = model.encode(label_data_2)

    if isinstance(embeddings_1[0], torch.Tensor):
        embeddings_1 = [embedding.cpu().numpy() for embedding in embeddings_1]
    if isinstance(embeddings_2[0], torch.Tensor):
        embeddings_2 = [embedding.cpu().numpy() for embedding in embeddings_2]

    # Compute cosine similarities
    similarities = []
    with tqdm(total=len(embeddings_1) * len(embeddings_2), desc=f"Between Similarity - Label {label}") as pbar:
        for emb_1 in embeddings_1:
            for emb_2 in embeddings_2:
                similarity = cosine_similarity(emb_1.reshape(1, -1), emb_2.reshape(1, -1))
                similarities.append(similarity[0][0])
                pbar.update(1)

    avg_similarity = sum(similarities) / len(similarities)
    result_key = f"Between Similarity - Label {label}"
    print(f"{result_key}: {avg_similarity}")
    return avg_similarity

# Load your datasets
df_original = pd.read_csv('path/to/original.csv')
df_synthetic = pd.read_csv('path/to/synthetic.csv')

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Compute average cosine similarity for each scenario
results = {
    "Orig - Label 0": compute_within_similarity(df_original, 0),
    "Orig - Label 1": compute_within_similarity(df_original, 1),
    "Synth - Label 0": compute_within_similarity(df_synthetic, 0),
    "Synth - Label 1": compute_within_similarity(df_synthetic, 1),
    "Between Orig & Synth - Label 0": compute_between_similarity(df_original, df_synthetic, 0),
    "Between Orig & Synth - Label 1": compute_between_similarity(df_original, df_synthetic, 1)
}

for key, val in results.items():
    print(f"{key}: {val}")

