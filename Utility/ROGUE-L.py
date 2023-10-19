import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

def compute_avg_rouge_l(df_original, df_synthetic, label):
    # Filter and clean the data for the given label
    original_texts = df_original[df_original['labels'] == label]['text'].dropna().astype(str).tolist()
    synthetic_texts = df_synthetic[df_synthetic['labels'] == label]['text'].dropna().astype(str).tolist()
    
    if not original_texts or not synthetic_texts:
        return 0.0

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Compute ROUGE-L scores for pairs of original and synthetic texts
    scores = []
    for orig_text, synth_text in tqdm(zip(original_texts, synthetic_texts), total=min(len(original_texts), len(synthetic_texts)), desc=f"ROUGE-L - Label {label}"):
        score = scorer.score(orig_text, synth_text)
        scores.append(score['rougeL'].fmeasure)  # Using F1-score of ROUGE-L
    
    # Compute and return the average ROUGE-L score
    avg_rouge_l = sum(scores) / len(scores) if scores else 0.0
    print(f"Average ROUGE-L - Label {label}: {avg_rouge_l:.4f}")
    return avg_rouge_l


# Load your datasets
df_original = pd.read_csv('path/to/original.csv')
df_synthetic = pd.read_csv('path/to/synthetic.csv')

compute_avg_rouge_l(df_original, df_synthetic, 0)
compute_avg_rouge_l(df_original, df_synthetic, 1)
