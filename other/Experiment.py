import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# Load and preprocess the data
df = pd.read_csv('my_datasets/Stormfront/original/training_big.csv')
labels = df['labels'].astype(int)  # Convert string labels to integers
texts = df['text'].tolist()

# Tokenize the texts
tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT')
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Create a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, encoded_inputs, labels):
        self.input_ids = encoded_inputs['input_ids']
        self.attention_mask = encoded_inputs['attention_mask']
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

dataset = CustomDataset(encoded_inputs, labels)

# Create a data loader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the BERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained('GroNLP/hateBERT', num_labels=2)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    print(f'Epoch: {epoch+1}, Average Loss: {average_loss}')

# Save the trained model
model.save_pretrained('experiment/SF_hatebert')
