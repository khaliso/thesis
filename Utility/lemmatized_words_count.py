import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the CSV file into a DataFrame
df = pd.read_csv('my_datasets/Founta/founta_synth_cleaned.csv')  # Replace 'your_file.csv' with the path to your file
df = df.dropna(subset=['text'])

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Tokenize, remove stopwords, and lemmatize the text
all_words = []
for text in df['text']:
    text = str(text)  # Ensure that the value is a string
    words = word_tokenize(text.lower())  # Convert text to lowercase and tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    all_words.extend(words)

# Get unique lemmatized words
unique_lemmatized_words = set(all_words)

# Print the count of unique lemmatized words
print(f"Total number of unique, lemmatized words: {len(unique_lemmatized_words)}")
