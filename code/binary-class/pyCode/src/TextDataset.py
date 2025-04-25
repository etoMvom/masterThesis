from Bio import Entrez
from Bio import Medline
import re

from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

Entrez.email = "etowilfried1@gmail.com"

def search_pubmed(query, retmax=5000):
    """Search PubMed and return article IDs."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_abstracts(id_list):
    """Retrieve article abstracts."""
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
    records = Medline.parse(handle)
    abstracts = []

    for record in tqdm(records, desc="Fetching PubMed abstracts", total=len(id_list)):
        abstracts.append({
            "PMID": record.get("PMID", ""),
            "Title": record.get("TI", ""),
            "Abstract": record.get("AB", ""),
            "Keywords": record.get("OT", []),
            "PublicationYear": record.get("DP", "").split()[0],
            "MeSH_Terms": record.get("MH", []),
        })

    handle.close()
    return abstracts

def clean_text(text):
    """Clean text by removing special characters."""
    text = re.sub(r'[^\w\s><=,%]', ' ', text)
    return text.lower().strip()

def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing, tokenizing, removing stop words, and lemmatizing.

    Args:
        text: A string containing the text to be preprocessed.

    Returns:
        A string containing the processed text with stopwords removed and words lemmatized.
    """
    if pd.isna(text):
        return ""
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

class PubMedBERTDataset(Dataset):
    """
    Dataset class for multi-class classification using PubMedBERT embeddings.

    Args:
        texts (list of str): A list of raw text sequences.
        labels (list of int): A list of integer labels (0-8) corresponding to each text.
        tokenizer (transformers tokenizer): A tokenizer to preprocess the text data.
        max_length (int): Maximum length for padding/truncation. Default is 400.

    Attributes:
        texts (list of str): The raw text data.
        encodings (dict): Tokenized text data with input IDs and attention masks.
        labels (Tensor): Tensor of integer labels for the classification task.
    """

    def __init__(self, texts, labels, tokenizer, max_length=400):
        self.texts = texts
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Contains input IDs, attention masks, and the corresponding label and raw text.
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['text'] = self.texts[idx]
        return item

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

def load_glove_embeddings(glove_path, word_to_index, embedding_dim):
    """Loads GloVe embeddings and returns an embedding matrix."""
    embedding_matrix = np.zeros((len(word_to_index), embedding_dim), dtype='float32')
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            if word in word_to_index:
                vector = np.asarray(values[1:], dtype='float32')
                embedding_matrix[word_to_index[word]] = vector

    return torch.tensor(embedding_matrix, dtype=torch.float)
