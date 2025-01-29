import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import string

def clean_text(text):
    """Removes extra whitespace, special characters, and normalizes text."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.lower()  # Convert to lowercase for consistency

def remove_stopwords(text, stopwords=None):
    """Removes common stopwords from text."""
    if stopwords is None:
        stopwords = {"the", "is", "in", "and", "to", "a", "of", "for", "on", "with", "by", "an"}  # Minimal stopword list
    return " ".join([word for word in text.split() if word not in stopwords])

def lemmatize_text(text):
    """Performs basic lemmatization (placeholder for actual NLP library like spaCy or NLTK)."""
    return text  # Implement lemmatization if needed

def process_text(text):
    """Applies a sequence of text processing functions."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def chunk_text(file_path, chunk_size=300, overlap=50):
    """Reads a text file, processes it, and splits it into chunks."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = process_text(file.read())

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

if __name__ == "__main__":
    file_path = "sample_data.txt"  # Replace with your actual text file
    text_chunks = chunk_text(file_path)

    for i, chunk in enumerate(text_chunks[:3]):  # Display first few chunks
        print(f"Chunk {i+1}: {chunk[:200]}...\n")