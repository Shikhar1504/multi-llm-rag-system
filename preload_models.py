import os
from sentence_transformers import SentenceTransformer, CrossEncoder

def preload_models():
    print("Downloading SentenceTransformer: all-MiniLM-L6-v2")
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Downloading CrossEncoder: ms-marco-MiniLM-L-6-v2")
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("Models successfully preloaded!")

if __name__ == "__main__":
    preload_models()
