import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class FaissVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the embedding model and Faiss index."""
        self.model = SentenceTransformer(model_name)
        self.index = None

    def embed_texts(self, text_chunks):
        """Converts text chunks into vector embeddings."""
        try:
            embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error embedding texts: {e}")
            return None

    def create_faiss_index(self, embeddings):
        """Creates a Faiss index for efficient similarity search."""
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings available to create index.")
            return None
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance metric
        self.index.add(embeddings)
        return self.index

    def store_embeddings(self, embeddings, text_chunks, index_path="vector_store.index", metadata_path="metadata.pkl"):
        """Stores embeddings in Faiss and saves metadata."""
        if self.create_faiss_index(embeddings) is None:
            print("Failed to create Faiss index. Skipping storage.")
            return

        # Save Faiss index
        faiss.write_index(self.index, index_path)
        print(f"Faiss index saved to {index_path}")

        # Save metadata (original text chunks)
        with open(metadata_path, "wb") as f:
            pickle.dump(text_chunks, f)
        print(f"Metadata saved to {metadata_path}")

    def load_faiss_index(self, index_path="vector_store.index", metadata_path="metadata.pkl"):
        """Loads the Faiss index and metadata from storage."""
        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                text_chunks = pickle.load(f)
            print("Faiss index and metadata loaded successfully.")
            return text_chunks
        except Exception as e:
            print(f"Error loading Faiss index or metadata: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    sample_chunks = ["This is an example chunk.", "Another chunk of text about AI."]
    faiss_store = FaissVectorStore()

    # Generate and store embeddings
    sample_embeddings = faiss_store.embed_texts(sample_chunks)
    if sample_embeddings is not None:
        faiss_store.store_embeddings(sample_embeddings, sample_chunks)

    # Load stored embeddings for verification
    loaded_texts = faiss_store.load_faiss_index()
    print(f"Loaded Text Chunks: {loaded_texts}")