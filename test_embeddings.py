
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample sentences
sentences = [
    "How can duplicate data be avoided?",
    "Normalization reduces data redundancy."
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

print("Number of sentences:", len(embeddings))
print("Vector dimensions:", embeddings.shape)
print("Embedding generated successfully!")
