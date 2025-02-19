from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer
from decouple import config

PINECONE_API_KEY = config("PINECONE_API_KEY")


pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

INDEX_NAME = "product-recommendation"
if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load cleaned dataset
df = pd.read_csv("./dataset/cleaned_dataset.csv")

# Initialize text embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and effective for embeddings

# Generate embeddings for product descriptions
df["vector"] = df["Description"].apply(lambda text: model.encode(text).tolist())

# Prepare data for Pinecone (tuples of id, vector, metadata)
data_to_upsert = [
    (str(row["StockCode"]), row["vector"], {"Description": row["Description"], "UnitPrice": row["UnitPrice"]})
    for _, row in df.iterrows()
]

# Upload to Pinecone
index.upsert(vectors=data_to_upsert)

print("setup complete! Product vectors stored.")
