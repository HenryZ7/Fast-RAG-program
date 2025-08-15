import os
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import colorama
from colorama import Fore

# ==================== Setup ====================
colorama.init(autoreset=True)
# insert openai api key
os.environ["OPENAI_API_KEY"] =  ""

# ========= Fast KNN with Random Projection =========
def random_compress_matrix(k, new_dimension, seed=None):
    """Create and return a random projection matrix."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(size=(k, new_dimension))

def compress(vectors, projection_matrix):
    """Project high-dim vectors into lower-dim space."""
    return vectors @ projection_matrix

def knn_retrieve_batch(x_train, queries, top_k=4):
    """
    Vectorized batch retrieval.
    x_train: (num_docs, dim)
    queries: (num_queries, dim)
    Returns: list of arrays of top_k indices for each query
    """
    # Compute pairwise distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
    train_norms = np.sum(x_train**2, axis=1)  # (num_docs,)
    query_norms = np.sum(queries**2, axis=1)  # (num_queries,)

    # distances squared
    dists = (
        query_norms[:, None]  # (num_queries, 1)
        + train_norms[None, :]  # (1, num_docs)
        - 2 * queries @ x_train.T  # (num_queries, num_docs)
    )

    # Get top_k smallest distances per query
    top_indices = np.argpartition(dists, top_k, axis=1)[:, :top_k]

    # Sort the top_k by actual distance
    sorted_top_indices = np.take_along_axis(
        top_indices,
        np.argsort(np.take_along_axis(dists, top_indices, axis=1), axis=1),
        axis=1
    )

    return sorted_top_indices

# ==================== Load and Chunk PDF ====================
pdfpath = "1.pdf"
query_list = [
    "Where were samples extracted from?"
]

pdf_loader = PyPDFLoader(pdfpath)
docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# ==================== Embed All Chunks ====================
embedder = OpenAIEmbeddings()
chunk_embeddings = np.array([embedder.embed_query(doc.page_content) for doc in documents])

# ==================== Random Projection ====================
original_dim = chunk_embeddings.shape[1]  # 1536 for OpenAI
new_dim = 128
projection_matrix = random_compress_matrix(original_dim, new_dim, seed=42)

compressed_chunks = compress(chunk_embeddings, projection_matrix)
print(f"Original chunk embedding shape: {chunk_embeddings.shape}")   # (num_chunks, 1536)
print(f"Compressed chunk embedding shape: {compressed_chunks.shape}")  # (num_chunks, 128)

# ==================== Embed & Compress Queries ====================
query_embeddings = np.array([embedder.embed_query(q) for q in query_list])
compressed_queries = compress(query_embeddings, projection_matrix)
print(f"Original query embedding shape: {query_embeddings.shape}")   # (num_queries, 1536)
print(f"Compressed query embedding shape: {compressed_queries.shape}")  # (num_queries, 128)

# ==================== Retrieve Top-K for All Queries ====================
top_k = 4
retrieved_indices = knn_retrieve_batch(compressed_chunks, compressed_queries, top_k=top_k)

# ==================== Show Retrieved Chunks ====================
for q_idx, query in enumerate(query_list):
    print(f"\n=== Query: {query} ===")
    for rank, idx in enumerate(retrieved_indices[q_idx], start=1):
        snippet = documents[idx].page_content.replace("\n", " ")[:150]  # shorten for display
        print(f"  Chunk {rank} (Index {idx}): {snippet}...")

# ==================== Build Prompts & Call GPT ====================
client = OpenAI()

for q_idx, query in enumerate(query_list):
    top_chunks = [documents[i] for i in retrieved_indices[q_idx]]
    context = "\n\n---\n\n".join([doc.page_content for doc in top_chunks])

    prompt = f"""Use the following context to answer the question. 
If the answer is not contained within the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based only on provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()
    print(Fore.YELLOW + f"Q: {query}\nA: {answer}\n")
