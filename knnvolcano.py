import os
import numpy as np
from statistics import mode
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import colorama
from colorama import Fore

# Initialize colorama
colorama.init(autoreset=True)

# --- Your OpenAI key ---
os.environ["OPENAI_API_KEY"] =  ""


# ========= Fast KNN Functions =========
def random_compress(x_train, x_test, new_dimension):
    k = x_train.shape[1]
    w = np.random.normal(size=(k, new_dimension))
    x_train = x_train @ w
    x_test = x_test @ w
    return x_train, x_test, w

def knn_retrieve(x_train, y_train, query_vec, n_neighbors=4):
    distances = np.linalg.norm(x_train - query_vec, axis=1)
    top_idx = np.argsort(distances)[:n_neighbors]
    return top_idx, distances[top_idx]

# ========= Load and Split PDF =========
pdfpath = "1.pdf"
query = "Where were samples extracted from?"

pdf_loader = PyPDFLoader(pdfpath)
docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# ========= Embed all chunks =========
embedder = OpenAIEmbeddings()
chunk_embeddings = np.array([embedder.embed_query(doc.page_content) for doc in documents])

# ========= Random projection to speed up KNN =========
new_dim = 128  # compressed dimension
compressed_embeddings, _, projection_matrix = random_compress(chunk_embeddings, chunk_embeddings, new_dim)

# ========= Embed query and compress =========
query_embedding = np.array(embedder.embed_query(query))
query_compressed = query_embedding @ projection_matrix

# ========= Retrieve Top-K using Fast KNN =========
top_idx, top_distances = knn_retrieve(compressed_embeddings, None, query_compressed, n_neighbors=4)
top_chunks = [documents[i] for i in top_idx]

# ========= Build prompt =========
context = "\n\n---\n\n".join([doc.page_content for doc in top_chunks])
prompt = f"""Use the following context to answer the question. 
If the answer is not contained within the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:"""

# ========= Call GPT =========
client = OpenAI()
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
print(Fore.YELLOW + query + " " + answer)
