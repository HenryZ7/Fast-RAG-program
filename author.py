from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
import colorama
from colorama import Fore
import os
import numpy as np
# Initialize colorama for Windows
colorama.init(autoreset=True)

# Hardcoding the OpenAI API key
os.environ["OPENAI_API_KEY"] =  ""


pdfpath = "0.pdf"
query = "What are the authors of the paper? "

# Load PDF
pdf_loader = PyPDFLoader(pdfpath)
docs = pdf_loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
documents = text_splitter.split_documents(docs)


# Embed and store in FAISS
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(documents, embeddings)
# Create retriever and QA chain
retriever = vector_db.as_retriever()

# initialize client (will read OPENAI_API_KEY from env)
client = OpenAI()
# number of top chunks to retrieve
TOP_K = 4 
# --- 1. Retrieve relevant chunks manually ---
retrieved_docs = retriever.get_relevant_documents(query)  # from your vector store
top_chunks = retrieved_docs[:TOP_K]  # simple top-K
# --- 2. Build the prompt ---
# Concatenate the retrieved context, with a separator and the question.
context = "\n\n---\n\n".join([doc.page_content for doc in top_chunks])

prompt = f"""Use the following context to answer the question. 
If the answer is not contained within the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:"""

# call new Chat Completions API
response = client.chat.completions.create(
    model="gpt-4",  # or another model you have access to
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers based only on provided context."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.0,
    max_tokens=5000,
)

answer = response.choices[0].message.content.strip()
# print("Answer:", answer)



# response = qa_chain.run(query)
print(Fore.YELLOW + query + answer)


