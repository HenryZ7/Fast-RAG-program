# Fast-RAG-program
This program is a take on a basic RAG program that has multiple versions. The program authors.py is the most barebones, where the Faiss library is used to find chunks of relevant information. Based on those chunks, the program will return the authors of the document. Originally, at a smaller chunk size, the accuracy was lower and some authors were omitted. 

The second program, volcano.py, has two iterations: one that uses a fast k-nearest-neighbor program when sorting chunks, and one where it uses the Faiss library. The accuracy of both is about the same. Just like the first program, it will select chunks of relevent information and answer whatever questions about the article the user may have. 

NOTE: For this program to work, the user must have an Openai API key.


