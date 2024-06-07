import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def get_embedding():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

if __name__ == "__main__":
    embeddings = get_embedding()
    print(embeddings.embed_query("Hello, how are you?"))