import os
from dotenv import load_dotenv
load_dotenv()

import pinecone
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.legacy.vector_stores import PineconeVectorStore

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if __name__ == "__main__":
    print("RAG ...")

    pinecone_index = pinecone.Index(
        index="llamaindex-doc-helper",
        api_key=os.environ.get("PINECONE_API_KEY"),
        host="https://llamaindex-doc-helper-fxwsgnk.svc.aped-4627-b74a.pinecone.io",
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    query = "What is LlamaIndex query engine?"
    query_engine = index.as_query_engine()

    response = query_engine.query(query)



