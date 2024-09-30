import pinecone
from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.legacy.vector_stores import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.readers.file import UnstructuredReader
from nltk.corpus.reader import documents
from pinecone import Pinecone, ServerlessSpec
from openai import api_key

load_dotenv()
# pinecone.init(
#     api_key=os.environ["PINECONE_API_KEY"],
#     environment=os.environ["PINECONE_ENVIRONMENT"])

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if __name__ == "__main__":
    print("*****")
    print("Going to ingest pinecone documentation ...")
    UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        file_extractor={".html": UnstructuredReader()},
    )
    documents = dir_reader.load_data()
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    # embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    # node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    # nodes = node_parser.get_nodes_from_documents(documents=documents)

    # service_context = ServiceContext.from_defaults(
    #     llm=llm, embed_model=embed_model, node_parser=node_parser
    # )

    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )
    Settings.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=500, chunk_overlap=20
    )
    nodes = Settings.node_parser.get_nodes_from_documents(documents=documents)

    pinecone_index = pinecone.Index(
        index="llamaindex-doc-helper-tmp",
        api_key=os.environ.get("PINECONE_API_KEY"),
        host="https://llamaindex-doc-helper-fxwsgnk.svc.aped-4627-b74a.pinecone.io",
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        # service_context=service_context,
        show_progress=True,
    )

    print("finished ingesting ...")
