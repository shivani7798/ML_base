from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load and split documents
docs = loader.load()
splits = text_splitter.split_documents(docs)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)
