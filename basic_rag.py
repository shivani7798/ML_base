import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set API key
os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"

# STEP 1: Create knowledge base
with open("my_knowledge.txt", "w") as f:
    f.write("""
RAG (Retrieval Augmented Generation) is a technique that combines 
large language models with external knowledge retrieval.

The three main stages of RAG are:
1. Indexing - Convert documents into searchable format
2. Retrieval - Find relevant documents for a query
3. Generation - Use LLM to generate answer from retrieved docs

RAG is used for chatbots, Q&A systems, and knowledge assistants.
It helps LLMs access private or recent information they weren't trained on.
""")

# STEP 2: Load documents
loader = TextLoader("my_knowledge.txt")
documents = loader.load()

# STEP 3: Split and index
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

# STEP 4: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# STEP 5: Build RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:
""")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions!
questions = [
    "What are the three stages of RAG?",
    "What is RAG used for?",
    "How does RAG help LLMs?"
]

for q in questions:
    answer = rag_chain.invoke(q)
    print(f"\n❓ {q}")
    print(f"✨ {answer}")
