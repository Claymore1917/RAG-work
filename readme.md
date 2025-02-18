# RAG System with Hugging Face and FAISS

This project sets up a Retrieval-Augmented Generation (RAG) pipeline that answers questions based on context extracted from a PDF document. The system leverages LangChain and Hugging Face models with FAISS for efficient document retrieval. The example focuses on the Korean singer Kim Chungha, but can be adapted to any domain.

## Prerequisites

Ensure that you have the following Python libraries installed:
- langchain
- langchain_huggingface
- langchain_community
- faiss
- sentence-transformers
- getpass
- huggingface_hub

You can install them via pip:
```bash
pip install langchain langchain-huggingface langchain-community faiss-cpu sentence-transformers huggingface_hub
```

## Step-by-Step Setup

### 1. Input Your Hugging Face API Token

First, you'll need a Hugging Face API token to access Hugging Face models. Enter your token using the `getpass` function:

```python
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass("Enter your Hugging Face API Token: ")
```

### 2. Set Up the Language Model

We’ll use the `Qwen2.5-72B-Instruct` model for generating responses. You can modify this to use a different model as needed. Set the token for Hugging Face's API:

```python
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
```

Then initialize the model with `HuggingFaceEndpoint`:

```python
repo_id = "Qwen/Qwen2.5-72B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens=2000
)

llm = ChatHuggingFace(llm=llm)
```

### 3. Test the Language Model

Create a simple template for asking questions:

```python
template = """Answer the question below:
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
llm_chain = prompt | llm | output_parser

# Test the model
question = "Who is KIM Chungha?"
response = llm_chain.invoke({"question": question})
print(response)
```

### 4. Set Up the Knowledge Base

For document retrieval, we’ll use a PDF file (`Chungha.pdf`) and an embedding model (`sentence-transformers/all-MiniLM-l6-v2`) to convert documents into vector embeddings. Then, we store the embeddings in a FAISS index.

Load and split the PDF:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Chungha.pdf")
docs = loader.load_and_split()
```

Generate embeddings for the documents:

```python
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector_store = FAISS.from_documents(documents, embeddings)
```

### 5. Create the RAG Pipeline

We’ll create a document chain and a retriever chain for generating responses from the retrieved documents:

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context:

Question: {input}

Context: {context}
""")

# Create the document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

### 6. Generate Responses

Define a function to generate responses based on user queries:

```python
def generate_response(query):
    response = retrieval_chain.invoke({"input": query})
    return response["answer"].strip()
```

### 7. Test the RAG System

Finally, test the system by asking a question. Here, we query about the release date of the song "Roller Coaster":

```python
query = "When did Roller Coaster come out?"
print(generate_response(query))
```
