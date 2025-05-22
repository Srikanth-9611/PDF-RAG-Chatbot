import os
import pdfplumber
import tempfile
import shutil
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# Constants
PERSIST_DIRECTORY = os.path.join("data", "vectors")
EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3"

def load_pdf_and_create_vectorstore(pdf_path: str) -> Chroma:
    print(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"pdf_{hash(pdf_path)}"
    )
    return vectordb

def run_rag(question: str, vector_db: Chroma, model_name: str) -> str:
    print(f"\nRunning query on model: {model_name}")
    llm = ChatOllama(model=model_name)

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Rephrase the following question in 2 different ways to improve document retrieval:\nOriginal question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=query_prompt
    )

    # Updated prompt for structured answer
    final_prompt = ChatPromptTemplate.from_template(
        """Use only the context below to answer the question. 
Format the response in concise bullet points or a numbered list.

Context:
{context}

Question: {question}

Format:
- If the answer includes multiple facts, split them into clear, short bullet points.
- Be brief and specific. Avoid unnecessary explanation.
- If the answer is a list or steps, use a numbered format.
"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | final_prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

def main():
    print("🧠 Ollama PDF RAG CLI Tool")
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print("❌ File does not exist.")
        return

    model_name = input(f"Enter the Ollama model to use [default: {DEFAULT_LLM_MODEL}]: ").strip() or DEFAULT_LLM_MODEL

    try:
        vectordb = load_pdf_and_create_vectorstore(pdf_path)

        while True:
            question = input("\nAsk a question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                print("Exiting...")
                break
            answer = run_rag(question, vectordb, model_name)
            print("\n🤖 Answer:\n", answer)

    except Exception as e:
        print("Error:", str(e))

if __name__ == "_main_":
    main()