import os
import tempfile
from flask import Flask, request, render_template_string
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
PERSIST_DIRECTORY = os.path.join("data", "vectors")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

vector_db = None

def create_vector_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = splitter.split_documents(data)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)

def process_question(question, vector_db, selected_model):
    llm = ChatOllama(model=selected_model)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}"
    )
    chain = (
        {"context": vector_db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

HTML_TEMPLATE = """
<!doctype html>
<title>Ollama PDF Q&A</title>
<h2>Upload a PDF and Ask Questions</h2>
<form method=post enctype=multipart/form-data>
  <p><input type=file name=pdf>
     <input type=text name=question placeholder="Ask a question">
     <input type=text name=model value="llama3" placeholder="Model name">
     <input type=submit value=Ask>
</form>
{% if answer %}
<h3>Answer:</h3>
<p>{{ answer }}</p>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    global vector_db
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        model = request.form['model']
        file = request.files['pdf']
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            vector_db = create_vector_db(tmp_path)
        if question and vector_db:
            answer = process_question(question, vector_db, model)
    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
