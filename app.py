from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import fitz
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from dotenv import load_dotenv
from langchain_community.llms import GooglePalm
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MODEL_NAME'] = 'distilbert-base-uncased-distilled-squad'
app.config['MAX_SEQUENCE_LENGTH'] = 512  # Define max sequence length

# FAISS setup
embeddings = HuggingFaceInstructEmbeddings()
vector_db_path = "/Users/rafaelzieganpalg/Projects/Cognizant AI Bot/New_Faiss_Index"  # Update this path!
vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.8)
prompt_template = """Given the following context and a question, generate an answer based on this context.
                    In the answer, try to provide some text as possible from the "text" section in the source 
                    document context without making much changes. If the answer is not found in the context, 
                    kindly try to make up an answer but dont mention it as this document does not contain the answer.
                    CONTEXT: {context}
                    QUESTION: {question}"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

faiss_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class PDFQueryBot:
    def __init__(self, pdf_path, model_name):
        self.pdf_path = pdf_path
        self.text = self.extract_text_from_pdf()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def extract_text_from_pdf(self):
        text = ""
        pdf_document = fitz.open(self.pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        return text

    def query(self, question):
        inputs = self.tokenizer(
            question,
            self.text,
            return_tensors='pt',
            truncation=True,
            max_length=app.config['MAX_SEQUENCE_LENGTH']
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        answer_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)

        return answer if answer else "No answer found."


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return fitz.redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                return jsonify({'filename': filename})
    return render_template('index.html')


@app.route('/query_pdf', methods=['POST'])
def query_pdf():
    try:
        question = request.form['question']
        filename = request.form['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        bot = PDFQueryBot(filepath, app.config['MODEL_NAME'])
        answer = bot.query(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error as JSON with 500 status code


@app.route('/query_faiss', methods=['POST'])
def query_faiss():
    try:
        query = request.form['query']
        response = faiss_qa_chain({"query": query})
        return jsonify({'response': response['result']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error as JSON with 500 status code


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)